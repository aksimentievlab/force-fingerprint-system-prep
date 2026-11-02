#!/usr/bin/env python3
"""
gcmc_hydrate.py - fill buried cavities of a solvated CHARMM system with water by
grand canonical Monte Carlo (grand + OpenMM). Replaces the Dowser step of
af_prep.py and runs AFTER the psfgen build + solvation, so the final force field,
protonation states and built cofactors decide where water goes.

Steps:
  1. Load PSF/coordinates/toppar with OpenMM (CharmmPsfFile).
  2. Detect buried cavities on a grid (probe distance + ray-casting burial test),
     cluster them, one GCMC sphere per cavity.
  3. Borrow bulk waters far from the solute as grand "ghost" waters.
  4. Restrained minimisation, then for each cavity: GCMC moves interleaved
     with restrained NVT MD (solute heavy atoms restrained).
  5. Strip waters that end as ghosts; write PSF/PDB with ParmEd.

Requirements: openmm, parmed, numpy, scipy, grand (github.com/essex-lab/grand)
with its dependencies mdtraj and openmmtools.

Example:
  python gcmc_hydrate.py --psf build/model_ion.psf --pdb build/model_ion.pdb \
      --box-file build/model_ion_box.txt \
      --toppar top_all36_prot.rtf par_all36m_prot.prm toppar_water_ions.str \
      --out model_hyd --drop-nbfix

Notes:
  * grand rebuilds Lennard-Jones with standard combining rules and cannot handle
    CHARMM NBFIX terms. With --drop-nbfix they are ignored during water placement
    only; the PSF written here is unchanged and your production run uses NBFIX.
  * The default excess chemical potential (-6.09 kcal/mol) and standard volume
    (30.345 A^3) are grand's values for TIP3P; CHARMM's TIP3P (with H LJ) differs
    slightly. grand.potential.calc_mu_ex / calc_std_volume can compute exact
    values if occupancies of marginal sites matter to you.
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import openmm as mm
import openmm.app as app
import openmm.unit as u

log = logging.getLogger("gcmc_hydrate")

WATER_RES = {"TIP3", "HOH", "WAT", "SOL", "TIP3P", "SPC"}
BULK_ION_RES = {"SOD", "CLA", "POT", "CES", "LIT", "RUB", "NA", "CL", "K"}
KCAL_A2 = 418.4  # kJ/mol/nm^2 per kcal/mol/A^2


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def read_pdb_coords(path):
    """Coordinates in file order, read straight from ATOM/HETATM lines.
    OpenMM's PDB reader groups atoms by chain/residue and ignores segment IDs, so
    psfgen output (chain = first letter of segid) can merge distinct one-atom
    residues such as several ZN2 segments; the PSF order is what matters here."""
    xyz, cryst = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                cryst = [float(line[6:15]), float(line[15:24]), float(line[24:33])]
            elif line.startswith(("ATOM", "HETATM")):
                xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return u.Quantity(np.array(xyz) / 10.0, u.nanometer), cryst


def read_box(args, cryst):
    if args.box:
        return [float(x) for x in args.box]
    if args.box_file:
        p = Path(args.box_file)
        lines = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
        f = lines[-1].split()
        if p.suffix == ".xsc":          # step ax ay az bx by bz cx cy cz ...
            return [float(f[1]), float(f[5]), float(f[9])]
        return [float(x) for x in f[:3]]
    if cryst and min(cryst) > 1.0:
        return cryst
    sys.exit("No box: give --box a b c, --box-file (VMD box.txt or NAMD .xsc), "
             "or a PDB with CRYST1")


def load(args):
    psf = app.CharmmPsfFile(args.psf)
    params = app.CharmmParameterSet(*args.toppar)
    cryst = None
    if args.pdb.lower().endswith(".crd"):
        pos = app.CharmmCrdFile(args.pdb).positions
    else:
        pos, cryst = read_pdb_coords(args.pdb)
    if len(pos) != len(psf.atom_list):
        sys.exit(f"{args.pdb} has {len(pos)} atoms but {args.psf} has "
                 f"{len(psf.atom_list)}; they must come from the same build")
    box = read_box(args, cryst)
    psf.setBox(box[0] * u.angstrom, box[1] * u.angstrom, box[2] * u.angstrom)
    return psf, params, pos, box


def handle_nbfix(psf, params, drop):
    psf.loadParameters(params)   # PSF may store numeric types until this point
    types = {a.type.name for a in psf.atom_list}
    pairs = set()
    for t in types:
        typ = params.atom_types_str.get(t)
        if typ is None:
            continue
        for t2 in typ.nbfix:
            if t2 in types:
                pairs.add(tuple(sorted((t, t2))))
    if not pairs:
        return []
    if not drop:
        sys.exit(f"NBFIX terms present for {len(pairs)} type pairs (e.g. {sorted(pairs)[:3]}); "
                 "grand cannot handle them. Re-run with --drop-nbfix to ignore NBFIX "
                 "during water placement only.")
    for typ in params.atom_types_str.values():
        typ.nbfix = {}
    params.nbfix_types.clear()
    log.warning("ignoring %d NBFIX pairs during GCMC (output PSF unaffected)", len(pairs))
    return sorted(pairs)


def classify(top):
    """Rename waters to HOH (grand's convention), return index sets."""
    waters, solute_heavy, solute_all = [], [], []
    for res in top.residues():
        atoms = list(res.atoms())
        if res.name in WATER_RES:
            if atoms[0].element is None or atoms[0].element.symbol != "O":
                sys.exit(f"water residue {res.name} {res.id}: first atom must be oxygen")
            res.name = "HOH"
            waters.append(res.index)
            continue
        if len(atoms) == 1 and (res.name in BULK_ION_RES or res.chain.id.upper().startswith("ION")):
            continue  # bulk ions are neither solute nor water
        for a in atoms:
            solute_all.append(a.index)
            if a.element is not None and a.element.symbol not in ("H", "D"):
                solute_heavy.append(a.index)
    return waters, solute_heavy, solute_all


def make_system(psf, params, args, ref_pos, restrain_idx):
    system = psf.createSystem(params, nonbondedMethod=app.PME,
                              nonbondedCutoff=args.cutoff * u.angstrom,
                              switchDistance=args.switch * u.angstrom,
                              constraints=app.HBonds, rigidWater=True)
    for f in system.getForces():
        if isinstance(f, mm.CustomNonbondedForce):
            sys.exit("system contains a CustomNonbondedForce (NBFIX/NBTHOLE?); "
                     "grand cannot handle it")
    if restrain_idx and args.restraint_k > 0:
        rf = mm.CustomExternalForce("k*periodicdistance(x,y,z,x0,y0,z0)^2")
        rf.addGlobalParameter("k", args.restraint_k * KCAL_A2)
        for p in ("x0", "y0", "z0"):
            rf.addPerParticleParameter(p)
        for i in restrain_idx:
            rf.addParticle(i, ref_pos[i].value_in_unit(u.nanometer))
        system.addForce(rf)
    return system


# --------------------------------------------------------------------------
# Cavity detection
# --------------------------------------------------------------------------

def fibonacci_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def find_cavities(solute_xyz, args):
    """Grid points >= probe from solute heavy atoms that are enclosed in all
    ray directions; clustered into cavities. Coordinates in Angstrom."""
    tree = cKDTree(solute_xyz)
    lo, hi = solute_xyz.min(0), solute_xyz.max(0)
    axes = [np.arange(lo[i], hi[i] + args.grid, args.grid) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d, _ = tree.query(grid)
    cand = grid[d >= args.probe]
    # cheap pre-filter: enough solute atoms around to possibly be enclosed
    n8 = tree.query_ball_point(cand, 8.0, return_length=True)
    cand = cand[n8 >= args.min_neighbors]
    dirs = fibonacci_dirs(args.rays)
    L, r2 = args.ray_length, args.ray_radius ** 2
    buried = []
    for p, nb in zip(cand, tree.query_ball_point(cand, L)):
        if not nb:
            continue
        v = solute_xyz[nb] - p
        t = v @ dirs.T
        perp2 = (v * v).sum(1)[:, None] - t * t
        blocked = ((t > 0) & (t < L) & (perp2 < r2)).any(0)
        if blocked.sum() >= args.rays - args.ray_leak:
            buried.append(p)
    buried = np.array(buried).reshape(-1, 3)
    if len(buried) == 0:
        return []
    pairs = cKDTree(buried).query_pairs(args.grid * 1.5, output_type="ndarray")
    n = len(buried)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) \
        if len(pairs) else coo_matrix((n, n))
    ncomp, labels = connected_components(adj, directed=False)
    # merge clusters whose centroids are close: one GCMC sphere per pocket region
    cents = np.array([buried[labels == c].mean(0) for c in range(ncomp)])
    mp = cKDTree(cents).query_pairs(args.merge, output_type="ndarray")
    madj = coo_matrix((np.ones(len(mp)), (mp[:, 0], mp[:, 1])), shape=(ncomp, ncomp)) \
        if len(mp) else coo_matrix((ncomp, ncomp))
    _, group = connected_components(madj, directed=False)
    labels = group[labels]
    cavities = []
    for c in np.unique(labels):
        pts = buried[labels == c]
        cen = pts.mean(0)
        rad = max(args.min_radius, np.linalg.norm(pts - cen, axis=1).max() + args.radius_pad)
        cavities.append({"centre": cen, "radius": float(rad), "points": len(pts)})
    cavities.sort(key=lambda c: -c["points"])
    return cavities


# --------------------------------------------------------------------------
# GCMC
# --------------------------------------------------------------------------

def pick_ghosts(pos_A, waters, top, solute_xyz, n):
    """The n waters whose oxygen is farthest from any solute atom."""
    tree = cKDTree(solute_xyz)
    res = list(top.residues())
    ox = np.array([pos_A[next(res[w].atoms()).index] for w in waters])
    d, _ = tree.query(ox)
    order = np.argsort(-d)[:n]
    return [waters[i] for i in order]


def _nm(x):
    """Quantity/Vec3/array -> plain numpy array in nm."""
    if hasattr(x, "value_in_unit"):
        x = x.value_in_unit(u.nanometer)
    return np.asarray([v for v in x] if not isinstance(x, np.ndarray) else x, dtype=float)


def make_screened_sampler_class(grand):
    """grand's StandardGCMCSphereSampler, but insertion attempts whose oxygen
    lands within `screen` A of a heavy atom of a real (non-ghost) molecule are
    rejected before any GPU work (their acceptance probability is ~exp(-40) or
    smaller). Accepted-candidate handling is delegated to grand's own
    insertionMove, so this works across grand versions."""
    base = grand.samplers.StandardGCMCSphereSampler

    class ScreenedGCMCSphereSampler(base):
        def __init__(self, *a, screen=1.8, **k):
            super().__init__(*a, **k)
            self.screen = screen
            self.n_screened = 0
            self._res_atoms = [[a.index for a in r.atoms()] for r in self.topology.residues()]
            self._is_water = [r.name == "HOH" for r in self.topology.residues()]
            self._heavy = np.array([a.index for a in self.topology.atoms()
                                    if a.element is not None and a.element.symbol != "H"])
            self._tree, self._acc_at_build = None, -1

        def move(self, context, n=1):
            self._tree = None            # positions change between batches
            return super().move(context, n)

        def _overlaps(self, point):
            box = _nm(self.simulation_box)
            if self._tree is None or self._acc_at_build != self.n_accepted:
                ghosts = [int(r) for r in self.getWaterStatusResids(0)]
                if not all(0 <= r < len(self._is_water) and self._is_water[r] for r in ghosts):
                    raise RuntimeError("grand ghost IDs do not map to water residues; "
                                       "run with --screen 0")
                ghost_atoms = {i for r in ghosts for i in self._res_atoms[r]}
                real = np.array([i for i in self._heavy if i not in ghost_atoms])
                pos = _nm(self.positions)
                self._tree = cKDTree(np.mod(pos[real], box), boxsize=box)
                self._acc_at_build = self.n_accepted
            d, _ = self._tree.query(np.mod(point, box))
            return d < self.screen / 10.0

        def insertionMove(self):
            cand = base.insertRandomWater(self)          # grand's own proposal
            new_positions, _, atom_indices = cand
            if self._overlaps(_nm(new_positions[atom_indices[0]])):
                self.n_screened += 1
                if hasattr(self, "acceptance_probabilities"):
                    self.acceptance_probabilities.append(0.0)
                return None
            # evaluate this same candidate with grand's acceptance code
            self.insertRandomWater = lambda: cand
            try:
                return base.insertionMove(self)
            finally:
                del self.insertRandomWater

    return ScreenedGCMCSphereSampler


def reset_grand_logging():
    """grand adds a FileHandler per sampler and never removes it."""
    lg = logging.getLogger("grand.samplers")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    lg.propagate = False


def n_moves_for(cav, args):
    vol = 4.0 / 3.0 * math.pi * cav["radius"] ** 3
    return int(min(args.max_moves, max(args.min_moves, args.move_density * vol)))


def run(args):
    import time
    import grand  # imported late so --help works without it

    t_start = time.time()
    psf, params, pos, box = load(args)
    dropped = handle_nbfix(psf, params, args.drop_nbfix)
    top = psf.topology
    waters, solute_heavy, solute_all = classify(top)
    if not waters:
        sys.exit("no water residues found - run on the solvated system")
    pos_A = np.array(pos.value_in_unit(u.angstrom))
    solute_xyz = pos_A[solute_heavy]

    cavities = [] if args.manual_only else find_cavities(solute_xyz, args)
    for sp in args.sphere or []:
        x, y, z, r = map(float, sp)
        cavities.append({"centre": np.array([x, y, z]), "radius": r, "points": 0,
                         "manual": True})
    if args.max_cavities and len(cavities) > args.max_cavities:
        log.warning("keeping the %d largest of %d cavities", args.max_cavities, len(cavities))
        cavities = cavities[:args.max_cavities]
    log.warning("%d GCMC spheres (%d cavity grid points)", len(cavities),
                sum(c["points"] for c in cavities))
    half = 0.5 * min(box)
    for c in cavities:
        c["radius"] = min(c["radius"], half - 0.1)

    platform = (mm.Platform.getPlatformByName(args.platform) if args.platform
                else max((mm.Platform.getPlatform(i) for i in range(mm.Platform.getNumPlatforms())),
                         key=lambda p: p.getSpeed()))
    log.warning("platform: %s", platform.getName())
    T = args.temperature * u.kelvin
    box_vecs = [mm.Vec3(box[0], 0, 0) * 0.1, mm.Vec3(0, box[1], 0) * 0.1,
                mm.Vec3(0, 0, box[2]) * 0.1] * u.nanometer

    # restrained minimisation (fixes guessed hydrogens before any MC)
    system = make_system(psf, params, args, pos, solute_heavy)
    integ = mm.LangevinMiddleIntegrator(T, 1 / u.picosecond, 2 * u.femtosecond)
    sim = app.Simulation(top, system, integ, platform)
    sim.context.setPeriodicBoxVectors(*box_vecs)
    sim.context.setPositions(pos)
    sim.minimizeEnergy(maxIterations=args.min_steps)
    pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    del sim

    n_ghost = args.ghosts or max(10, sum(int(c["points"] / 8) + 2 for c in cavities))
    ghosts = pick_ghosts(np.array(pos.value_in_unit(u.angstrom)), waters, top,
                         solute_xyz, n_ghost)
    log.warning("using %d bulk waters as ghosts", len(ghosts))

    outdir = Path(args.out + "_gcmc")
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {"platform": platform.getName(), "cavities": [],
               "nbfix_dropped": len(dropped), "ghosts": len(ghosts),
               "time_budget_min": args.time_budget}
    budget = args.time_budget * 60 if args.time_budget else None
    Sampler = (make_screened_sampler_class(grand) if args.screen > 0
               else grand.samplers.StandardGCMCSphereSampler)

    for k, cav in enumerate(cavities):
        t0 = time.time()
        system = make_system(psf, params, args, pos, solute_heavy)
        centre = u.Quantity(cav["centre"] / 10.0, u.nanometer)
        extra = {"screen": args.screen} if args.screen > 0 else {}
        sampler = Sampler(
            **extra, system=system, topology=top, temperature=T,
            excessChemicalPotential=args.mu_ex * u.kilocalories_per_mole,
            standardVolume=args.std_volume * u.angstroms ** 3,
            sphereCentre=centre, sphereRadius=cav["radius"] * u.angstrom,
            ghostFile=str(outdir / f"cav{k:03d}_ghosts.txt"),
            log=str(outdir / f"cav{k:03d}.log"), overwrite=True)
        integ = mm.LangevinMiddleIntegrator(T, 1 / u.picosecond, 2 * u.femtosecond)
        sim = app.Simulation(top, system, integ, platform)
        sim.context.setPeriodicBoxVectors(*box_vecs)
        sim.context.setPositions(pos)
        sim.context.setVelocitiesToTemperature(T)
        sampler.initialise(sim.context, ghosts)
        n0 = sampler.N
        if args.fresh:
            sampler.deleteWatersInGCMCSphere()
        t_setup = time.time() - t0

        target = n_moves_for(cav, args)
        cycles = max(1, math.ceil(target / args.moves))
        # share what is left of the budget equally among the remaining cavities
        deadline = None
        if budget:
            left = budget - (time.time() - t_start)
            deadline = time.time() + max(0.0, left) / (len(cavities) - k)
        stopped = False
        t_mc = 0.0
        for c in range(cycles):
            tm = time.time()
            sampler.move(sim.context, args.moves)
            t_mc += time.time() - tm
            sim.step(args.md_steps)
            if deadline and time.time() > deadline and c < cycles - 1:
                stopped = True
                break
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        ghosts = [int(r) for r in sampler.getWaterStatusResids(0)]
        acc = sampler.n_accepted / max(1, sampler.n_moves)
        screened = getattr(sampler, "n_screened", 0)
        dt = time.time() - t0
        per_move = t_mc / max(1, sampler.n_moves)
        log.warning("sphere %d (r=%.1f A): N %d -> %d, %d moves (%d screened), "
                    "acceptance %.4f, %.0f s (%.1f ms/move, setup %.0f s)%s", k,
                    cav["radius"], n0, sampler.N, sampler.n_moves, screened, acc, dt,
                    1000 * per_move, t_setup, " - TIME LIMIT" if stopped else "")
        summary["cavities"].append({
            "centre_A": np.round(cav["centre"], 2).tolist(),
            "radius_A": round(cav["radius"], 2), "grid_points": cav["points"],
            "manual": cav.get("manual", False), "N_start": n0, "N_end": sampler.N,
            "moves": sampler.n_moves, "moves_target": target, "acceptance": acc,
            "screened": screened,
            "seconds": round(dt, 1), "ms_per_move": round(1000 * per_move, 2),
            "time_limited": stopped})
        reset_grand_logging()
        del sim, sampler
        if len(ghosts) == 0:
            log.warning("ran out of ghost waters - increase --ghosts")

    summary["total_seconds"] = round(time.time() - t_start, 1)
    write_outputs(args, pos, box, ghosts, top, summary)


def write_outputs(args, pos, box, ghosts, top, summary):
    import parmed
    s = parmed.load_file(args.psf)
    res_omm = list(top.residues())
    if len(s.residues) != len(res_omm):
        sys.exit("ParmEd and OpenMM residue counts differ; cannot map ghost waters")
    s.coordinates = np.array(pos.value_in_unit(u.angstrom))
    s.box = [box[0], box[1], box[2], 90.0, 90.0, 90.0]
    for g in ghosts:
        if len(s.residues[g].atoms) != len(list(res_omm[g].atoms())):
            sys.exit(f"residue {g} mismatch between ParmEd and OpenMM")
    if ghosts:
        s.strip(":" + ",".join(str(g + 1) for g in sorted(ghosts)))
    s.save(args.out + ".psf", overwrite=True)
    s.save(args.out + ".pdb", overwrite=True)
    Path(args.out + "_box.txt").write_text(f"{box[0]:.3f} {box[1]:.3f} {box[2]:.3f}\n")
    summary["waters_removed_as_ghosts"] = len(ghosts)
    Path(args.out + "_gcmc_summary.json").write_text(json.dumps(summary, indent=2))
    log.warning("wrote %s.psf/.pdb (%d ghost waters stripped)", args.out, len(ghosts))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--psf", required=True)
    p.add_argument("--pdb", required=True, help="coordinates (.pdb or .crd)")
    p.add_argument("--toppar", nargs="+", required=True)
    p.add_argument("--box", nargs=3, metavar=("A", "B", "C"))
    p.add_argument("--box-file", help="VMD box.txt (a b c) or NAMD .xsc")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--drop-nbfix", action="store_true")
    p.add_argument("--platform", default=None, help="CUDA, OpenCL, CPU (default: fastest)")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--cutoff", type=float, default=12.0)
    p.add_argument("--switch", type=float, default=10.0)
    p.add_argument("--restraint-k", type=float, default=5.0,
                   help="solute heavy-atom restraint, kcal/mol/A^2 (k*dr^2)")
    p.add_argument("--min-steps", type=int, default=2000)
    # cavity detection
    p.add_argument("--grid", type=float, default=1.0)
    p.add_argument("--probe", type=float, default=2.8,
                   help="min distance from solute heavy atoms for a cavity point (A)")
    p.add_argument("--min-neighbors", type=int, default=55,
                   help="solute atoms within 8 A for a point to be considered")
    p.add_argument("--rays", type=int, default=30)
    p.add_argument("--ray-length", type=float, default=12.0)
    p.add_argument("--ray-radius", type=float, default=2.0)
    p.add_argument("--ray-leak", type=int, default=0, help="unblocked rays tolerated")
    p.add_argument("--min-radius", type=float, default=3.0)
    p.add_argument("--merge", type=float, default=5.0,
                   help="merge cavity clusters with centroids closer than this (A)")
    p.add_argument("--radius-pad", type=float, default=1.5)
    # GCMC
    p.add_argument("--ghosts", type=int, default=0, help="0 = automatic")
    p.add_argument("--mu-ex", type=float, default=-6.09, help="kcal/mol")
    p.add_argument("--std-volume", type=float, default=30.345, help="A^3")
    p.add_argument("--move-density", type=float, default=5.0,
                   help="GCMC moves per A^3 of sphere volume")
    p.add_argument("--min-moves", type=int, default=1000)
    p.add_argument("--max-moves", type=int, default=10000)
    p.add_argument("--moves", type=int, default=100, help="GCMC moves per cycle")
    p.add_argument("--md-steps", type=int, default=50, help="2 fs MD steps per cycle")
    p.add_argument("--screen", type=float, default=1.8,
                   help="reject insertions with O closer than this (A) to a heavy atom "
                        "before evaluating the energy; 0 = plain grand sampler")
    p.add_argument("--fresh", action="store_true",
                   help="delete existing waters in each sphere before sampling")
    p.add_argument("--time-budget", type=float, default=0,
                   help="minutes per structure for all spheres (0 = no limit)")
    p.add_argument("--max-cavities", type=int, default=0, help="largest N only")
    p.add_argument("--sphere", nargs=4, action="append", metavar=("X", "Y", "Z", "R"),
                   help="extra GCMC sphere (A), repeatable")
    p.add_argument("--manual-only", action="store_true",
                   help="use only --sphere sites, skip cavity detection")
    p.add_argument("--cavities-only", action="store_true",
                   help="only detect cavities and write them as a PDB, no GCMC")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("gcmc_hydrate").setLevel(logging.INFO)

    if args.cavities_only:
        psf, params, pos, box = load(args)
        _, solute_heavy, _ = classify(psf.topology)
        xyz = np.array(pos.value_in_unit(u.angstrom))[solute_heavy]
        cavs = find_cavities(xyz, args)
        with open(args.out + "_cavities.pdb", "w") as fh:
            for i, c in enumerate(cavs, 1):
                x, y, z = c["centre"]
                fh.write(f"HETATM{i:5d}  O   CAV X{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                         f"  1.00{c['radius']:6.2f}           O\n")
        print(f"{len(cavs)} cavities; centres in {args.out}_cavities.pdb "
              "(B-factor = sphere radius)")
        return
    run(args)


if __name__ == "__main__":
    main()
