#!/usr/bin/env python3

"""
af_prep.py - prepare AlphaFold models for CHARMM/NAMD simulation via psfgen.

Pipeline, per model (order matters):
  1. UniProt annotations: cofactors, binding sites, mature chain, covalent mods
  2. AlphaFill transplants, filtered to annotated cofactors, best copy per site
  3. Terminal trimming by mature-chain boundaries and pLDDT (UniProt numbering kept)
  4. Dowser internal waters (optional), with cofactors present
  5. PDB2PQR + PROPKA protonation, decoded from the hydrogens PDB2PQR adds
  6. Metal-site overrides and disulfide detection
  7. psfgen build script (+ optional solvate/autoionize and NAMD extraBonds
     restraints for metal sites)

Anything a human should look at is flagged REVIEW in <outdir>/<model>/report.txt
and summarised in <outdir>/summary.tsv.

Requirements: python>=3.8, numpy, scipy, gemmi, requests, pdb2pqr (CLI).
Optional: Dowser (any CLI variant), VMD (to run the generated build script).

Example:
  python af_prep.py models/AF-*.pdb --outdir prep --ph 7.0 \
      --toppar toppar/top_all36_prot.rtf toppar/toppar_water_ions.str \
               toppar/toppar_all36_prot_heme.str \
      --dowser-cmd "dowserx {input}" --dowser-output dowserwat.pdb \
      --solvate --run-vmd
"""

import argparse
import json
import logging
import re
import shlex
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger("af_prep")

# --------------------------------------------------------------------------
# Configuration tables - extend these for your systems
# --------------------------------------------------------------------------

# Endpoints. VERIFY the AlphaFill one against its API docs; it is a template.
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"
ALPHAFILL_URL = "https://alphafill.eu/v1/aff/{acc}"

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
WATER_NAMES = {"HOH", "WAT", "DOD", "TIP3", "TIP", "SOL", "H2O"}
METAL_ELEMENTS = {"ZN", "MG", "CA", "MN", "FE", "CU", "NI", "CO", "CD", "NA",
                  "K", "HG", "MO", "W"}

# UniProt cofactor / binding-site ligand names -> PDB CCD codes.
# Matching is case- and whitespace-insensitive. Unmapped *cofactor* names are
# flagged REVIEW so you can extend this table.
COFACTOR_NAME_TO_CCD = {
    "Zn(2+)": ["ZN"], "Mg(2+)": ["MG"], "Ca(2+)": ["CA"], "Mn(2+)": ["MN"],
    "Fe(2+)": ["FE2", "FE"], "Fe(3+)": ["FE", "FE2"], "Fe cation": ["FE", "FE2"],
    "Cu(2+)": ["CU"], "Cu(+)": ["CU1", "CU"], "Cu cation": ["CU", "CU1"],
    "Ni(2+)": ["NI"], "Co(2+)": ["CO"], "Cd(2+)": ["CD"],
    "K(+)": ["K"], "Na(+)": ["NA"],
    "heme": ["HEM", "HEC"], "heme b": ["HEM"], "heme c": ["HEC"],
    "FAD": ["FAD"], "FMN": ["FMN"], "NAD(+)": ["NAD"], "NADP(+)": ["NAP"],
    "NADH": ["NAI"], "NADPH": ["NDP"],
    "pyridoxal 5'-phosphate": ["PLP"], "thiamine diphosphate": ["TPP"],
    "[4Fe-4S] cluster": ["SF4"], "[2Fe-2S] cluster": ["FES"],
    "[3Fe-4S] cluster": ["F3S"], "Mo-molybdopterin": ["MTE"],
    "biotin": ["BTN"], "coenzyme A": ["COA"],
}

# CCD code -> (CHARMM residue name, {PDB atom name: CHARMM atom name}).
# Only ligands listed here are built automatically; everything else is written
# out and flagged NEEDS_PARAMS. VERIFY names against your toppar version.
CHARMM_LIGANDS = {
    "ZN": ("ZN2", {"ZN": "ZN"}),     # toppar_water_ions.str
    "MG": ("MG", {"MG": "MG"}),
    "CA": ("CAL", {"CA": "CAL"}),
    "NA": ("SOD", {"NA": "SOD"}),
    "K": ("POT", {"K": "POT"}),
    "HEM": ("HEME", {}),             # toppar_all36_prot_heme.str; names ~match PDB
}

# Ligands that need bonded models / special parameters in any case.
NEEDS_BONDED = {"SF4", "FES", "F3S", "HEC"}
COVALENT_HINTS = ("pyridoxal", "heme", "flavin", "fad", "fmn", "lipoyl",
                  "biotin", "phosphopantetheine", "retinal")

DEFAULT_TOPPAR = ["top_all36_prot.rtf", "toppar_water_ions.str",
                  "toppar_all36_prot_heme.str"]


# --------------------------------------------------------------------------
# Minimal structure handling
# --------------------------------------------------------------------------

@dataclass
class Atom:
    record: str
    name: str
    resname: str
    chain: str
    resnum: int
    icode: str
    x: float
    y: float
    z: float
    b: float
    element: str

    @property
    def xyz(self):
        return (self.x, self.y, self.z)

    @property
    def is_h(self):
        return self.element in ("H", "D")

    @property
    def rkey(self):
        return (self.chain, self.resnum, self.icode)


def _element(name, field):
    e = field.strip().upper()
    if e:
        return e
    s = re.sub(r"[^A-Za-z]", "", name)
    return s[:1].upper() if s else "X"


def read_pdb(path):
    """Column-based PDB reader. Residue name is read from cols 17-21 so that
    PDB2PQR's 4-character CHARMM names (ASPP, CTER, ...) survive."""
    atoms = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        atoms.append(Atom(
            record=line[:6].strip(), name=name,
            resname=line[16:21].strip(), chain=line[21].strip() or "A",
            resnum=int(line[22:26]), icode=line[26].strip(),
            x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
            b=float(line[60:66]) if line[60:66].strip() else 0.0,
            element=_element(name, line[76:78] if len(line) >= 78 else "")))
    return atoms


def read_cif(path):
    import gemmi
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    atoms = []
    for ch in st[0]:
        for res in ch:
            rec = "ATOM" if res.het_flag == "A" else "HETATM"
            for a in res:
                atoms.append(Atom(rec, a.name, res.name, ch.name, res.seqid.num,
                                  res.seqid.icode.strip(), a.pos.x, a.pos.y,
                                  a.pos.z, a.b_iso, a.element.name.upper()))
    return atoms


def read_structure(path):
    p = str(path).lower()
    return read_cif(path) if p.endswith((".cif", ".mmcif")) else read_pdb(path)


def pdb_line(a, serial, segid=""):
    if len(a.name) >= 4 or len(a.element) == 2:
        name = f"{a.name:<4s}"
    else:
        name = f" {a.name:<3s}"
    return (f"{a.record:<6s}{serial % 100000:5d} {name} {a.resname:<4s}"
            f"{a.chain[:1]:1s}{a.resnum % 10000:4d}{(a.icode or ' '):1s}   "
            f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}{1.0:6.2f}{a.b:6.2f}      "
            f"{segid:<4s}{a.element:>2s}\n")


def write_pdb(atoms, path, segid=""):
    with open(path, "w") as fh:
        for i, a in enumerate(atoms, 1):
            fh.write(pdb_line(a, i, segid))
        fh.write("END\n")


def group_residues(atoms):
    res = OrderedDict()
    for a in atoms:
        res.setdefault((a.chain, a.resnum, a.icode, a.resname), []).append(a)
    return res


def is_protein(a):
    return a.resname in AA3 or a.resname in ("HSD", "HSE", "HSP", "MSE")


def is_water(a):
    return a.resname in WATER_NAMES


def heavy(atoms):
    return [a for a in atoms if not a.is_h]


def coords(atoms):
    return np.array([a.xyz for a in atoms], dtype=float).reshape(-1, 3)


# --------------------------------------------------------------------------
# Report helper
# --------------------------------------------------------------------------

class Report:
    def __init__(self, name):
        self.name = name
        self.flags = []
        self.data = OrderedDict()

    def review(self, msg):
        log.warning("[%s] REVIEW: %s", self.name, msg)
        self.flags.append(("REVIEW", msg))

    def info(self, msg):
        log.info("[%s] %s", self.name, msg)
        self.flags.append(("INFO", msg))

    @property
    def status(self):
        if any(l == "ERROR" for l, _ in self.flags):
            return "ERROR"
        return "REVIEW" if any(l == "REVIEW" for l, _ in self.flags) else "OK"

    def write(self, outdir):
        outdir = Path(outdir)
        (outdir / "report.json").write_text(json.dumps(
            {"model": self.name, "status": self.status,
             "flags": [{"level": l, "msg": m} for l, m in self.flags],
             **self.data}, indent=2, default=str))
        lines = [f"Model: {self.name}", f"Status: {self.status}", ""]
        for lvl in ("ERROR", "REVIEW", "INFO"):
            items = [m for l, m in self.flags if l == lvl]
            if items:
                lines.append(f"{lvl}:")
                lines += [f"  - {m}" for m in items]
                lines.append("")
        (outdir / "report.txt").write_text("\n".join(lines))


# --------------------------------------------------------------------------
# 1. UniProt
# --------------------------------------------------------------------------

def norm_name(s):
    return re.sub(r"\s+", "", (s or "").lower())


NAME_TO_CCD_NORM = {norm_name(k): v for k, v in COFACTOR_NAME_TO_CCD.items()}


def accession_from_name(path):
    m = re.search(r"AF-([A-Z0-9]+(?:-\d+)?)-F(\d+)", Path(path).name)
    if m:
        return m.group(1), f"AF-{m.group(1)}-F{m.group(2)}"
    return None, None


def fetch_uniprot(acc, cache_dir):
    import requests
    cache = Path(cache_dir) / f"{acc}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = requests.get(UNIPROT_URL.format(acc=acc), timeout=30)
    r.raise_for_status()
    cache.write_text(r.text)
    return r.json()


def parse_uniprot(d):
    info = {"cofactors": [], "sites": [], "modified": [], "chains": [],
            "length": d.get("sequence", {}).get("length")}
    for cm in d.get("comments", []):
        if cm.get("commentType") == "COFACTOR":
            for c in cm.get("cofactors", []):
                if c.get("name"):
                    info["cofactors"].append(c["name"])
    for f in d.get("features", []):
        loc = f.get("location", {})
        s = loc.get("start", {}).get("value")
        e = loc.get("end", {}).get("value")
        t = f.get("type")
        if t == "Binding site" and f.get("ligand"):
            info["sites"].append({"ligand": f["ligand"].get("name"),
                                  "start": s, "end": e})
        elif t in ("Modified residue", "Cross-link", "Lipidation", "Glycosylation"):
            info["modified"].append({"type": t, "pos": s,
                                     "desc": f.get("description", "")})
        elif t == "Chain" and s and e:
            info["chains"].append((s, e))
    return info


def allowed_ligands(uni, extra, rep):
    """CCD codes to keep, plus binding-site residue numbers per CCD code."""
    allowed = set(extra)
    expected = set()
    for name in uni["cofactors"]:
        ccds = NAME_TO_CCD_NORM.get(norm_name(name))
        if ccds:
            allowed |= set(ccds)
            expected.add(tuple(ccds))
        else:
            rep.review(f"UniProt cofactor '{name}' has no CCD mapping; extend "
                       "COFACTOR_NAME_TO_CCD")
    site_res = {}
    for s in uni["sites"]:
        ccds = NAME_TO_CCD_NORM.get(norm_name(s["ligand"]))
        if not ccds:
            continue
        allowed |= set(ccds)
        for c in ccds:
            site_res.setdefault(c, set()).update(range(s["start"], s["end"] + 1))
    return allowed, expected, site_res


# --------------------------------------------------------------------------
# 2. AlphaFill
# --------------------------------------------------------------------------

def get_alphafill(acc, af_id, mode, url_tmpl, cache_dir, rep):
    if mode == "none" or acc is None:
        return None
    if mode != "auto":
        d = Path(mode)
        for pat in (f"{af_id}*.cif", f"*{acc}*.cif"):
            hits = sorted(d.glob(pat))
            if hits:
                return hits[0]
        rep.review(f"no AlphaFill file for {acc} in {d}")
        return None
    dest = Path(cache_dir) / f"{acc}.alphafill.cif"
    if dest.exists():
        return dest
    try:
        import requests
        r = requests.get(url_tmpl.format(acc=acc, af_id=af_id), timeout=120)
        r.raise_for_status()
        if not r.text.lstrip().startswith("data_"):
            raise ValueError("response is not mmCIF")
        dest.write_text(r.text)
        return dest
    except Exception as exc:  # noqa: BLE001
        rep.review(f"AlphaFill download failed ({exc}); no cofactors transplanted")
        return None


def select_ligands(protein, ligands, allowed, site_res, cfg, rep):
    """Keep annotated ligands; one copy per site; reject clashes/overlaps."""
    ph = heavy(protein)
    tree = cKDTree(coords(ph))
    cands = []
    for key, atoms in ligands.items():
        ccd = key[3]
        if ccd not in allowed:
            rep.info(f"dropped {ccd} {key[0]}{key[1]} (not an annotated cofactor)")
            continue
        la = heavy(atoms) or atoms
        xyz = coords(la)
        d, _ = tree.query(xyz)
        metal = len(la) == 1 and la[0].element in METAL_ELEMENTS
        clashes = int(np.sum(d < (cfg.metal_clash if metal else cfg.clash)))
        near = {(ph[i].chain, ph[i].resnum)
                for lst in tree.query_ball_point(xyz, 4.0) for i in lst}
        if not tree.query_ball_point(xyz, 5.0, return_length=True).any():
            rep.info(f"dropped {ccd} {key[0]}{key[1]} (only contacts trimmed residues)")
            continue
        annotated = len({r for _, r in near} & site_res.get(ccd, set()))
        cands.append(dict(key=key, ccd=ccd, atoms=atoms, xyz=xyz, metal=metal,
                          clashes=clashes, annotated=annotated))

    cands.sort(key=lambda c: (c["clashes"], -c["annotated"]))
    kept = []
    for c in cands:
        tag = f"{c['ccd']} {c['key'][0]}{c['key'][1]}"
        limit = 0 if c["metal"] else cfg.max_clash
        if c["clashes"] > limit:
            rep.review(f"rejected {tag}: {c['clashes']} clashing atoms")
            continue
        if any(np.min(np.linalg.norm(c["xyz"][:, None] - k["xyz"][None], axis=2))
               < cfg.overlap for k in kept):
            continue  # another copy of this site (or a competing ligand) won
        if site_res.get(c["ccd"]) and c["annotated"] == 0:
            rep.review(f"{tag} placed but touches none of the UniProt binding-site "
                       "residues")
        kept.append(c)
        rep.info(f"kept {tag} (clashes={c['clashes']}, "
                 f"annotated contacts={c['annotated']})")
    return kept


# --------------------------------------------------------------------------
# 3. Trimming
# --------------------------------------------------------------------------

def trim_protein(protein, plddt, uni, cfg, rep):
    """Returns trimmed atoms and {chain: (first_patch, last_patch)}."""
    out, termini = [], {}
    by_chain = OrderedDict()
    for key, atoms in group_residues(protein).items():
        by_chain.setdefault(key[0], []).append((key, atoms))
    span = None
    if uni and len(uni["chains"]) == 1:
        span = uni["chains"][0]
    elif uni and len(uni["chains"]) > 1:
        rep.review(f"UniProt lists {len(uni['chains'])} mature chains "
                   "(polyprotein?); mature boundaries not applied")
    for ch, residues in by_chain.items():
        nums = [k[1] for k, _ in residues]
        lo, hi = (span if span else (nums[0], nums[-1]))
        idx = [i for i, n in enumerate(nums) if lo <= n <= hi]
        if not idx:
            rep.review(f"chain {ch}: no residues inside mature span {lo}-{hi}")
            idx = list(range(len(nums)))
        conf = [plddt.get((ch, nums[i]), 100.0) >= cfg.plddt for i in idx]
        w = cfg.trim_window
        first = next((j for j in range(len(idx)) if all(conf[j:j + w])), 0)
        last = next((j for j in range(len(idx) - 1, -1, -1)
                     if all(conf[max(0, j - w + 1):j + 1])), len(idx) - 1)
        keep = idx[first:last + 1]
        n0, n1 = nums[keep[0]], nums[keep[-1]]
        n_native, c_native = n0 == lo, n1 == hi
        if span and lo > nums[0]:
            rep.info(f"chain {ch}: removed {lo - nums[0]} residues before mature "
                     f"chain (signal/transit/propeptide/iMet)")
        if not n_native:
            rep.info(f"chain {ch}: N-terminus trimmed to {n0} by pLDDT -> ACE cap")
        if not c_native:
            rep.info(f"chain {ch}: C-terminus trimmed to {n1} by pLDDT -> CT3 cap")
        low = [nums[i] for i in keep if plddt.get((ch, nums[i]), 100) < cfg.plddt]
        if low:
            rep.info(f"chain {ch}: {len(low)} internal residues below pLDDT "
                     f"{cfg.plddt} kept")
        first_res = residues[keep[0]][0][3]
        if n_native:
            fp = {"PRO": "PROP", "GLY": "GLYP"}.get(first_res, "NTER")
        else:
            fp = "ACP" if first_res == "PRO" else "ACE"
        termini[ch] = (fp, "CTER" if c_native else "CT3", n0, n1)
        for i in keep:
            out.extend(residues[i][1])
    return out, termini


# --------------------------------------------------------------------------
# 4. Dowser
# --------------------------------------------------------------------------

def run_dowser(protein, ligand_atoms, workdir, cfg, rep):
    if not cfg.dowser_cmd:
        return []
    wd = (Path(workdir) / "dowser").resolve()
    wd.mkdir(exist_ok=True)

    def attempt(atoms, label):
        for f in wd.glob("*"):
            if f.is_file():
                f.unlink()
        inp = wd / "dowser_in.pdb"
        write_pdb(heavy(atoms), inp)
        cmd = shlex.split(cfg.dowser_cmd.format(input=inp.name))
        r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True)
        (wd / f"dowser_{label}.log").write_text(r.stdout + r.stderr)
        out = wd / cfg.dowser_output
        return out if out.exists() and out.stat().st_size else None

    out = attempt(protein + ligand_atoms, "with_ligands")
    if out is None and ligand_atoms:
        rep.review("Dowser failed with cofactors present; retried protein-only "
                   "(cofactor pockets protected only by overlap filter)")
        out = attempt(protein, "protein_only")
    if out is None:
        rep.review("Dowser produced no output; no internal waters added")
        return []

    waters = [a for a in read_structure(out)
              if is_water(a) and a.element == "O"]
    solute = coords(heavy(protein + ligand_atoms))
    tree = cKDTree(solute)
    kept = []
    for w in waters:
        if tree.query(w.xyz)[0] < cfg.water_overlap:
            continue
        if any(np.linalg.norm(np.subtract(w.xyz, k.xyz)) < 2.4 for k in kept):
            continue
        kept.append(w)
    rep.info(f"Dowser: {len(kept)} internal waters kept "
             f"({len(waters) - len(kept)} removed for overlap)")
    return kept


# --------------------------------------------------------------------------
# 5. PDB2PQR protonation
# --------------------------------------------------------------------------

PKA_RE = re.compile(r"^\s+([A-Z]{3}|N\+|C-)\s+(-?\d+)\s+(\S)\s+(-?\d+\.\d+)\s+"
                    r"(-?\d+\.\d+)")


def parse_pka_summary(text):
    pkas, on = {}, False
    for line in text.splitlines():
        if "SUMMARY OF THIS PREDICTION" in line:
            on = True
            continue
        if on:
            m = PKA_RE.match(line)
            if m:
                pkas[(m.group(3), int(m.group(2)), m.group(1))] = float(m.group(4))
            elif pkas and not line.strip().startswith("Group"):
                break
    return pkas


def run_pdb2pqr(protein, waters, workdir, cfg, rep):
    wd = (Path(workdir) / "pdb2pqr").resolve()
    wd.mkdir(exist_ok=True)
    inp = wd / "p2p_in.pdb"
    wat = [replace(w, record="HETATM", resname="HOH", name="O") for w in waters]
    write_pdb(heavy(protein) + wat, inp)
    out_pdb, out_pqr = wd / "p2p_out.pdb", wd / "p2p_out.pqr"
    cmd = [cfg.pdb2pqr, "--ff=CHARMM", "--ffout=CHARMM",
           "--titration-state-method=propka", f"--with-ph={cfg.ph}",
           "--keep-chain", f"--pdb-output={out_pdb}", str(inp), str(out_pqr)]
    r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True)
    (wd / "pdb2pqr_stdout.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0 or not out_pdb.exists():
        raise RuntimeError(f"pdb2pqr failed (see {wd}/pdb2pqr_stdout.log)")
    logtxt = ""
    for f in (wd / "p2p_out.log", wd / "pdb2pqr_stdout.log"):
        if f.exists():
            logtxt += f.read_text()
    atoms = read_pdb(out_pdb)
    # CHARMM output renames disulfide residues DISU with atoms "1CB"/"1SG";
    # restore plain heavy-atom names, remember which residues PDB2PQR bonded.
    p2p_disu = {(a.chain, a.resnum) for a in atoms
                if a.resname == "DISU" or re.match(r"^\dSG$", a.name)}
    for i, a in enumerate(atoms):
        if not a.is_h and re.match(r"^\d[A-Z]", a.name):
            atoms[i] = replace(a, name=a.name[1:])
    return atoms, parse_pka_summary(logtxt), p2p_disu


def decode_states(h_atoms, orig_resname, rep):
    """Protonation state per residue from hydrogens present (naming-robust)."""
    names = {}
    for a in h_atoms:
        names.setdefault((a.chain, a.resnum), set()).add(a.name)
    states, free_cys = {}, []
    for (ch, rn), rname in orig_resname.items():
        n = names.get((ch, rn), set())
        if rname == "HIS":
            d, e = "HD1" in n, "HE2" in n
            states[(ch, rn)] = "HSP" if d and e else ("HSD" if d else "HSE")
            if not (d or e):
                rep.review(f"HIS {ch}{rn}: no ring H from PDB2PQR; defaulted HSE")
        elif rname == "ASP" and n & {"HD1", "HD2"}:
            states[(ch, rn)] = "ASPP"
        elif rname == "GLU" and n & {"HE1", "HE2"}:
            states[(ch, rn)] = "GLUP"
        elif rname == "LYS" and "HZ1" in n and "HZ3" not in n:
            states[(ch, rn)] = "LSN"
        elif rname == "CYS" and "HG" not in n and "HG1" not in n:
            free_cys.append((ch, rn))
        elif rname == "TYR" and "HH" not in n:
            rep.review(f"TYR {ch}{rn} deprotonated by PDB2PQR; no standard "
                       "CHARMM patch applied")
    return states, free_cys


# --------------------------------------------------------------------------
# 6. Metal sites and disulfides
# --------------------------------------------------------------------------

COORD_ATOMS = {("HIS", "ND1"), ("HIS", "NE2"), ("CYS", "SG"), ("ASP", "OD1"),
               ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}


def metal_sites(prot_heavy, orig_resname, ligands, cfg):
    """List of (ligand dict, metal Atom, protein Atom, distance)."""
    pts = [a for a in prot_heavy
           if (orig_resname.get((a.chain, a.resnum)), a.name) in COORD_ATOMS]
    if not pts:
        return []
    tree = cKDTree(coords(pts))
    sites = []
    for lig in ligands:
        for m in lig["atoms"]:
            if m.element not in METAL_ELEMENTS:
                continue
            for i in tree.query_ball_point(m.xyz, cfg.metal_cutoff):
                d = float(np.linalg.norm(np.subtract(m.xyz, pts[i].xyz)))
                sites.append((lig, m, pts[i], d))
    return sites


def find_disulfides(prot_heavy, exclude, cfg):
    sg = [a for a in prot_heavy if a.name == "SG" and (a.chain, a.resnum) not in exclude]
    cand = []
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            d = float(np.linalg.norm(np.subtract(sg[i].xyz, sg[j].xyz)))
            if cfg.ss_min < d < cfg.ss_cutoff:
                cand.append((d, i, j))
    used, pairs = set(), []
    for d, i, j in sorted(cand):
        if i in used or j in used:
            continue
        used |= {i, j}
        pairs.append(((sg[i].chain, sg[i].resnum), (sg[j].chain, sg[j].resnum), d))
    return pairs


# --------------------------------------------------------------------------
# 7. Build files
# --------------------------------------------------------------------------

def normalize_cterm(atoms, patch):
    names = {a.name for a in atoms}
    out = []
    for a in atoms:
        if patch == "CTER":
            if a.name == "O" and "OT1" not in names:
                a = replace(a, name="OT1")
            elif a.name == "OXT":
                a = replace(a, name="OT2")
        else:  # CT3 cap: single carbonyl O, cap atoms guessed by psfgen
            if a.name in ("OXT", "OT2"):
                continue
            if a.name == "OT1":
                a = replace(a, name="O")
        out.append(a)
    return out


def tcl_list(items):
    return " ".join("{" + " ".join(str(x) for x in it) + "}" for it in items)


def build_files(name, outdir, prot_heavy, orig_resname, termini, states, patches,
                disulfides, ligands, waters, phem, extrabonds, cfg, rep):
    bd = Path(outdir) / "build"
    bd.mkdir(exist_ok=True)
    seg = lambda ch: f"PRO{ch}"
    tcl = [f"# psfgen build for {name} - generated by af_prep.py",
           "package require psfgen", "resetpsf"]
    tcl += [f"topology {{{Path(t).resolve() if Path(t).exists() else t}}}"
            for t in cfg.toppar]
    tcl += ["pdbalias residue HIS HSE", "pdbalias atom ILE CD1 CD",
            "pdbalias residue HOH TIP3", "pdbalias atom TIP3 O OH2", ""]

    by_chain = OrderedDict()
    for a in prot_heavy:
        by_chain.setdefault(a.chain, []).append(a)
    for ch, atoms in by_chain.items():
        fp, lp, n0, n1 = termini[ch]
        res = group_residues(atoms)
        final = []
        for (c, rn, ic, _), ra in res.items():
            rname = orig_resname[(c, rn)]
            st = states.get((c, rn), "")
            rname = st if st.startswith("HS") else rname
            ra = [replace(a, resname=rname, record="ATOM") for a in ra]
            if rn == n1:
                ra = normalize_cterm(ra, lp)
            final.extend(ra)
        f = bd / f"protein_{ch}.pdb"
        write_pdb(final, f, seg(ch))
        tcl += [f"segment {seg(ch)} {{", f"    first {fp}", f"    last {lp}",
                f"    pdb {f.name}", "}", f"coordpdb {f.name} {seg(ch)}"]

    tcl += ["", "# protonation patches"]
    tcl += [f"patch {p} {seg(ch)}:{rn}" for p, ch, rn in patches]
    tcl += ["", "# disulfides"]
    tcl += [f"patch DISU {seg(a[0])}:{a[1]} {seg(b[0])}:{b[1]}  ;# {d:.2f} A"
            for a, b, d in disulfides]

    tcl += ["", "# cofactors"]
    lig_seg = {}
    for k, lig in enumerate(ligands, 1):
        sid = f"HT{k:02d}"
        ccd = lig["ccd"]
        f = bd / f"lig_{k:02d}_{ccd}.pdb"
        if ccd in CHARMM_LIGANDS and ccd not in NEEDS_BONDED:
            rname, amap = CHARMM_LIGANDS[ccd]
            la = [replace(a, record="HETATM", resname=rname, chain="X", resnum=1,
                          icode="", name=amap.get(a.name, a.name))
                  for a in heavy(lig["atoms"])]
            write_pdb(la, f, sid)
            auto = "    auto none\n" if len(la) == 1 else ""
            tcl += [f"segment {sid} {{\n    first none\n    last none\n{auto}"
                    f"    pdb {f.name}\n}}", f"coordpdb {f.name} {sid}"]
            lig_seg[id(lig)] = (sid, la)
        else:
            write_pdb(heavy(lig["atoms"]), f, sid)
            rep.review(f"{ccd}: no automatic CHARMM parameters - written to "
                       f"build/{f.name}, NOT built (CHARMM-GUI Ligand Reader / "
                       "CGenFF, or a bonded model for metal clusters)")
            tcl += [f"# NEEDS_PARAMS: {ccd} in {f.name} (segment {sid}) not built"]

    for his, lig in phem:
        if id(lig) in lig_seg:
            tcl.append(f"patch PHEM {seg(his[0])}:{his[1]} {lig_seg[id(lig)][0]}:1"
                       "  ;# VERIFY patch name/order in your heme toppar")

    if waters:
        f = bd / "water_dowser.pdb"
        wa = [replace(w, record="HETATM", resname="TIP3", name="OH2", chain="W",
                      resnum=i, icode="", element="O")
              for i, w in enumerate(waters, 1)]
        write_pdb(wa, f, "DWAT")
        tcl += ["", "# Dowser internal waters",
                "segment DWAT {\n    first none\n    last none\n    auto none\n"
                f"    pdb {f.name}\n}}", f"coordpdb {f.name} DWAT"]

    tcl += ["", "regenerate angles dihedrals", "guesscoord",
            f"writepsf {name}.psf", f"writepdb {name}.pdb"]
    final = name
    if cfg.solvate:
        tcl += ["", "package require solvate",
                f"solvate {name}.psf {name}.pdb -t {cfg.padding} -o {name}_solv",
                "package require autoionize",
                f"autoionize -psf {name}_solv.psf -pdb {name}_solv.pdb "
                f"-sc {cfg.salt} -o {name}_ion"]
        final = f"{name}_ion"
        tcl += ["", "# box dimensions for NAMD / gcmc_hydrate.py",
                f"set bm [mol new {final}.psf]", f"mol addfile {final}.pdb waitfor all",
                "set mm [measure minmax [atomselect $bm all]]",
                f"set fh [open {final}_box.txt w]",
                "puts $fh [vecsub [lindex $mm 1] [lindex $mm 0]]", "close $fh",
                "mol delete $bm"]

    eb = []
    for lig, m, p, d in extrabonds:
        if id(lig) not in lig_seg:
            continue
        sid, la = lig_seg[id(lig)]
        mname = CHARMM_LIGANDS[lig["ccd"]][1].get(m.name, m.name)
        eb.append((sid, 1, mname, seg(p.chain), p.resnum, p.name, f"{d:.3f}"))
    if eb:
        tcl += ["", "# NAMD extraBonds restraints for metal coordination",
                f"set m [mol new {final}.psf]", f"mol addfile {final}.pdb waitfor all",
                "set fh [open metal_extrabonds.txt w]",
                f"foreach {{s1 r1 n1 s2 r2 n2 r0}} {{{' '.join(' '.join(map(str, e)) for e in eb)}}} {{",
                "    set a [atomselect $m \"segname $s1 and resid $r1 and name $n1\"]",
                "    set b [atomselect $m \"segname $s2 and resid $r2 and name $n2\"]",
                f"    puts $fh \"bond [$a get index] [$b get index] {cfg.metal_k} $r0\"",
                "    $a delete; $b delete", "}", "close $fh"]
    tcl += ["", "exit"]
    (bd / "build.tcl").write_text("\n".join(tcl) + "\n")
    return bd


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def process(model, cfg):
    model = Path(model)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", model.stem)
    outdir = Path(cfg.outdir) / name
    outdir.mkdir(parents=True, exist_ok=True)
    rep = Report(name)
    acc, af_id = accession_from_name(model)
    if cfg.accession:
        acc = cfg.accession
        af_id = af_id or acc
    rep.data["accession"] = acc

    # --- input model and pLDDT
    inp = read_structure(model)
    plddt = {(a.chain, a.resnum): a.b for a in inp if a.name == "CA" and is_protein(a)}

    # --- 1. UniProt
    uni = None
    if acc and not cfg.no_uniprot:
        try:
            uni = parse_uniprot(fetch_uniprot(acc, cfg.cache))
        except Exception as exc:  # noqa: BLE001
            rep.review(f"UniProt lookup failed for {acc}: {exc}")
    elif not acc:
        rep.review("no UniProt accession (filename not AF-<acc>-F<n>; use --accession)")
    if uni:
        rep.data["uniprot"] = uni
        for m in uni["modified"]:
            if any(h in m["desc"].lower() for h in COVALENT_HINTS):
                rep.review(f"covalent cofactor/modification at {m['pos']}: "
                           f"{m['desc']} - needs a custom patch")
            elif m["type"] == "Glycosylation":
                rep.info(f"glycosylation at {m['pos']}: {m['desc']} (not modelled)")
    allowed, expected, site_res = (allowed_ligands(uni, cfg.keep_ligands, rep)
                                   if uni else (set(cfg.keep_ligands), set(), {}))

    # --- 2. AlphaFill: use its protein coordinates if available
    af_file = get_alphafill(acc, af_id, cfg.alphafill, cfg.alphafill_url, cfg.cache, rep)
    src = read_structure(af_file) if af_file else inp
    protein = [a for a in src if is_protein(a) and a.record == "ATOM"]
    ligs = OrderedDict((k, v) for k, v in group_residues(
        [a for a in src if not is_protein(a) and not is_water(a)]).items())
    if af_file:
        ca_in = {(a.chain, a.resnum): a.xyz for a in inp if a.name == "CA"}
        ca_af = {(a.chain, a.resnum): a.xyz for a in protein if a.name == "CA"}
        common = sorted(set(ca_in) & set(ca_af))
        if len(common) != len(ca_in) or len(common) != len(ca_af):
            rep.review("AlphaFill protein differs in length from input model "
                       "(different AF DB version?); AlphaFill coordinates used")
        elif common:
            dev = np.sqrt(np.mean(np.sum((np.array([ca_in[k] for k in common]) -
                                          np.array([ca_af[k] for k in common])) ** 2, 1)))
            if dev > 0.5:
                rep.review(f"AlphaFill protein deviates {dev:.2f} A from input "
                           "model (different AF version?); AlphaFill coordinates used")

    # --- 3. trim
    protein, termini = trim_protein(protein, plddt, uni, cfg, rep)
    orig_resname = {(a.chain, a.resnum): ("HIS" if a.resname in ("HSD", "HSE", "HSP")
                                          else a.resname) for a in protein}

    ligands = select_ligands(protein, ligs, allowed, site_res, cfg, rep) if ligs else []
    found = {l["ccd"] for l in ligands}
    for ccds in expected:
        if not found & set(ccds):
            rep.review(f"annotated cofactor {'/'.join(ccds)} not placed - "
                       "consider co-folding (Boltz/Chai-1) or a homolog structure")
    lig_atoms = [a for l in ligands for a in l["atoms"]]
    write_pdb(heavy(protein) + heavy(lig_atoms), outdir / "prepped.pdb")

    # --- 4. Dowser
    waters = run_dowser(protein, lig_atoms, outdir, cfg, rep)

    # --- 5. PDB2PQR
    p2p_atoms, pkas, p2p_disu = run_pdb2pqr(protein, waters, outdir, cfg, rep)
    p2p_prot = [a for a in p2p_atoms if (a.chain, a.resnum) in orig_resname
                and not is_water(a)]
    missing = set(orig_resname) - {(a.chain, a.resnum) for a in p2p_prot}
    if missing:
        raise RuntimeError(f"PDB2PQR dropped residues: {sorted(missing)[:10]}")
    states, free_cys = decode_states([a for a in p2p_prot if a.is_h],
                                     orig_resname, rep)
    prot_heavy = heavy(p2p_prot)   # PDB2PQR coordinates include its NQH flips

    # --- 6. metal overrides
    sites = metal_sites(prot_heavy, orig_resname, ligands, cfg)
    metal_bound, phem, extrabonds = set(), [], []
    his_coord = {}
    for lig, m, p, d in sites:
        key = (p.chain, p.resnum)
        rname = orig_resname[key]
        metal_bound.add(key)
        if lig["ccd"] in ("HEM", "HEC"):
            if rname == "HIS" and p.name == "NE2":
                phem.append((key, lig))
            if rname == "HIS":
                his_coord.setdefault(key, set()).add(p.name)
            continue
        extrabonds.append((lig, m, p, d))
        if rname == "HIS":
            his_coord.setdefault(key, set()).add(p.name)
        elif rname in ("ASP", "GLU") and states.get(key) in ("ASPP", "GLUP"):
            rep.info(f"{rname} {key[0]}{key[1]} coordinates {m.name}: forced "
                     "deprotonated")
            states.pop(key)
        elif rname == "CYS":
            if cfg.cys_deprot_patch:
                states[key] = cfg.cys_deprot_patch
            else:
                rep.review(f"CYS {key[0]}{key[1]} coordinates {m.name} and should "
                           "be deprotonated; set --cys-deprot-patch for your toppar")
    for key, ns in his_coord.items():
        if ns == {"ND1"}:
            new = "HSE"
        elif ns == {"NE2"}:
            new = "HSD"
        else:
            rep.review(f"HIS {key[0]}{key[1]} coordinates via both ring N")
            continue
        if states.get(key) != new:
            rep.info(f"HIS {key[0]}{key[1]}: {states.get(key)} -> {new} "
                     "(metal coordination)")
        states[key] = new
    for lig in ligands:
        if lig["ccd"] in NEEDS_BONDED:
            rep.review(f"{lig['ccd']} needs a bonded model / dedicated parameters")
        elif any(a.element in METAL_ELEMENTS for a in lig["atoms"]) \
                and lig["ccd"] not in CHARMM_LIGANDS:
            rep.review(f"metal {lig['ccd']} has no CHARMM mapping")
    if extrabonds:
        rep.info(f"{len(extrabonds)} metal-ligand contacts -> extraBonds restraints")

    # --- disulfides
    ss = find_disulfides(prot_heavy, metal_bound, cfg)
    ss_res = {r for a, b, _ in ss for r in (a, b)}
    if p2p_disu != ss_res:
        rep.review("disulfide detection disagrees with PDB2PQR: only geometric "
                   f"{sorted(ss_res - p2p_disu)}, only PDB2PQR "
                   f"{sorted(p2p_disu - ss_res)}")
    for key in free_cys:
        if key not in ss_res and key not in metal_bound:
            rep.review(f"CYS {key[0]}{key[1]} lacks HG in PDB2PQR output but is "
                       "neither in a disulfide nor metal-bound")
    for (ch, rn, rname), pka in pkas.items():
        if (ch, rn) in ss_res or rname not in ("ASP", "GLU", "HIS", "LYS", "CYS", "TYR"):
            continue
        if abs(pka - cfg.ph) < cfg.pka_window and (ch, rn) not in metal_bound:
            rep.info(f"{rname} {ch}{rn} pKa {pka:.2f} within {cfg.pka_window} of pH")

    # titratable residues near organic cofactors: PROPKA did not see them
    org = [a for l in ligands if not l["metal"] for a in heavy(l["atoms"])]
    if org:
        t = cKDTree(coords(org))
        near = {(a.chain, a.resnum) for a in prot_heavy
                if orig_resname[(a.chain, a.resnum)] in ("ASP", "GLU", "HIS", "LYS", "CYS")
                and t.query(a.xyz)[0] < 4.5}
        for key in sorted(near - metal_bound):
            rep.review(f"{orig_resname[key]} {key[0]}{key[1]} is within 4.5 A of a "
                       "cofactor; its protonation was computed without it")

    patches = [(st, k[0], k[1]) for k, st in sorted(states.items())
               if not st.startswith("HS")]
    rep.data["termini"] = termini
    rep.data["protonation"] = {f"{k[0]}{k[1]}": v for k, v in sorted(states.items())}
    rep.data["disulfides"] = [f"{a[0]}{a[1]}-{b[0]}{b[1]} {d:.2f}" for a, b, d in ss]
    rep.data["cofactors"] = [f"{l['ccd']} {l['key'][0]}{l['key'][1]}" for l in ligands]
    rep.data["internal_waters"] = len(waters)
    rep.info(f"{len(ss)} disulfides; {len(patches)} protonation patches; "
             f"{sum(1 for s in states.values() if s.startswith('HS'))} HIS assigned")

    # --- 7. build
    bd = build_files(name, outdir, prot_heavy, orig_resname, termini, states,
                     patches, ss, ligands, waters, phem, extrabonds, cfg, rep)
    if cfg.run_vmd:
        r = subprocess.run([cfg.vmd, "-dispdev", "text", "-e", "build.tcl"],
                           cwd=bd, capture_output=True, text=True)
        (bd / "vmd.log").write_text(r.stdout + r.stderr)
        if not (bd / f"{name}.psf").exists():
            rep.review("VMD/psfgen build did not produce a PSF; see build/vmd.log")
        elif "MOLECULE DESTROYED BY FATAL ERROR" in r.stdout or \
                re.search(r"ERROR", r.stdout):
            rep.review("psfgen reported errors; see build/vmd.log")
    rep.write(outdir)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="AlphaFold PDB/mmCIF files")
    p.add_argument("--outdir", default="af_prep_out")
    p.add_argument("--cache", default=None, help="cache dir for UniProt/AlphaFill "
                   "(default <outdir>/cache)")
    p.add_argument("--ph", type=float, default=7.0)
    p.add_argument("--accession", help="UniProt accession (single-model runs)")
    p.add_argument("--no-uniprot", action="store_true")
    p.add_argument("--alphafill", default="auto",
                   help="'auto' (download), a directory of AlphaFill .cif files, or 'none'")
    p.add_argument("--alphafill-url", default=ALPHAFILL_URL)
    p.add_argument("--keep-ligands", default="",
                   help="extra CCD codes to keep, comma-separated (e.g. ATP,ADP)")
    p.add_argument("--plddt", type=float, default=50.0, help="terminal trim threshold")
    p.add_argument("--trim-window", type=int, default=3)
    p.add_argument("--clash", type=float, default=2.2)
    p.add_argument("--metal-clash", type=float, default=1.6)
    p.add_argument("--max-clash", type=int, default=2)
    p.add_argument("--overlap", type=float, default=2.0,
                   help="ligand copies closer than this are the same site")
    p.add_argument("--dowser-cmd", default="",
                   help="e.g. 'dowserx {input}' (run inside a scratch dir)")
    p.add_argument("--dowser-output", default="dowserwat.pdb")
    p.add_argument("--water-overlap", type=float, default=2.6)
    p.add_argument("--pdb2pqr", default="pdb2pqr")
    p.add_argument("--pka-window", type=float, default=1.0)
    p.add_argument("--metal-cutoff", type=float, default=2.8)
    p.add_argument("--metal-k", type=float, default=100.0,
                   help="extraBonds force constant, kcal/mol/A^2")
    p.add_argument("--cys-deprot-patch", default="",
                   help="patch name for deprotonated Cys in your toppar")
    p.add_argument("--ss-cutoff", type=float, default=2.5)
    p.add_argument("--ss-min", type=float, default=1.8)
    p.add_argument("--toppar", nargs="+", default=DEFAULT_TOPPAR)
    p.add_argument("--solvate", action="store_true")
    p.add_argument("--padding", type=float, default=12.0)
    p.add_argument("--salt", type=float, default=0.15)
    p.add_argument("--run-vmd", action="store_true")
    p.add_argument("--vmd", default="vmd")
    p.add_argument("-v", "--verbose", action="store_true")
    cfg = p.parse_args(argv)
    cfg.keep_ligands = {s.strip().upper() for s in cfg.keep_ligands.split(",") if s.strip()}
    logging.basicConfig(level=logging.INFO if cfg.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
    cfg.cache = Path(cfg.cache or Path(cfg.outdir) / "cache")
    cfg.cache.mkdir(parents=True, exist_ok=True)
    if shutil.which(cfg.pdb2pqr) is None:
        sys.exit(f"pdb2pqr executable '{cfg.pdb2pqr}' not found")

    rows = []
    for m in cfg.models:
        try:
            rep = process(m, cfg)
            n_rev = sum(1 for l, _ in rep.flags if l == "REVIEW")
            rows.append((Path(m).name, rep.status, n_rev))
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] ERROR: %s", m, exc)
            rows.append((Path(m).name, f"ERROR: {exc}", "-"))
    with open(Path(cfg.outdir) / "summary.tsv", "w") as fh:
        fh.write("model\tstatus\tn_review\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    for r in rows:
        print("\t".join(map(str, r)))


if __name__ == "__main__":
    main()
