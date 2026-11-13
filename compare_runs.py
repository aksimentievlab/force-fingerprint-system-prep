#!/usr/bin/env python3
"""
compare_runs.py - compare two af_prep.py output folders, model by model.

For every model whose status is OK in both runs (or every model present in
both, with --all), it reports:

  composition   protein residues per chain (prepped.pdb), ligands by PDB code,
                internal waters, termini patches, protonation states and
                disulfides (report.json), and atom count and net charge of the
                vacuum PSF (build/<model>.psf) when both runs built one
  placement     ligands and cofactors of run B moved onto run A by superposing
                the C-alpha atoms the two models share, then paired per PDB code
                (optimal assignment on centroids); for each pair the RMSD over
                atom names both copies share and the centroid shift.  A copy whose
                code changed but which sits in the same place (FE -> FE2, NAD ->
                NAI) is reported as a code change, not as one removal plus one
                addition.

Each model gets a category: "unchanged", "waters only" (only the number of
Dowser++ waters differs, and the PSF by 3 atoms per water - Dowser++ samples
water positions, so reruns differ), "waters + protonation" (the waters differ,
and so do protonation states and with them the PSF's atom count or charge:
PDB2PQR optimises hydrogen bonds with the waters present, so this is likely a
consequence), or "changed".  --ignore-waters counts "waters only" as unchanged;
"waters + protonation" stays, since protonation differences are real.  Water
positions are not compared.

Output (in --out, default compare_<A>_vs_<B>):
  models.tsv     one line per model: category and what changed
  ligands.tsv    one line per ligand copy: paired, moved, code changed, only in A/B
  summary.txt    counts over all models, plus the models that changed
Exit status 0; differences are data, not errors.

Example:
  python compare_runs.py 1-prep-old 1-prep-new --shift 0.5
"""
import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

WATER = {"HOH", "WAT", "TIP3", "TP3M", "DOD", "SOL", "H2O"}


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def read_pdb(path):
    """[(record, name, resname, chain, resnum, icode, xyz, element)] from a PDB file."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        el = line[76:78].strip().upper() if len(line) >= 78 else ""
        name = line[12:16].strip()
        out.append((line[:6].strip(), name, line[17:21].strip(), line[21].strip() or "A",
                    int(line[22:26]), line[26].strip(),
                    np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                    el or "".join(c for c in name if c.isalpha())[:1].upper()))
    return out


def split_model(atoms):
    """Protein residues {(chain, resnum, icode): {resname, atoms{name: xyz}}} and
    ligand copies {(chain, resnum, resname): {name: xyz}, ...} (heavy atoms)."""
    prot, ligs = OrderedDict(), OrderedDict()
    for rec, name, resn, ch, rn, ic, xyz, el in atoms:
        if el in ("H", "D") or resn in WATER:
            continue
        if rec == "ATOM":
            r = prot.setdefault((ch, rn, ic), {"resname": resn, "atoms": {}})
            r["atoms"][name] = xyz
        else:
            ligs.setdefault((ch, rn, resn), {})[name] = xyz
    return prot, ligs


def psf_summary(path):
    """(number of atoms, net charge) from a PSF file, or None."""
    p = Path(path)
    if not p.exists():
        return None
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines):
        if "!NATOM" in line:
            n = int(line.split()[0])
            q = sum(float(l.split()[6]) for l in lines[i + 1:i + 1 + n])
            return n, round(q, 3)
    return None


def load_model(run, name):
    d = Path(run) / name
    rep = json.loads((d / "report.json").read_text())
    pdb = d / "prepped.pdb"
    prot, ligs = split_model(read_pdb(pdb)) if pdb.exists() else ({}, {})
    return {"report": rep, "prot": prot, "ligs": ligs,
            "psf": psf_summary(d / "build" / f"{name}.psf")}


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def kabsch(P, Q):
    """R, t minimising |P @ R.T + t - Q|."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - pc @ R.T


def superpose(prot_a, prot_b):
    """Transform taking B onto A over shared C-alpha atoms: (R, t, n, rmsd).
    Residues are matched by (chain, number); if chain IDs differ between runs
    (e.g. mature chains split into segments), by residue number alone."""
    def ca(prot, by_chain):
        return {(k if by_chain else (k[1], k[2])): r["atoms"]["CA"]
                for k, r in prot.items() if "CA" in r["atoms"]}
    for by_chain in (True, False):
        a, b = ca(prot_a, by_chain), ca(prot_b, by_chain)
        common = sorted(set(a) & set(b), key=str)
        if len(common) >= 3:
            P = np.array([b[k] for k in common])
            Q = np.array([a[k] for k in common])
            R, t = kabsch(P, Q)
            rmsd = float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, 1))))
            return R, t, len(common), rmsd
    return np.eye(3), np.zeros(3), 0, float("nan")


def ligand_rmsd(la, lb):
    """RMSD over shared atom names; if the copies share fewer than half their
    atoms by name, the symmetric nearest-atom RMSD instead (name-independent)."""
    names = [n for n in la if n in lb]
    if len(names) >= 0.5 * max(len(la), len(lb)):
        d = np.array([la[n] - lb[n] for n in names])
        return float(np.sqrt(np.mean(np.sum(d ** 2, 1)))), "names"
    A, B = np.array(list(la.values())), np.array(list(lb.values()))
    D = np.linalg.norm(A[:, None] - B[None], axis=2)
    sq = np.concatenate([D.min(1) ** 2, D.min(0) ** 2])
    return float(np.sqrt(sq.mean())), "nearest"


def centroid(lig):
    return np.mean(np.array(list(lig.values())), axis=0)


# --------------------------------------------------------------------------
# Comparison of one model
# --------------------------------------------------------------------------

def compare_model(name, A, B, cfg):
    row, lig_rows = OrderedDict(model=name, category="", changed=""), []
    # --- protein composition
    ka, kb = set(A["prot"]), set(B["prot"])
    # chain IDs can change when mature chains are split: compare by number too
    na = {(k[1], k[2]) for k in ka}
    nb = {(k[1], k[2]) for k in kb}
    only_a, only_b = sorted(na - nb), sorted(nb - na)
    row["residues_A"], row["residues_B"] = len(ka), len(kb)
    row["residues_only_A"] = compress([r for r, _ in only_a])
    row["residues_only_B"] = compress([r for r, _ in only_b])
    chains_a = sorted({k[0] for k in ka})
    chains_b = sorted({k[0] for k in kb})
    row["chains"] = ("".join(chains_a) if chains_a == chains_b
                     else f"{''.join(chains_a)} -> {''.join(chains_b)}")
    resname_a = {(k[1], k[2]): r["resname"] for k, r in A["prot"].items()}
    resname_b = {(k[1], k[2]): r["resname"] for k, r in B["prot"].items()}
    mut = [f"{k[0]}{resname_a[k]}>{resname_b[k]}" for k in sorted(set(resname_a) & set(resname_b))
           if resname_a[k] != resname_b[k] and not (resname_a[k] in HIS and resname_b[k] in HIS)]
    row["residue_names_changed"] = ",".join(mut)

    # --- report.json data
    ra, rb = A["report"], B["report"]
    row["termini"] = diff_map(ra.get("termini", {}), rb.get("termini", {}),
                              fmt=lambda v: f"{v[0]}/{v[1]} {v[2]}-{v[3]}" if v else "-")
    row["protonation"] = diff_map(ra.get("protonation", {}), rb.get("protonation", {}))
    sa = {s.rsplit(" ", 1)[0] for s in ra.get("disulfides", [])}
    sb = {s.rsplit(" ", 1)[0] for s in rb.get("disulfides", [])}
    row["disulfides"] = diff_sets(sa, sb)
    wa, wb = ra.get("internal_waters", 0), rb.get("internal_waters", 0)
    row["internal_waters"] = str(wa) if wa == wb else f"{wa} -> {wb}"
    pa, pb = A["psf"], B["psf"]
    if pa and pb:
        row["psf_atoms"] = str(pa[0]) if pa[0] == pb[0] else f"{pa[0]} -> {pb[0]}"
        row["psf_charge"] = (f"{pa[1]:+g}" if pa[1] == pb[1] else f"{pa[1]:+g} -> {pb[1]:+g}")
    else:
        row["psf_atoms"] = row["psf_charge"] = "n/a"

    # --- ligands: superpose B onto A, pair copies per code
    R, t, n_ca, ca_rmsd = superpose(A["prot"], B["prot"])
    row["ca_common"], row["ca_rmsd"] = n_ca, f"{ca_rmsd:.2f}"
    la = {k: v for k, v in A["ligs"].items()}
    lb = {k: {n: x @ R.T + t for n, x in v.items()} for k, v in B["ligs"].items()}
    ca_a, ca_b = Counter(k[2] for k in la), Counter(k[2] for k in lb)
    row["ligands_A"] = fmt_counter(ca_a)
    row["ligands_B"] = fmt_counter(ca_b)
    unpaired_a, unpaired_b = [], []
    for code in sorted(set(ca_a) | set(ca_b)):
        ka_ = [k for k in la if k[2] == code]
        kb_ = [k for k in lb if k[2] == code]
        pairs = []
        if ka_ and kb_:
            C = np.array([[np.linalg.norm(centroid(la[x]) - centroid(lb[y])) for y in kb_]
                          for x in ka_])
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] <= cfg.max_match:
                    pairs.append((ka_[i], kb_[j]))
        used_a = {p[0] for p in pairs}
        used_b = {p[1] for p in pairs}
        for x, y in pairs:
            lig_rows.append(lig_row(name, "paired", x, y, la[x], lb[y], cfg))
        unpaired_a += [k for k in ka_ if k not in used_a]
        unpaired_b += [k for k in kb_ if k not in used_b]
    # same place, different code (FE -> FE2, NAD -> NAI ...)
    for x in list(unpaired_a):
        best = min(unpaired_b, default=None,
                   key=lambda y: np.linalg.norm(centroid(la[x]) - centroid(lb[y])))
        if best is not None and len(la[x]) == len(lb[best]) and \
                np.linalg.norm(centroid(la[x]) - centroid(lb[best])) <= cfg.code_change:
            lig_rows.append(lig_row(name, "code changed", x, best, la[x], lb[best], cfg))
            unpaired_a.remove(x)
            unpaired_b.remove(best)
    for x in unpaired_a:
        lig_rows.append(lig_row(name, "only in A", x, None, la[x], None, cfg))
    for y in unpaired_b:
        lig_rows.append(lig_row(name, "only in B", None, y, None, lb[y], cfg))

    moved = [r for r in lig_rows if r["status"] in ("paired", "code changed")
             and r["moved"] == "yes"]
    row["ligands_moved"] = ",".join(f"{r['code']}({r['rmsd']})" for r in moved)
    row["ligands_only_A"] = ",".join(r["code"] for r in lig_rows if r["status"] == "only in A")
    row["ligands_only_B"] = ",".join(r["code"] for r in lig_rows if r["status"] == "only in B")
    row["ligand_codes_changed"] = ",".join(r["code"] for r in lig_rows
                                           if r["status"] == "code changed")
    keys = ("residues_only_A", "residues_only_B", "residue_names_changed", "protonation",
            "disulfides", "ligands_moved", "ligands_only_A", "ligands_only_B",
            "ligand_codes_changed")
    changed = [k for k in keys if row[k]]
    changed += [k for k in ("termini", "internal_waters", "psf_atoms", "psf_charge", "chains")
                if "->" in str(row[k]) or (k == "termini" and row[k] and row[k] != "same")]
    changed = list(dict.fromkeys(changed))
    # Dowser++ samples water positions, so reruns differ in their internal waters;
    # the vacuum PSF then differs by 3 atoms (TIP3) per water and nothing else
    water_keys = {"internal_waters"}
    if pa and pb and wa != wb and pb[0] - pa[0] == 3 * (wb - wa):
        water_keys.add("psf_atoms")
    other = [k for k in changed if k not in water_keys]
    row["category"] = ("unchanged" if not changed else
                       "waters only" if not other else
                       # PDB2PQR optimises hydrogen bonds with the waters present, so a
                       # different water set can flip protonation states (and with them
                       # the PSF's atom count and charge): likely a consequence
                       "waters + protonation" if wa != wb and "protonation" in other
                       and set(other) <= {"protonation", "psf_atoms", "psf_charge"} else
                       "changed")
    if row["category"] == "waters only" and cfg.ignore_waters:
        row["category"], changed = "unchanged", []
    row["changed"] = ",".join(changed) or "-"
    return row, lig_rows


HIS = {"HIS", "HSD", "HSE", "HSP"}


def lig_row(model, status, ka, kb, la, lb, cfg):
    k = ka or kb
    r = OrderedDict(model=model, status=status,
                    code=(f"{ka[2]}>{kb[2]}" if ka and kb and ka[2] != kb[2] else k[2]),
                    copy_A=f"{ka[0]}{ka[1]}" if ka else "", copy_B=f"{kb[0]}{kb[1]}" if kb else "",
                    atoms_A=len(la) if la else "", atoms_B=len(lb) if lb else "",
                    centroid_shift="", rmsd="", rmsd_by="", moved="")
    if la is not None and lb is not None:
        rmsd, by = ligand_rmsd(la, lb)
        r["centroid_shift"] = f"{np.linalg.norm(centroid(la) - centroid(lb)):.2f}"
        r["rmsd"], r["rmsd_by"] = f"{rmsd:.2f}", by
        r["moved"] = "yes" if rmsd > cfg.shift else "no"
    return r


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

def compress(nums):
    """[1,2,3,7,9,10] -> '1-3,7,9-10'"""
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}" if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ",".join(out)


def fmt_counter(c):
    return ",".join(f"{k}x{n}" if n > 1 else k for k, n in sorted(c.items())) or "-"


def diff_map(a, b, fmt=str):
    keys = sorted(set(a) | set(b), key=str)
    d = [f"{k}:{fmt(a.get(k))}->{fmt(b.get(k))}" for k in keys
         if json.dumps(a.get(k)) != json.dumps(b.get(k))]
    return ";".join(d)


def diff_sets(a, b):
    d = [f"-{x}" for x in sorted(a - b)] + [f"+{x}" for x in sorted(b - a)]
    return ",".join(d)


def write_tsv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    cols = list(rows[0])
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def status_of(run, name):
    f = Path(run) / name / "report.json"
    try:
        return json.loads(f.read_text()).get("status")
    except (OSError, ValueError):
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a", help="output folder of the first (e.g. older) version")
    p.add_argument("run_b", help="output folder of the second version")
    p.add_argument("--out", help="folder for the comparison (default compare_<A>_vs_<B>)")
    p.add_argument("--all", action="store_true",
                   help="compare every model present in both runs, whatever its status")
    p.add_argument("--status", default="OK",
                   help="comma-separated statuses both runs must report (default OK)")
    p.add_argument("--shift", type=float, default=0.5,
                   help="ligand RMSD (A) above which a paired copy counts as moved")
    p.add_argument("--max-match", type=float, default=5.0,
                   help="copies of the same code farther apart than this (A, centroids) "
                        "are not paired: treated as removed in one site, added in another")
    p.add_argument("--ignore-waters", action="store_true",
                   help="count models that differ only in their Dowser++ waters as unchanged "
                        "(\"waters + protonation\" is still reported)")
    p.add_argument("--code-change", type=float, default=1.0,
                   help="an unpaired copy this close (A, centroids) to one with another code "
                        "and the same atom count counts as a code change")
    cfg = p.parse_args(argv)
    ra, rb = Path(cfg.run_a), Path(cfg.run_b)
    for r in (ra, rb):
        if not r.is_dir():
            sys.exit(f"{r} is not a folder")
    out = Path(cfg.out or f"compare_{ra.name}_vs_{rb.name}")
    out.mkdir(parents=True, exist_ok=True)
    statuses = {s.strip() for s in cfg.status.split(",")}

    names_a = {d.name for d in ra.iterdir() if (d / "report.json").exists()}
    names_b = {d.name for d in rb.iterdir() if (d / "report.json").exists()}
    both = sorted(names_a & names_b)
    selected, skipped = [], Counter()
    for n in both:
        sa, sb = status_of(ra, n), status_of(rb, n)
        if cfg.all or (sa in statuses and sb in statuses):
            selected.append(n)
        else:
            skipped[f"{sa} -> {sb}"] += 1

    rows, lig_rows, failed = [], [], []
    for n in selected:
        try:
            row, lr = compare_model(n, load_model(ra, n), load_model(rb, n), cfg)
        except Exception as exc:  # noqa: BLE001 - one broken folder must not stop the rest
            failed.append(f"{n}: {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        lig_rows += lr
    write_tsv(out / "models.tsv", rows)
    write_tsv(out / "ligands.tsv", lig_rows)

    # --- summary
    category = Counter(r["category"] for r in rows)
    changed = [r for r in rows if r["category"] == "changed"]
    waters_only = [r for r in rows if r["category"] == "waters only"]
    waters_prot = [r for r in rows if r["category"] == "waters + protonation"]
    what = Counter(k for r in changed for k in r["changed"].split(","))
    lig_status = Counter(r["status"] for r in lig_rows)
    moved = [r for r in lig_rows if r["moved"] == "yes"]
    lines = [f"A: {ra}", f"B: {rb}",
             f"models in both runs: {len(both)}; compared ({'all' if cfg.all else '/'.join(sorted(statuses)) + ' in both'}): {len(rows)}",
             f"models only in A: {len(names_a - names_b)}; only in B: {len(names_b - names_a)}"]
    if skipped:
        lines.append("not compared (status A -> B): " +
                     ", ".join(f"{k} x{v}" for k, v in skipped.most_common()))
    lines += ["", "models: " + ", ".join(f"{k} {category.get(k, 0)}" for k in
                                       ("unchanged", "waters only",
                                        "waters + protonation", "changed")
                                       if k != "waters only" or not cfg.ignore_waters)
              + (" (models differing only in Dowser++ waters counted as unchanged)"
                 if cfg.ignore_waters else "")]
    lines += ["changed models by what changed:"] if what else []
    lines += [f"  {k}: {v}" for k, v in what.most_common()]
    lines += ["", "ligand copies: " + ", ".join(f"{k} {v}" for k, v in lig_status.most_common())]
    if moved:
        rm = np.array([float(r["rmsd"]) for r in moved])
        lines.append(f"  moved (RMSD > {cfg.shift} A): {len(moved)}; median {np.median(rm):.2f} A, "
                     f"max {rm.max():.2f} A")
        by_code = Counter(r["code"] for r in moved)
        lines.append("  moved by code: " + ", ".join(f"{k} x{v}" for k, v in by_code.most_common()))
    paired = [float(r["rmsd"]) for r in lig_rows if r["status"] == "paired"]
    if paired:
        lines.append(f"  paired copies: RMSD median {np.median(paired):.2f} A, "
                     f"95th percentile {np.percentile(paired, 95):.2f} A")
    if failed:
        lines += ["", f"could not compare {len(failed)} model(s):"] + [f"  {f}" for f in failed]
    if changed:
        lines += ["", "changed models:"]
        lines += [f"  {r['model']}: {r['changed']}" for r in changed]
    if waters_prot:
        lines += ["", "models differing in Dowser++ waters and protonation states "
                      "(waters A -> B; protonation):"]
        lines += [f"  {r['model']}: {r['internal_waters']}; {r['protonation']}"
                  for r in waters_prot]
    if waters_only:
        lines += ["", "models differing only in Dowser++ waters (water count A -> B):"]
        lines += [f"  {r['model']}: {r['internal_waters']}" for r in waters_only]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:min(len(lines), 40)]))
    if len(lines) > 40:
        print(f"... see {out / 'summary.txt'}")
    print(f"\nwritten: {out / 'models.tsv'}, {out / 'ligands.tsv'}, {out / 'summary.txt'}")


if __name__ == "__main__":
    main()
