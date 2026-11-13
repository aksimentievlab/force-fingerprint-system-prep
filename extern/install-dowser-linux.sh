#! /bin/bash

## Downloads, unpacks and patches Dowser++ after installing (locally)
##   its mgltools dependency. The patched version restricts Dowser
##   calculations to burried regions of the protein identified using a
##   ray-tracing algorithm implemented by the select_boxes.py script.

wget -O mgltools_x86_64Linux2_1.5.7.tar.gz https://ccsb.scripps.edu/download/532/
tar -xzf mgltools_x86_64Linux2_1.5.7.tar.gz
(
cd mgltools_x86_64Linux2_1.5.7
bash install.sh -c 1
)

wget http://stuchebrukhov.ucdavis.edu/dowserplusplus/dowserplusplus.tar.gz
tar -xzf dowserplusplus.tar.gz
(
    cd dowserplusplus/files
    cp -f ../../mgltools_x86_64Linux2_1.5.7/bin/pythonsh .
    chmod a+x run load_data reform drain sorting internal_predicted placeWat choosing main.sh pdbqt conf make_config pythonsh

    cat > select_boxes.py <<"EOF"
#!/usr/bin/env python3
"""
select_boxes.py - print the Dowser++ box numbers that overlap a buried cavity.

Usage (from Dowser++'s main.sh, after ./conf):
    keep=$(python3 select_boxes.py input.pdb $num_boxes)
    for kconf in $keep; do ...

Cavities: grid points at least --probe A from every heavy atom that are enclosed
in all ray directions (same method as gcmc_hydrate.py). A box is kept if it comes
within --margin A of a cavity's sphere. Box files are AutoDock Vina configs
(box<k>.txt with center_x/y/z and size_x/y/z). A summary goes to stderr.

Requires numpy and scipy only.
"""
import argparse
import math
import re
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

WATER = {"HOH", "WAT", "TIP3", "SOL", "DOD"}


def heavy_atoms(pdb):
    xyz = []
    for line in open(pdb):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[17:21].strip() in WATER:
            continue
        el = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not el:
            el = re.sub(r"[^A-Za-z]", "", line[12:16])[:1].upper()
        if el in ("H", "D"):
            continue
        xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.array(xyz)


def fibonacci_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def find_cavities(xyz, a):
    tree = cKDTree(xyz)
    lo, hi = xyz.min(0), xyz.max(0)
    axes = [np.arange(lo[i], hi[i] + a.grid, a.grid) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d, _ = tree.query(grid)
    cand = grid[d >= a.probe]
    cand = cand[tree.query_ball_point(cand, 8.0, return_length=True) >= a.min_neighbors]
    dirs = fibonacci_dirs(a.rays)
    L, r2 = a.ray_length, a.ray_radius ** 2
    buried = []
    for p, nb in zip(cand, tree.query_ball_point(cand, L)):
        if not nb:
            continue
        v = xyz[nb] - p
        t = v @ dirs.T
        perp2 = (v * v).sum(1)[:, None] - t * t
        if ((t > 0) & (t < L) & (perp2 < r2)).any(0).sum() >= a.rays - a.ray_leak:
            buried.append(p)
    if not buried:
        return []
    buried = np.array(buried)
    n = len(buried)
    pairs = cKDTree(buried).query_pairs(a.grid * 1.5, output_type="ndarray")
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) \
        if len(pairs) else coo_matrix((n, n))
    _, labels = connected_components(adj, directed=False)
    cavs = []
    for c in np.unique(labels):
        pts = buried[labels == c]
        cen = pts.mean(0)
        cavs.append((cen, float(np.linalg.norm(pts - cen, axis=1).max())))
    return cavs


def read_box(path):
    kv = {}
    for line in open(path):
        m = re.match(r"\s*(\w+)\s*=\s*([-+0-9.eE]+)", line)
        if m:
            kv[m.group(1).lower()] = float(m.group(2))
    try:
        c = np.array([kv["center_x"], kv["center_y"], kv["center_z"]])
        s = np.array([kv["size_x"], kv["size_y"], kv["size_z"]])
    except KeyError as exc:
        sys.exit(f"{path}: missing {exc} - not an AutoDock Vina box config?")
    return c, s


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb")
    p.add_argument("num_boxes", type=int)
    p.add_argument("--box-pattern", default="box{}.txt")
    p.add_argument("--margin", type=float, default=2.0,
                   help="keep boxes within this distance of a cavity sphere (A)")
    p.add_argument("--grid", type=float, default=1.0)
    p.add_argument("--probe", type=float, default=2.8)
    p.add_argument("--min-neighbors", type=int, default=55)
    p.add_argument("--rays", type=int, default=30)
    p.add_argument("--ray-length", type=float, default=12.0)
    p.add_argument("--ray-radius", type=float, default=2.0)
    p.add_argument("--ray-leak", type=int, default=0)
    p.add_argument("--all-if-error", action="store_true",
                   help="print every box if anything goes wrong (never skip silently)")
    a = p.parse_args()
    try:
        cavs = find_cavities(heavy_atoms(a.pdb), a)
        keep = []
        for k in range(1, a.num_boxes + 1):
            c, s = read_box(a.box_pattern.format(k))
            lo, hi = c - s / 2, c + s / 2
            for cen, rad in cavs:
                # distance from sphere centre to the axis-aligned box
                d = np.linalg.norm(np.maximum(0, np.maximum(lo - cen, cen - hi)))
                if d <= rad + a.margin:
                    keep.append(k)
                    break
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        if a.all_if_error:
            print(f"select_boxes: {exc}; keeping all boxes", file=sys.stderr)
            print(" ".join(str(k) for k in range(1, a.num_boxes + 1)))
            return
        raise
    print(f"select_boxes: {len(cavs)} buried cavities, keeping {len(keep)} of "
          f"{a.num_boxes} boxes", file=sys.stderr)
    print(" ".join(map(str, keep)))


if __name__ == "__main__":
    main()
EOF
    
    patch main.sh <<"EOF"
11c11
<  ./pythonsh prepare_receptor4.py -r $1 -A bonds_hydrogens
---
>  ./pythonsh prepare_receptor4.py -o "$pdbqt" -r $1 -A bonds_hydrogens
14c14
<  ./pythonsh prepare_receptor4.py -r $1 -A hydrogens
---
>  ./pythonsh prepare_receptor4.py -o "$pdbqt" -r $1 -A hydrogens
17c17
<  ./pythonsh prepare_receptor4.py -r $1
---
>  ./pythonsh prepare_receptor4.py -o "$pdbqt" -r $1
19a20
> 
37c38,50
< for kconf in $(seq 1 $num_boxes); do
---
> # after the last prepare_receptor4 call:
> [[ -s "$pdbqt" ]] || { echo "ERROR: receptor preparation failed" >&2; exit 1; }
> # after 'read num_boxes':
> (( num_boxes > 0 )) || { echo "ERROR: no boxes generated" >&2; exit 1; }
> 
> PY=${PY:-python3}     # a Python with numpy + scipy
> keep=$($PY select_boxes.py "$1" "$num_boxes") || { echo "ERROR: box selection failed" >&2; exit 1; }
> if [[ -z "$keep" ]]; then
>     echo "No buried cavities found - no internal waters"
>     exit 0
> fi
> 
> for kconf in $keep; do
EOF
)
