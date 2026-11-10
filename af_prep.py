#!/usr/bin/env python3
"""
af_prep.py - prepare AlphaFold models for CHARMM/NAMD simulation via psfgen.

Pipeline, per model (order matters):
  1. UniProt annotations: cofactors, binding sites, mature chain, covalent mods;
     ligand names mapped to PDB codes through their ChEBI IDs (ligand_resolver.py),
     with LIGAND_NAME_TO_CCD and --name-map as overrides
  2. AlphaFill transplants, moved onto the input model by local superposition,
     filtered to annotated binding sites, best copy per site
  3. Terminal trimming by mature-chain boundaries and pLDDT (UniProt numbering kept)
  4. optional: internal waters with Dowser++ (on the trimmed protein + cofactors)
  5. PDB2PQR + PROPKA protonation, decoded from the hydrogens PDB2PQR adds
  6. Metal-site overrides and disulfide detection
  7. psfgen build script (+ optional solvate/autoionize and NAMD extraBonds
     restraints for metal sites)

Anything a human should look at is flagged REVIEW in <outdir>/<model>/report.txt
and summarised in <outdir>/summary.tsv.

Requirements: python>=3.8, numpy, scipy, networkx, gemmi, requests, pdb2pqr (CLI);
ligand_resolver.py in the same folder; tqdm (optional, progress bars).
Optional: VMD (to run the generated build script); Dowser++ for internal waters.

Example:
  python af_prep.py models/AF-*.pdb --outdir prep --ph 7.0 \
      --toppar ./toppar \
      --solvate --run-vmd \
      --dowser-dir /path/to/dowserplusplus --dowser-path /path/to/R/bin -j 8
"""
import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:                                   # progress bars are optional
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:                    # pragma: no cover
    tqdm = None

## local import
import ligand_resolver
    
    
log = logging.getLogger("af_prep")

# --------------------------------------------------------------------------
# Configuration tables - extend these for your systems
# --------------------------------------------------------------------------

# Endpoints. VERIFY the AlphaFill one against its API docs; it is a template.
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"
ALPHAFILL_URL = "https://alphafill.eu/v1/aff/{acc}"

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
WATER_NAMES = {"HOH", "WAT", "DOD", "TIP3", "TP3M", "TIP", "SOL", "H2O"}  # TP3M: PDB2PQR
METAL_ELEMENTS = {"ZN", "MG", "CA", "MN", "FE", "CU", "NI", "CO", "CD", "NA",
                  "K", "HG", "MO", "W"}

# UniProt ligand names (cofactor comments and binding-site features) -> PDB CCD
# codes.  Matching is case- and whitespace-insensitive.  Every ligand kept must
# fit its pocket in the model (see select_ligands); whether a ligand is a cofactor
# (named in a UniProt cofactor comment) only decides how a failure is reported.
#
# This table is an OVERRIDE: any other UniProt ligand is mapped automatically
# from its ChEBI ID (ligand_resolver.py).  Keep entries here only where the
# automatic mapping is not what you want, e.g.:
#   - deliberate choices: accept either redox state ("NAD(+)": NAD and NAI),
#     because AlphaFill may transplant either and the CHARMM state comes from
#     uniprot_states.json anyway
#   - ligands ChEBI has no structure for: iron-sulfur clusters, generic classes
#     such as "a divalent metal cation"
# <outdir>/ligand_table_audit.tsv compares every entry used in a run with the
# automatic mapping; entries that agree can be removed.
# Unmapped *cofactor* names are flagged REVIEW.
LIGAND_NAME_TO_CCD = {
    # metal ions
    "Zn(2+)": ["ZN"], "Mg(2+)": ["MG"], "Ca(2+)": ["CA"], "Mn(2+)": ["MN"],
    "Fe(2+)": ["FE2", "FE"], "Fe(3+)": ["FE", "FE2"], "Fe cation": ["FE", "FE2"],
    "Cu(2+)": ["CU"], "Cu(+)": ["CU1", "CU"], "Cu cation": ["CU", "CU1"],
    "Ni(2+)": ["NI"], "Co(2+)": ["CO"], "Cd(2+)": ["CD"],
    "K(+)": ["K"], "Na(+)": ["NA"], "chloride": ["CL"],
    "a divalent metal cation": ["MG", "MN", "ZN", "CO", "NI", "FE2", "CA", "CU"],
    # hemes, flavins, nicotinamides (reduced forms: accept either code)
    "heme": ["HEM", "HEC"], "heme b": ["HEM"], "heme c": ["HEC"],
    "FAD": ["FAD", "FDA"], "FMN": ["FMN", "FNR"], "NAD(+)": ["NAD", "NAI"],
    "NADP(+)": ["NAP", "NDP"],
    "NADH": ["NAI", "NAD"], "NADPH": ["NDP", "NAP"],
    "FADH2": ["FDA", "FAD"], "FMNH2": ["FNR", "FMN"],
    # other cofactors
    "pyridoxal 5'-phosphate": ["PLP"], "thiamine diphosphate": ["TPP"],
    "[4Fe-4S] cluster": ["SF4"], "[2Fe-2S] cluster": ["FES"],
    "[3Fe-4S] cluster": ["F3S"], "hybrid [4Fe-2O-2S] cluster": ["FSO"],   # verify FSO
    "Mo-molybdopterin": ["MTE"], "biotin": ["BTN"], "coenzyme A": ["COA"],
    "CoA": ["COA"], "acetyl-CoA": ["ACO"], "glutathione": ["GSH"],
    "S-adenosyl-L-methionine": ["SAM"], "S-adenosyl-L-homocysteine": ["SAH"],
    "(6S)-5,6,7,8-tetrahydrofolate": ["THG"],
    "(6R)-5,10-methylene-5,6,7,8-tetrahydrofolate": ["MHF"],             # verify MHF
    "L-ascorbate": ["ASC"], "(R)-lipoate": ["LPA"], "cob(II)alamin": ["B12"],
    "O2": ["OXY"],
    # nucleotides (CCD GMP is guanosine and CMP is cyclic AMP - hence 5GP, C5P)
    "ATP": ["ATP"], "ADP": ["ADP"], "AMP": ["AMP"],
    "GTP": ["GTP"], "GDP": ["GDP"], "GMP": ["5GP"],
    "CTP": ["CTP"], "CDP": ["CDP"], "CMP": ["C5P"],
    "UTP": ["UTP"], "UDP": ["UDP"],
    # substrates / metabolites
    "2-oxoglutarate": ["AKG"], "succinate": ["SIN"], "phosphoenolpyruvate": ["PEP"],
    "3-phosphoshikimate": ["S3P"], "isochorismate": ["ISC"],
    "(2E,6E)-farnesyl diphosphate": ["FPP"], "(2E)-geranyl diphosphate": ["GPP"],
    "dimethylallyl diphosphate": ["DMA"], "isopentenyl diphosphate": ["IPE"],
    "3'-phosphoadenylyl sulfate": ["PPS"], "UDP-alpha-D-glucose": ["UPG"],
    "dTDP-beta-L-rhamnose": ["TRH"], "GDP-beta-L-fucose": ["GFB"],
    "all-trans-retinoate": ["REA"], "quercetin": ["QUE"],
    "(2R)-3-phosphoglycerate": ["3PG"], "nicotinamide": ["NCA"],
    "adenosine": ["ADN"], "streptomycin": ["SRY"],
    "4-amino-5-hydroxymethyl-2-methylpyrimidine": ["HMH"],
}

# UniProt ligand names to ignore entirely: generic or non-specific names that
# cannot be mapped to one molecule.  Binding sites and cofactor comments with
# these names are skipped (no REVIEW flags, nothing kept).  Extend with
# --ignore-ligand; a name given with --name-map is never ignored.
# Ligands whose ChEBI entry has no usable structure are recognised as generic
# automatically; this list is still needed for names UniProt gives without a
# ChEBI ID ("substrate", "product").
IGNORED_LIGAND_NAMES = {
    "substrate", "product", "a protein", "DNA", "RNA", "tRNA", "mRNA", "rRNA",
    "chitin", "peptidoglycan", "a carbohydrate", "a beta-lactam", "an acyl-CoA",
    "an L-alpha-amino acid", "a 2'-deoxyribonucleoside 5'-triphosphate", "pyridine",
    "pepstatin A",              # inhibitor
}

# (UniProt ligand name, PDB code) pairs that are known coincidences: the
# transplanted molecule happens to sit in a site annotated for a different
# ligand (an ion in a sugar site, a buffer component in a substrate site, a
# fragment of a larger ligand ...).  These are not suggested as name mappings.
# Curate with <outdir>/ligand_name_suggestions.tsv, which lists all suggestions.
# Only consulted for names that remain unmapped (no ChEBI ID, or no CCD match):
# a transplant in the site of a ChEBI-mapped ligand is checked chemically
# instead, so coincidences like an ion in a phosphate site are never suggested.
NOT_A_MAPPING = {
    ("phosphate", "CA"), ("phosphate", "ABU"),
    ("acetate", "NA"),
    ("L-cysteine", "NA"), ("L-cysteine", "MG"), ("L-cysteine", "AMP"),
    ("L-glutamate", "AMP"), ("L-glutamine", "NO3"),
    ("beta-D-glucose", "MG"), ("beta-D-galactose", "NO3"),
    ("alpha-D-glucose 1,6-bisphosphate", "CO"), ("alpha-D-glucose 1,6-bisphosphate", "CA"),
    ("adenosine 3',5'-bisphosphate", "CA"),
    ("a 1,2-diacyl-sn-glycero-3-phospho-(1D-myo-inositol-4,5-bisphosphate)", "MG"),
    ("a 1,2-diacyl-sn-glycero-3-phospho-(1D-myo-inositol-4,5-bisphosphate)", "CA"),
    ("UDP-N-acetyl-alpha-D-muramoyl-L-alanine", "NO3"),
    ("UDP-N-acetyl-alpha-D-muramoyl-L-alanine", "CA"),
    ("oxalate", "ADP"), ("1D-myo-inositol 1,3,4-trisphosphate", "ADP"),
    ("1D-myo-inositol hexakisphosphate", "ATP"),
    ("ADP-alpha-D-glucose", "COA"), ("4-coumaroyl-CoA", "COA"),
    ("N(7)-methyl-GMP", "CTN"), ("N(7)-methyl-GMP", "U5P"),
}
NOT_A_MAPPING.update({("(E)-5-hydroxyferulate", ccd) for ccd in
                      ("FE", "STL", "SLX", "SFG", "SAM", "PPY", "N7I", "EUG", "ENO", "EMO",
                       "DNA", "CIY", "AGI")})
NOT_A_MAPPING.update({("UDP-beta-L-rhamnose", ccd) for ccd in
                      ("KMP", "SAL", "NAR", "MG", "B3P")})

# PDB-code -> CHARMM aliases and UniProt state -> CHARMM residue come from the
# toppar folder (ccd_aliases.json, uniprot_states.json, written by
# toppar_prepare.py), so they always match the topologies psfgen will load.

# Ligands that need bonded models / special parameters in any case.
NEEDS_BONDED = {"SF4", "FES", "F3S", "FSO", "HEC"}
COVALENT_HINTS = ("pyridoxal", "heme", "flavin", "fad", "fmn", "lipoyl",
                  "biotin", "phosphopantetheine", "retinal")



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
        self.candidates = []     # empty sites with their candidate copies (candidates.pdb)

    def review(self, msg):
        log.warning("[%s] REVIEW: %s", self.name, msg)
        self.flags.append(("REVIEW", msg))

    def needs_params(self, code, msg):
        """A kept ligand that could not be built for lack of CHARMM parameters."""
        log.warning("[%s] NEEDS_PARAMS: %s", self.name, msg)
        self.flags.append(("NEEDS_PARAMS", msg))
        self.data.setdefault("needs_params", []).append(code)

    def info(self, msg):
        log.info("[%s] %s", self.name, msg)
        self.flags.append(("INFO", msg))

    def count(self, level):
        return sum(1 for l, _ in self.flags if l == level)

    @property
    def status(self):
        """ERROR > REVIEW > NEEDS_PARAMS > OK"""
        for level in ("ERROR", "REVIEW", "NEEDS_PARAMS"):
            if self.count(level):
                return level
        return "OK"

    def write(self, outdir):
        outdir = Path(outdir)
        (outdir / "report.json").write_text(json.dumps(
            {"model": self.name, "status": self.status,
             "flags": [{"level": l, "msg": m} for l, m in self.flags],
             **self.data}, indent=2, default=str))
        lines = [f"Model: {self.name}", f"Status: {self.status}", ""]
        for lvl in ("ERROR", "REVIEW", "NEEDS_PARAMS", "INFO"):
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


NAME_TO_CCD_NORM = {norm_name(k): v for k, v in LIGAND_NAME_TO_CCD.items()}
LIGAND_NAME_TO_CCD_ORIG = dict(NAME_TO_CCD_NORM)   # before --name-map changes it
NOT_A_MAPPING_NORM = {(norm_name(n), c.upper()) for n, c in NOT_A_MAPPING}


def accession_from_name(path):
    m = re.search(r"AF-([A-Z0-9]+(?:-\d+)?)-F(\d+)", Path(path).name)
    if m:
        return m.group(1), f"AF-{m.group(1)}-F{m.group(2)}"
    return None, None


def accession_from_header(path):
    """UniProt accession from DBREF (PDB) or _struct_ref (mmCIF) records."""
    path = Path(path)
    accs = []
    if path.suffix.lower() in (".cif", ".mmcif"):
        import gemmi
        block = gemmi.cif.read(str(path)).sole_block()
        for tag in ("_struct_ref.pdbx_db_accession",
                    "_ma_target_ref_db_details.db_accession"):
            accs += [gemmi.cif.as_string(v) for v in block.find_values(tag)]
    else:
        with open(path) as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    break
                t = line.split()
                if t[0] == "DBREF" and len(t) >= 7 and t[5] in ("UNP", "SWS", "TREMBL"):
                    accs.append(t[6])
                elif t[0] == "DBREF2" and len(t) >= 4:   # long accessions
                    accs.append(t[3])
                elif t[0].startswith("TITLE") and not accs:
                    m = re.search(r"\(([A-Z0-9]{6,10})\)", line)
                    if m:
                        accs.append(m.group(1))
    accs = list(dict.fromkeys(a for a in accs if a and a not in ("?", ".")))
    return accs


RETRY_STATUS = {429, 500, 502, 503, 504}
HTTP_RETRIES = 6          # set from --retries
HTTP_BACKOFF = 2.0        # seconds; doubles each attempt, plus random jitter


def http_get(url, timeout):
    """GET with retries for dropped connections, timeouts and HTTP 429/5xx
    (exponential backoff with jitter; honours Retry-After).  Other HTTP errors,
    e.g. 404, are returned to the caller without retrying."""
    import random
    import time
    import requests
    for attempt in range(HTTP_RETRIES):
        last = attempt == HTTP_RETRIES - 1
        wait = HTTP_BACKOFF * 2 ** attempt + random.uniform(0, HTTP_BACKOFF)
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "af_prep.py"})
        except (requests.ConnectionError, requests.Timeout):
            if last:
                raise
            time.sleep(wait)
            continue
        if r.status_code in RETRY_STATUS and not last:
            try:
                wait = max(wait, float(r.headers.get("Retry-After", 0)))
            except ValueError:
                pass
            time.sleep(wait)
            continue
        return r


def _atomic_write(path, text):
    """Write via a temporary file, so a parallel reader never sees half a file."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def fetch_uniprot(acc, cache_dir):
    cache = Path(cache_dir) / f"{acc}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = http_get(UNIPROT_URL.format(acc=acc), timeout=30)
    r.raise_for_status()
    _atomic_write(cache, r.text)
    return r.json()


_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7, "eight": 8}


def cofactor_count(note):
    """Copies per subunit from a UniProt cofactor note such as 'Binds 2 Mg(2+)
    ions per subunit.' or 'Binds 1 FAD per subunit.'; 0 if not stated."""
    m = re.search(r"\bBinds\s+(\d+|an?|one|two|three|four|five|six|seven|eight)\b"
                  r"[^.;]*?\bper\s+(?:subunit|monomer|chain|polypeptide)", note or "", re.I)
    if not m:
        return 0
    w = m.group(1).lower()
    return int(w) if w.isdigit() else _NUMBER_WORDS.get(w, 0)


def parse_uniprot(d):
    info = {"cofactors": [], "sites": [], "modified": [], "chains": [],
            "chebi": {},       # UniProt ligand name -> ChEBI ID
            "cofactor_counts": {},   # cofactor name -> copies per subunit (from notes)
            "length": d.get("sequence", {}).get("length"),
            "protein_name": ((d.get("proteinDescription") or {}).get("recommendedName") or
                             {}).get("fullName", {}).get("value", "")}
    for cm in d.get("comments", []):
        if cm.get("commentType") == "COFACTOR":
            note = " ".join(t.get("value", "") for t in (cm.get("note") or {}).get("texts", []))
            n = cofactor_count(note)
            for c in cm.get("cofactors", []):
                if c.get("name"):
                    info["cofactors"].append(c["name"])
                    if n:
                        info["cofactor_counts"][c["name"]] = max(
                            n, info["cofactor_counts"].get(c["name"], 0))
                    xref = c.get("cofactorCrossReference") or {}
                    chebi = (ligand_resolver.normalize_chebi(xref.get("id"))
                             if xref.get("database") == "ChEBI" else None)
                    if chebi:
                        info["chebi"].setdefault(c["name"], chebi)
    for f in d.get("features", []):
        loc = f.get("location", {})
        s = loc.get("start", {}).get("value")
        e = loc.get("end", {}).get("value")
        t = f.get("type")
        if t == "Binding site" and f.get("ligand"):
            chebi = ligand_resolver.normalize_chebi(f["ligand"].get("id"))
            if chebi and f["ligand"].get("name"):
                info["chebi"].setdefault(f["ligand"]["name"], chebi)
            info["sites"].append({"ligand": f["ligand"].get("name"),
                                  "label": f["ligand"].get("label") or "",
                                  "start": s, "end": e})
        elif t in ("Modified residue", "Cross-link", "Lipidation", "Glycosylation"):
            info["modified"].append({"type": t, "pos": s,
                                     "desc": f.get("description", "")})
        elif t == "Chain" and s and e:
            info["chains"].append((s, e))
    return info


CHEBI_CACHE_NAME = "chebi_ccd.json"     # in the --cache folder
_RESOLUTIONS = {}


def chebi_resolutions(cfg):
    """ChEBI ID -> resolution (ligand_resolver.py), read once per process from
    the cache file that resolve_chebi_ids() writes before models are processed."""
    key = str(cfg.cache)
    if key not in _RESOLUTIONS:
        _RESOLUTIONS[key] = ligand_resolver.load_cache(Path(cfg.cache) / CHEBI_CACHE_NAME)
    return _RESOLUTIONS[key]


def ligand_codes(name, chebi_by_name, resolutions):
    """PDB codes for a UniProt ligand name: (codes, how, resolution).
    LIGAND_NAME_TO_CCD and --name-map take precedence; otherwise the codes come
    from the ligand's ChEBI ID.  `how` says where the codes (or their absence)
    come from; `resolution` is the ligand_resolver result, if any."""
    codes = NAME_TO_CCD_NORM.get(norm_name(name))
    chebi = chebi_by_name.get(name)
    res = resolutions.get(chebi) if chebi else None
    if codes:
        return list(codes), "table", res
    if not chebi:
        return [], "no ChEBI ID in UniProt", None
    if res is None:
        return [], f"{chebi} not resolved (ChEBI/RCSB lookup failed; rerun to retry)", None
    if res["codes"]:
        methods = sorted({res["methods"][c] for c in res["codes"]},
                         key=ligand_resolver.METHOD_PRIORITY.get)
        return list(res["codes"]), f"{chebi} via {', '.join(methods)}", res
    if res["status"] == "generic":
        return [], f"{chebi} has no structure in ChEBI (generic class or cluster)", res
    return [], f"no CCD entry matches {chebi}", res


def allowed_ligands(uni, extra, ignore, resolutions, rep):
    """CCD codes to keep, binding-site residues per CCD code, binding-site
    residues of UniProt ligand names that have no CCD mapping, and related
    codes per mapped site (stereo-ambiguous matches and analogs, for
    reporting).  Names in `ignore` (normalised) are skipped; names whose ChEBI
    entry has no structure are treated as generic and skipped too, except
    cofactors, which are flagged REVIEW."""
    allowed = set(extra)
    expected = set()
    ignored = sorted({n for n in uni["cofactors"] + [x["ligand"] for x in uni["sites"]]
                      if norm_name(n) in ignore})
    if ignored:
        rep.info(f"ignored UniProt ligand names: {', '.join(ignored)}")
    uni = dict(uni, cofactors=[n for n in uni["cofactors"] if norm_name(n) not in ignore],
               sites=[x for x in uni["sites"] if norm_name(x["ligand"]) not in ignore])
    chebi_by_name = uni.get("chebi", {})
    mapping = {}

    def codes_for(name):
        if name not in mapping:
            mapping[name] = ligand_codes(name, chebi_by_name, resolutions)
        return mapping[name][0]

    def is_generic(name):
        codes, _, res = mapping[name]
        return not codes and res is not None and res["status"] == "generic"

    for name in uni["cofactors"]:
        ccds = codes_for(name)
        if ccds:
            allowed |= set(ccds)
            expected.add(tuple(ccds))
        else:
            rep.review(f"UniProt cofactor '{name}' has no CCD mapping "
                       f"({mapping[name][1]}); add it to LIGAND_NAME_TO_CCD or use "
                       f"--name-map \"{name}=CODE\"")
    # site_res[ccd][label] = residue numbers. UniProt labels distinguish several
    # copies of the same ligand ("1", "2"); unlabelled features form one site.
    site_res, unmapped, site_names, related = {}, {}, {}, []
    generic = set()
    for s in uni["sites"]:
        ccds = codes_for(s["ligand"])
        residues = set(range(s["start"], s["end"] + 1))
        res = mapping[s["ligand"]][2]
        if res and mapping[s["ligand"]][1] != "table":
            # codes that match only loosely: reported if AlphaFill puts them here
            kinds = {c: "stereo_ambiguous" for c in res.get("stereo_ambiguous", [])}
            for c in res.get("similar", []):
                kinds.setdefault(c, "similar")
            kinds = {c: k for c, k in kinds.items() if c not in ccds}
            if kinds:
                related.append((s["ligand"], residues, kinds, ccds))
        if not ccds:
            if is_generic(s["ligand"]):
                generic.add(s["ligand"])
            else:
                unmapped.setdefault(s["ligand"], set()).update(residues)
            continue
        allowed |= set(ccds)
        for c in ccds:
            site_res.setdefault(c, {}).setdefault(s.get("label", ""), set()).update(
                residues)
            site_names.setdefault(c, {}).setdefault(s.get("label", ""), set()).add(s["ligand"])
    for name in uni["cofactors"]:                 # fallback when a site gives no state
        for c in codes_for(name):
            site_names.setdefault(c, {}).setdefault("*cofactor*", set()).add(name)
    if generic:
        rep.info("UniProt ligands without a defined structure (generic classes), "
                 f"skipped: {', '.join(sorted(generic))}")
    lines = [f"{n} -> {'/'.join(c)} ({how})" for n, (c, how, _) in sorted(mapping.items())
             if c]
    if lines:
        rep.info("UniProt ligands mapped to PDB codes: " + "; ".join(lines))
    rep.data["ligand_mapping"] = {n: {"chebi": chebi_by_name.get(n), "codes": c, "via": how}
                                  for n, (c, how, _) in sorted(mapping.items())}
    return allowed, expected, site_res, unmapped, site_names, related


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
        rep.info(f"no AlphaFill file for {acc} in {d}")
        return None
    try:
        dest = download_alphafill(acc, af_id, url_tmpl, cache_dir)
    except Exception as exc:  # noqa: BLE001
        rep.info(f"AlphaFill download failed ({exc}); no cofactors transplanted")
        return None
    if dest is None:
        rep.info(f"AlphaFill has no entry for {acc}")
    return dest


def download_alphafill(acc, af_id, url_tmpl, cache_dir):
    """Cached AlphaFill mmCIF path, or None if AlphaFill has no entry (remembered
    with an empty '<acc>.alphafill.none' file so it is not asked again)."""
    dest = Path(cache_dir) / f"{acc}.alphafill.cif"
    none = Path(cache_dir) / f"{acc}.alphafill.none"
    if dest.exists():
        return dest
    if none.exists():
        return None
    r = http_get(url_tmpl.format(acc=acc, af_id=af_id), timeout=120)
    if r.status_code == 404:
        none.touch()
        return None
    r.raise_for_status()
    if not r.text.lstrip().startswith("data_"):
        raise ValueError("response is not mmCIF")
    _atomic_write(dest, r.text)
    return dest


def kabsch(P, Q):
    """Rotation R and translation t minimising |(P @ R.T + t) - Q|."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - pc @ R.T


NOSITE = "*nosite*"      # label prefix of pseudo-sites for cofactors without residues


SYMMETRIC_ATOMS = {"PHE": [("CD1", "CD2"), ("CE1", "CE2")],
                   "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
                   "ASP": [("OD1", "OD2")], "GLU": [("OE1", "OE2")],
                   "ARG": [("NH1", "NH2")]}


def _pocket_rmsd(res_keys, af_atoms, in_atoms, R, t):
    """Heavy-atom RMSD of the given residues after superposition, taking the
    better of the two namings for chemically symmetric side-chain atoms."""
    sq, n = 0.0, 0
    for key in res_keys:
        a = af_atoms.get(key, {})
        b = in_atoms.get(key, {})
        names = [nm for nm in a if nm in b and not nm.startswith("_")]
        if not names:
            continue
        pa = np.array([a[nm] for nm in names]) @ R.T + t
        best = np.sum((pa - np.array([b[nm] for nm in names])) ** 2)
        swaps = SYMMETRIC_ATOMS.get(a["_resname"], []) if "_resname" in a else []
        if swaps:
            alt = {}
            for x, y in swaps:
                alt[x], alt[y] = y, x
            names2 = [alt.get(nm, nm) for nm in names]
            if all(nm in b for nm in names2):
                best = min(best, np.sum((pa - np.array([b[nm] for nm in names2])) ** 2))
        sq += best
        n += len(names)
    return math.sqrt(sq / n) if n else float("nan")


def transplant_ligands(ligs, af_protein, protein, cfg, rep):
    """Move each AlphaFill ligand onto the input model by superposing the CA atoms
    of residues within cfg.fit_radius of it (AlphaFill's own approach), then check
    that the pocket really is the same in both structures:
      * backbone: local CA RMSD after the fit           (--max-fit-rmsd)
      * side chains: heavy-atom RMSD of pocket residues (--max-pocket-rmsd)
      * contacts: fraction of AlphaFill's ligand-protein contacts that the input
        model reproduces                                 (--min-contact-frac)"""
    def per_residue(atoms):
        d = {}
        for a in heavy(atoms):
            r = d.setdefault((a.chain, a.resnum), {"_resname": a.resname})
            r[a.name] = np.array(a.xyz)
        return d
    af_res, in_res = per_residue(af_protein), per_residue(protein)
    keys = [k for k in af_res if "CA" in af_res[k] and k in in_res and "CA" in in_res[k]]
    if len(keys) < 3:
        rep.review("AlphaFill and input model share too few residues; ligands dropped")
        return OrderedDict(), {}, []
    af_ca = np.array([af_res[k]["CA"] for k in keys])
    in_ca = np.array([in_res[k]["CA"] for k in keys])
    ca_tree = cKDTree(af_ca)
    af_heavy, in_heavy = heavy(af_protein), heavy(protein)
    af_tree, in_tree = cKDTree(coords(af_heavy)), cKDTree(coords(in_heavy))
    ident = lambda a: (a.chain, a.resnum, a.name)
    out, metrics, poor, rejected = OrderedDict(), {}, [], []

    def reject(key, atoms, R, t, why):
        """Keep a not-transferred copy, placed with its fit, for candidates.pdb."""
        la = heavy(atoms) or atoms
        new = coords(la) @ R.T + t
        rejected.append({"key": key, "why": why,
                         "atoms": [replace(a, x=float(p[0]), y=float(p[1]), z=float(p[2]))
                                   for a, p in zip(la, new)]})

    for key, atoms in ligs.items():
        tag = f"{key[3]} {key[0]}{key[1]}"
        xyz = coords(heavy(atoms) or atoms)
        idx = sorted({i for lst in ca_tree.query_ball_point(xyz, cfg.fit_radius) for i in lst})
        if len(idx) < 3:
            poor.append(f"{tag}: no nearby residues")
            continue
        R, t = kabsch(af_ca[idx], in_ca[idx])
        fit = float(np.sqrt(np.mean(np.sum((af_ca[idx] @ R.T + t - in_ca[idx]) ** 2, 1))))
        if fit > cfg.max_fit_rmsd:
            poor.append(f"{tag}: backbone fit {fit:.2f} A")
            reject(key, atoms, R, t, f"not transferred: backbone fit {fit:.2f} A")
            continue
        new_xyz = xyz @ R.T + t
        # pocket residues and contacts in AlphaFill's frame
        af_near = {i for lst in af_tree.query_ball_point(xyz, cfg.contact_dist) for i in lst}
        af_contacts = {ident(af_heavy[i]) for i in af_near}
        # pocket = residues near the ligand in EITHER structure, so a side chain
        # that moved into or out of the site in one of them is compared too
        pocket = sorted({(af_heavy[i].chain, af_heavy[i].resnum) for lst in
                         af_tree.query_ball_point(xyz, cfg.pocket_radius) for i in lst} |
                        {(in_heavy[i].chain, in_heavy[i].resnum) for lst in
                         in_tree.query_ball_point(new_xyz, cfg.pocket_radius) for i in lst})
        prmsd = _pocket_rmsd(pocket, af_res, in_res, R, t)
        if prmsd == prmsd and prmsd > cfg.max_pocket_rmsd:      # not NaN
            poor.append(f"{tag}: pocket side chains {prmsd:.2f} A")
            reject(key, atoms, R, t, f"not transferred: pocket side chains {prmsd:.2f} A")
            continue
        in_contacts = {ident(in_heavy[i]) for lst in
                       in_tree.query_ball_point(new_xyz, cfg.contact_dist) for i in lst}
        frac = len(af_contacts & in_contacts) / len(af_contacts) if af_contacts else 1.0
        if frac < cfg.min_contact_frac:
            poor.append(f"{tag}: {frac:.0%} of contacts kept")
            reject(key, atoms, R, t, f"not transferred: {frac:.0%} of contacts kept")
            continue
        out[key] = [replace(a, x=float(p[0]), y=float(p[1]), z=float(p[2]))
                    for a, p in zip(heavy(atoms) or atoms, new_xyz)]
        metrics[key] = (f"backbone fit {fit:.2f} A, pocket {prmsd:.2f} A, "
                        f"{frac:.0%} of contacts kept")
    cfg._transfer_drops = bool(poor)
    if poor:
        rep.info(f"{len(poor)} AlphaFill ligand(s) not transferred because the pocket "
                 f"differs between models: {'; '.join(poor[:10])}"
                 + (" ..." if len(poor) > 10 else ""))
    return out, metrics, rejected


# Typical metal-donor distances (A), from small-molecule and protein structure
# surveys; a protein atom counts as a donor up to the ideal distance + --donor-tol.
METAL_IDEAL = {
    "ZN": {"S": 2.33, "N": 2.08, "O": 2.08}, "MG": {"O": 2.08, "N": 2.20},
    "CA": {"O": 2.40, "N": 2.50}, "MN": {"O": 2.18, "N": 2.25, "S": 2.40},
    "FE": {"O": 2.10, "N": 2.15, "S": 2.30}, "CO": {"O": 2.10, "N": 2.12, "S": 2.30},
    "NI": {"O": 2.08, "N": 2.10, "S": 2.20}, "CU": {"O": 2.00, "N": 2.02, "S": 2.20},
    "CD": {"O": 2.30, "N": 2.30, "S": 2.50}, "HG": {"S": 2.40, "N": 2.30, "O": 2.40},
    "NA": {"O": 2.42, "N": 2.50}, "K": {"O": 2.80, "N": 2.90},
}
# protein atoms that can coordinate a metal (side-chain O/N/S and backbone O)
DONOR_NAMES = {"O", "OXT", "OT1", "OT2", "OD1", "OD2", "OE1", "OE2", "SG", "SD",
               "OG", "OG1", "OH"}


def metal_ideal(metal, donor_el):
    return METAL_IDEAL.get(metal, {}).get(donor_el, 2.2)


def metal_reach(metal, cfg):
    """Largest donor distance counted for this metal."""
    return max(METAL_IDEAL.get(metal, {"X": 2.2}).values()) + cfg.donor_tol


def is_donor(a):
    """Can this protein atom coordinate a metal? (not backbone or amide N)"""
    if a.name in DONOR_NAMES:
        return True
    return a.name in ("ND1", "NE2") and a.resname in ("HIS", "HSD", "HSE", "HSP")


def metal_donors(p, metal, atoms, tree, cfg):
    """Indices of donor atoms within ideal distance + tolerance of point p."""
    out = set()
    for i in tree.query_ball_point(p, metal_reach(metal, cfg)):
        a = atoms[i]
        if is_donor(a) and np.linalg.norm(p - np.array(a.xyz)) <= \
                metal_ideal(metal, a.element) + cfg.donor_tol:
            out.add(i)
    return out


def refine_metal(p0, metal, atoms, tree, cfg):
    """Move a transplanted metal ion by at most cfg.metal_shift A to where its
    potential donors sit at ideal distances, keeping other atoms at a distance.
    Corrects the small mismatch between the homolog's coordination geometry and
    the model's side chains.  Returns (position, shift, donor indices); the
    position is only changed if that does not lose donors."""
    p0 = np.asarray(p0, dtype=float)
    before = metal_donors(p0, metal, atoms, tree, cfg)
    if cfg.metal_shift <= 0:
        return p0, 0.0, before
    reach = metal_reach(metal, cfg) + cfg.metal_shift
    near = tree.query_ball_point(p0, reach + 1.0)
    don = [i for i in near if is_donor(atoms[i])
           and np.linalg.norm(p0 - np.array(atoms[i].xyz)) <= reach]
    if not don:
        return p0, 0.0, before
    X = coords([atoms[i] for i in don])
    D = np.array([metal_ideal(metal, atoms[i].element) for i in don])
    oth = [i for i in near if i not in set(don)]
    Y = coords([atoms[i] for i in oth])
    R = np.array([2.9 if atoms[i].element == "C" else 2.6 for i in oth])
    cap = 0.6 ** 2          # donors that cannot be reached stop pulling

    def f(p):
        d = np.linalg.norm(X - p, axis=1)
        e = np.minimum((d - D) ** 2, cap).sum()
        if len(Y):
            e += 10 * (np.maximum(0, R - np.linalg.norm(Y - p, axis=1)) ** 2).sum()
        e += 10 * max(0.0, np.linalg.norm(p - p0) - cfg.metal_shift) ** 2
        return e

    from scipy.optimize import minimize
    best = min((minimize(f, p0 + dp, method="Nelder-Mead",
                         options={"xatol": 1e-3, "fatol": 1e-5})
                for dp in [np.zeros(3)] + [0.4 * v for v in np.eye(3)] +
                [-0.4 * v for v in np.eye(3)]), key=lambda r: r.fun)
    p = best.x
    if np.linalg.norm(p - p0) > cfg.metal_shift + 0.05:
        return p0, 0.0, before
    after = metal_donors(p, metal, atoms, tree, cfg)
    if len(after) < len(before) or (len(after) == len(before) and f(p) >= f(p0)):
        return p0, 0.0, before
    return p, float(np.linalg.norm(p - p0)), after


def _sphere_dirs(n=60):
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), math.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


_DIRS = _sphere_dirs()


def water_space(p, metal, tree):
    """Is there room for a water molecule at bonding distance from the ion,
    i.e. a point at the metal-O distance with no protein heavy atom within 2.6 A?
    (The water could come from solvation or from Dowser++.)"""
    pts = np.asarray(p) + metal_ideal(metal, "O") * _DIRS
    d, _ = tree.query(pts)
    return bool(np.any(d >= 2.6))


def metal_element(ccd):
    el = re.sub(r"\d", "", ccd)
    return el if el in METAL_ELEMENTS else None


def place_metal_from_site(metal, ch, residues, atoms, tree, cfg):
    """Position for a metal ion at an annotated site without a usable copy:
    refine from the centroids of donor pairs/triples of the annotated residues.
    Returns (position, donor indices) of the best try, or (None, set())."""
    from itertools import combinations
    import types
    don = [i for i, a in enumerate(atoms) if a.chain == ch and a.resnum in residues
           and is_donor(a)]
    if len({atoms[i].resnum for i in don}) < 2:
        return None, set()
    X = {i: np.array(atoms[i].xyz) for i in don}
    starts = [np.mean(list(X.values()), axis=0)]
    for k in (3, 2):
        for c in combinations(don, k):
            if len({atoms[i].resnum for i in c}) == k and \
                    max(np.linalg.norm(X[a] - X[b]) for a, b in combinations(c, 2)) < 6.0:
                starts.append(np.mean([X[i] for i in c], axis=0))
            if len(starts) > 80:
                break
    wide = types.SimpleNamespace(**{**vars(cfg), "metal_shift": cfg.denovo_shift})
    best, best_don = None, set()
    for p0 in starts:
        p, _, d = refine_metal(p0, metal, atoms, tree, wide)
        if tree.query(p)[0] < cfg.metal_clash:
            continue
        if len(d) > len(best_don):
            best, best_don = p, d
    return best, best_don


def select_ligands(protein, ligands, allowed, site_res, unmapped, related, fit_rmsd,
                   cofactor_ccds, cfg, rep, rejected=()):
    """Keep ligands that sit in a UniProt-annotated binding site for that ligand
    and fit the model's pocket (no clashes, buried, covering the site), at most
    one copy per site and chain."""
    from collections import Counter
    ph = heavy(protein)
    tree = cKDTree(coords(ph))
    chains = sorted({a.chain for a in ph})
    chain_res = {}
    for a in ph:
        chain_res.setdefault(a.chain, set()).add(a.resnum)
    cands, no_sites = [], set()
    not_in_uniprot, unmapped_hits = Counter(), {}
    related_hits = Counter()      # (name, code, kind): loose matches in mapped sites
    others = []                   # not selectable, but listed as candidates for review
    NEAR = 8.0                    # A: how close a candidate must be to an empty site

    def site_touch(xyz, groups, radius):
        near = {(ph[i].chain, ph[i].resnum)
                for lst in tree.query_ball_point(xyz, radius) for i in lst}
        touched = {}
        for (ch, rn) in near:
            for label, res in groups.items():
                if rn in res:
                    touched[(ch, label)] = touched.get((ch, label), 0) + 1
        return {k: n / len(groups[k[1]]) for k, n in touched.items()}

    for key, atoms in ligands.items():
        ccd = key[3]
        tag = f"{ccd} {key[0]}{key[1]}"
        if ccd not in allowed:
            # UniProt does not mention it - or mentions it under a name with no
            # CCD mapping: check whether it sits in such a binding site
            xyz = coords(heavy(atoms) or atoms)
            near = {ph[i].resnum for lst in tree.query_ball_point(xyz, cfg.site_contact)
                    for i in lst}
            hits = [n for n, res in unmapped.items() if near & res
                    and (norm_name(n), ccd) not in NOT_A_MAPPING_NORM]
            # a ChEBI-mapped site holding a code that matches only loosely
            # (stereo left undefined in the CCD entry, or an analog)
            rel = [(n, kinds[ccd], codes) for n, res, kinds, codes in related
                   if ccd in kinds and near & res]
            for n in hits:
                unmapped_hits.setdefault((n, ccd), 0)
                unmapped_hits[(n, ccd)] += 1
            for n, kind, codes in rel:
                related_hits[(n, ccd, kind, tuple(codes))] += 1
            if not hits and not rel:
                not_in_uniprot[ccd] += 1
            continue
        groups = site_res.get(ccd)
        if not groups:
            no_sites.add(ccd)
            continue
        la = heavy(atoms) or atoms
        metal = len(la) == 1 and la[0].element in METAL_ELEMENTS
        shift, donors, xyz0 = 0.0, set(), None
        if metal:
            xyz0 = np.array(la[0].xyz)          # transplanted position (partner geometry)
            p, shift, donors = refine_metal(la[0].xyz, la[0].element, ph, tree, cfg)
            la = [replace(la[0], x=float(p[0]), y=float(p[1]), z=float(p[2]))]
            atoms = la
        xyz = coords(la)
        d, _ = tree.query(xyz)
        # metal atoms (also inside clusters and hemes) bond to protein N/O/S at
        # ~2.0-2.4 A, so they only clash below --metal-clash
        thr = np.array([cfg.metal_clash if a.element in METAL_ELEMENTS else cfg.clash
                        for a in la])
        clashes = int(np.sum(d < thr))
        burial = float(np.mean(d <= cfg.burial_dist))
        near = {(ph[i].chain, ph[i].resnum)
                for lst in tree.query_ball_point(xyz, cfg.site_contact) for i in lst}
        nosite = all(lb.startswith(NOSITE) for lb in groups)
        if nosite:
            # cofactor without annotated residues: any chain the copy sits in;
            # it must pass the fit tests (and, for metals, coordination) instead
            chs = {ch for ch, _ in near} or {ph[int(tree.query(xyz[0])[1])].chain}
            coverage = {(ch, lb): 1.0 for ch in chs for lb in groups}
        else:
            # coverage of each annotated site, per chain (homomers: each chain has it)
            touched = {}
            for (ch, rn) in near:
                for label, res in groups.items():
                    if rn in res:
                        touched[(ch, label)] = touched.get((ch, label), 0) + 1
            if not touched:
                rep.info(f"dropped {tag} (touches no annotated binding-site residue)")
                wide = site_touch(xyz, groups, NEAR)
                if wide:
                    others.append(dict(key=key, ccd=ccd, atoms=atoms, xyz=xyz, metal=metal,
                                       el=[a.element for a in la], shift=shift, nosite=False,
                                       clashes=clashes, burial=burial, coverage=wide,
                                       donors=donors, stage=1, tag=tag,
                                       why=f"no annotated residue within {cfg.site_contact} A",
                                       cofactor=ccd in cofactor_ccds))
                continue
            coverage = {k: n / len(groups[k[1]]) for k, n in touched.items()}
        cands.append(dict(key=key, ccd=ccd, atoms=atoms, xyz=xyz, metal=metal,
                          el=[a.element for a in la], shift=shift, nosite=nosite,
                          clashes=clashes, burial=burial, coverage=coverage, tag=tag,
                          donors=donors, cofactor=ccd in cofactor_ccds, xyz0=xyz0,
                          water=metal and water_space(xyz[0], la[0].element, tree)))
    # copies rejected during transfer: not selectable, but listed as candidates
    # for empty sites (placed with their fit, metal positions refined)
    for r in rejected:
        ccd = r["key"][3]
        groups = site_res.get(ccd)
        if ccd not in allowed or not groups:
            continue
        la = r["atoms"]
        metal = len(la) == 1 and la[0].element in METAL_ELEMENTS
        shift, donors = 0.0, set()
        if metal:
            p, shift, donors = refine_metal(la[0].xyz, la[0].element, ph, tree, cfg)
            la = [replace(la[0], x=float(p[0]), y=float(p[1]), z=float(p[2]))]
        xyz = coords(la)
        d, _ = tree.query(xyz)
        thr = np.array([cfg.metal_clash if a.element in METAL_ELEMENTS else cfg.clash
                        for a in la])
        near = {(ph[i].chain, ph[i].resnum)
                for lst in tree.query_ball_point(xyz, cfg.site_contact) for i in lst}
        nosite = all(lb.startswith(NOSITE) for lb in groups)
        if nosite:
            chs = {ch for ch, _ in near} or {ph[int(tree.query(xyz[0])[1])].chain}
            coverage = {(ch, lb): 1.0 for ch in chs for lb in groups}
        else:
            coverage = site_touch(xyz, groups, NEAR)
            if not coverage:
                continue
        others.append(dict(key=r["key"], ccd=ccd, atoms=la, xyz=xyz, metal=metal,
                           el=[a.element for a in la], shift=shift, nosite=nosite,
                           clashes=int(np.sum(d < thr)), burial=float(np.mean(d <= cfg.burial_dist)),
                           coverage=coverage, donors=donors, stage=1, why=r["why"],
                           tag=f"{ccd} {r['key'][0]}{r['key'][1]}",
                           cofactor=ccd in cofactor_ccds))

    if not_in_uniprot:
        rep.info("dropped AlphaFill ligands not mentioned in UniProt: " +
                 ", ".join(f"{c} x{n}" for c, n in not_in_uniprot.most_common()))
    rep.data["name_suggestions"] = [[name, ccd, n] for (name, ccd), n
                                    in sorted(unmapped_hits.items())]
    for (name, ccd), n in sorted(unmapped_hits.items()):
        rep.review(f"{n} AlphaFill {ccd} copy(ies) sit in the UniProt binding site for "
                   f"'{name}', which has no CCD mapping - to keep them, run with "
                   f"--name-map \"{name}={ccd}\" (or add it to LIGAND_NAME_TO_CCD)")
    rep.data["related_ligands"] = [[name, ccd, kind, n] for (name, ccd, kind, _), n
                                   in sorted(related_hits.items())]
    for (name, ccd, kind, codes), n in sorted(related_hits.items()):
        fix = f"--name-map \"{name}={','.join(codes + (ccd,))}\""
        if kind == "stereo_ambiguous":
            rep.review(f"{n} AlphaFill {ccd} copy(ies) sit in the binding site for "
                       f"'{name}' and match it only when stereochemistry is ignored "
                       f"(the CCD entry may leave it undefined); not kept - to keep "
                       f"them, run with {fix}")
        else:
            rep.info(f"{n} AlphaFill {ccd} copy(ies) in the binding site for '{name}' "
                     f"are a related compound (analog), not kept; {fix} would keep them")
    for ccd in sorted(no_sites):          # e.g. --keep-ligands codes without a site
        (rep.review if ccd in cofactor_ccds else rep.info)(
            f"{ccd} has no annotated binding site; no copy kept (place it manually if "
            "needed)")

    # Fit tests on the model's pocket: a pocket that is apo-like (collapsed or
    # opened up) shows up as clashes, low burial or low coverage of the site.
    def failures(c, cov):
        f = []
        if c["clashes"] > cfg.max_clash:
            f.append(f"{c['clashes']} clashing atoms")
        if c["burial"] < cfg.min_burial:
            f.append(f"burial {c['burial']:.0%}")
        if cov < cfg.min_site_coverage:
            f.append(f"site coverage {cov:.0%}")
        n = len(c["donors"])
        if c["nosite"] and c["metal"] and n < cfg.min_metal_donors and not (
                n == cfg.min_metal_donors - 1 and c.get("water")):
            f.append(f"only {n} protein donor(s)")
        return f

    # cofactors first (a substrate must not take the place of the cofactor it
    # needs), then best-fitting copies
    cands.sort(key=lambda c: (not c["cofactor"],
                              bool(failures(c, max(c["coverage"].values()))),
                              -max(c["coverage"].values()), c["clashes"]))
    # annotated sites of the same ligand that share residues: alternative or
    # adjacent positions; a second copy is only allowed where both can coexist
    def overlapping(ccd, l1, l2):
        g = site_res[ccd]
        return l1 != l2 and bool(g[l1] & g[l2])

    def overlaps(c, k):
        """Do copies c and k occupy the same space?  A metal atom bonds to ligand
        O/N/S at ~2 A, so pairs involving a metal only count below --metal-clash."""
        D = np.linalg.norm(c["xyz"][:, None] - k["xyz"][None], axis=2)
        mc = np.array([e in METAL_ELEMENTS for e in c["el"]])
        mk = np.array([e in METAL_ELEMENTS for e in k["el"]])
        thr = np.where(mc[:, None] | mk[None, :], cfg.metal_clash, cfg.overlap)
        return bool(np.any(D < thr))

    def coexists(c, k):
        """Can copy c sit next to kept copy k of the same ligand?"""
        if c["metal"] and k["metal"]:
            sep = float(np.linalg.norm(c["xyz"][0] - k["xyz"][0]))
            own = c["donors"] - k["donors"]
            if sep < cfg.metal_min_sep:
                return False, f"{sep:.1f} A from {k['tag']}"
            if len(c["donors"]) < 2 or not own:
                return False, (f"no protein coordination of its own next to {k['tag']} "
                               f"({len(c['donors'])} donors, {len(own)} not shared)")
            return True, ""
        return False, f"site overlaps the one filled by {k['tag']}"

    kept, filled, best_reject, covered = [], set(), {}, {}
    for c in cands:
        open_sites = [k for k in c["coverage"] if (c["ccd"],) + k not in filled]
        if not open_sites:
            c["why"] = "site already has a better-placed copy"
            rep.info(f"dropped {c['tag']} (site already has a better-placed copy)")
            continue
        # best coverage; among equals, the site fewest other copies could fill
        # (so e.g. two ions of a binuclear site each get their own site)
        later = cands[cands.index(c) + 1:]
        site = max(open_sites, key=lambda k: (
            c["coverage"][k], -sum(k in o["coverage"] for o in later if o["ccd"] == c["ccd"])))
        cov = c["coverage"][site]
        fail = failures(c, cov)
        if fail:
            c["why"] = "does not fit: " + ", ".join(fail)
            for k in c["coverage"]:
                best_reject.setdefault((c["ccd"],) + k, ", ".join(fail))
            rep.info(f"dropped {c['tag']} (does not fit the model's pocket: "
                     f"{', '.join(fail)})")
            continue
        if any(overlaps(c, k) for k in kept):
            c["why"] = "overlaps a kept ligand"
            rep.info(f"dropped {c['tag']} (overlaps a kept ligand)")
            continue
        # same ligand already placed in an overlapping site of this chain?
        clash_with = [k for k in kept if k["ccd"] == c["ccd"] and k["site"][0] == site[0]
                      and overlapping(c["ccd"], k["site"][1], site[1])]
        if c["metal"]:   # metal ions too close to coexist, whatever the labels
            clash_with += [k for k in kept if k["metal"] and k not in clash_with and
                           np.linalg.norm(c["xyz"][0] - k["xyz"][0]) < cfg.metal_min_sep]
        verdicts = [coexists(c, k) for k in clash_with]
        if any(not ok for ok, _ in verdicts):
            why = next(w for ok, w in verdicts if not ok)
            c["why"] = why
            covered.setdefault((c["ccd"],) + site, clash_with[0]["tag"])
            rep.info(f"dropped {c['tag']} ({why})")
            continue
        filled.add((c["ccd"],) + site)
        c["site"] = site
        c["site_label"] = site[1]
        kept.append(c)
        label = f" site {site[1]}" if site[1] and not site[1].startswith(NOSITE) else ""
        fit = fit_rmsd.get(c["key"])
        fit = f", {fit}" if fit else ""
        coord = (f", {len(c['donors'])} protein donors" if c["metal"] else "") + \
            (f", moved {c['shift']:.1f} A to fit its donors" if c["shift"] >= 0.05 else "") + \
            (", no annotated site: placed by fit" + (" and coordination" if c["metal"] else "")
             if c["nosite"] else "")
        c["kept_msg"] = (f"kept {c['tag']} at chain {site[0]}{label} (burial "
                         f"{c['burial']:.0%}, site coverage {cov:.0%}, clashes "
                         f"{c['clashes']}{coord}{fit})")

    # Metal ions held by few protein atoms are usually completed by a substrate
    # or nucleotide (e.g. Mg.ATP): keep them only if such a partner is kept too.
    low_coord, dropped, warn = {}, set(), []
    for c in [k for k in kept if k["metal"] and len(k["donors"]) < cfg.min_metal_donors]:
        m_el = c["el"][0]
        # partners must be able to anchor the ion themselves (e.g. ATP), not small
        # molecules that bind only to the metal (O2, NO, CO, acetate ...)
        def partners_at(pos):
            return [k["tag"] for k in kept if not k["metal"]
                    and len(k["el"]) >= cfg.min_partner_atoms and any(
                        el in ("O", "N", "S") and np.linalg.norm(p - pos)
                        <= metal_ideal(m_el, el) + cfg.donor_tol
                        for el, p in zip(k["el"], k["xyz"]))]
        partners = partners_at(c["xyz"][0])
        n = len(c["donors"])
        if not partners and c.get("xyz0") is not None and c["shift"] > 0:
            # the refinement only sees protein donors and may have pulled the ion
            # away from its partner: keep it where it was transplanted, with it
            partners = partners_at(c["xyz0"])
            if partners:
                p0 = c["xyz0"]
                c["atoms"] = [replace(c["atoms"][0], x=float(p0[0]), y=float(p0[1]),
                                      z=float(p0[2]))]
                c["xyz"] = coords(c["atoms"])
                c["kept_msg"] = c["kept_msg"].replace(
                    f", moved {c['shift']:.1f} A to fit its donors", "")
                c["shift"] = 0.0
                c["kept_msg"] += "; kept at its transplanted position"
        if partners:
            c["kept_msg"] += f"; bridged by {', '.join(partners)}"
            continue
        if n == cfg.min_metal_donors - 1 and c.get("water"):
            c["water_completed"] = True
            warn.append(f"{c['tag']} kept with only {n} protein donors: room for water at "
                        "bonding distance to complete its coordination - check the site")
            continue
        kept.remove(c)
        dropped.add(c["tag"])
        c["why"] = f"only {n} protein donor(s), no partner ligand kept"
        filled.discard((c["ccd"],) + c["site"])
        low_coord[(c["ccd"],) + c["site"]] = (
            f"{c['tag']} dropped: only {n} protein donor(s) within bonding distance "
            "and no partner ligand kept (substrate-bridged ion, or a site completed "
            "by waters?)")
    for c in kept:
        if c["nosite"] and c["metal"] and len(c["donors"]) < cfg.min_metal_donors \
                and not c.get("water_completed"):
            c["water_completed"] = True
            warn.append(f"{c['tag']} kept with only {len(c['donors'])} protein donors: room "
                        "for water at bonding distance to complete its coordination - check")

    # Metal sites still empty: place the ion from the annotated residues' donor
    # atoms when no transplanted copy could be used.
    for ccd, groups in site_res.items():
        metal = metal_element(ccd)
        if not metal:
            continue
        for label, residues in groups.items():
            if label.startswith(NOSITE):
                continue
            for ch in chains:
                if (ccd, ch, label) in filled or not residues & chain_res.get(ch, set()):
                    continue
                if any(k["ccd"] == ccd and k["site"][0] == ch and
                       overlapping(ccd, k["site"][1], label) for k in kept):
                    continue
                p, don = place_metal_from_site(metal, ch, residues, ph, tree, cfg)
                if p is None:
                    continue
                n = len(don)
                water = water_space(p, metal, tree)
                if n < cfg.min_metal_donors and not (n == cfg.min_metal_donors - 1 and water):
                    continue
                if any(k["metal"] and np.linalg.norm(p - k["xyz"][0]) < cfg.metal_min_sep
                       for k in kept) or any(
                        np.min(np.linalg.norm(k["xyz"] - p, axis=1)) < cfg.metal_clash
                        for k in kept):
                    continue
                k_new = sum(1 for k in kept if k["key"][0] == "new") + 1
                atom = Atom("HETATM", metal, ccd, ch, k_new, "", float(p[0]), float(p[1]),
                            float(p[2]), 0.0, metal)
                c = dict(key=("new", k_new, "", ccd), ccd=ccd, atoms=[atom],
                         xyz=coords([atom]), metal=True, el=[metal], shift=0.0,
                         nosite=False, clashes=0, burial=1.0,
                         coverage={(ch, label): 1.0}, donors=don, water=water,
                         cofactor=ccd in cofactor_ccds, tag=f"{ccd} new{k_new}",
                         site=(ch, label), site_label=label)
                lb = f" site {label}" if label else ""
                c["kept_msg"] = (f"placed {ccd} at chain {ch}{lb} from the annotated "
                                 f"residues' donor atoms (no usable AlphaFill copy; "
                                 f"{n} protein donors)")
                if n < cfg.min_metal_donors:
                    c["water_completed"] = True
                    warn.append(f"{c['tag']} placed with only {n} protein donors: room for "
                                "water at bonding distance to complete its coordination - "
                                "check the site")
                kept.append(c)
                filled.add((ccd, ch, label))
                low_coord.pop((ccd, ch, label), None)
    # A non-cofactor ligand coordinating the position of a metal ion that was
    # not placed (e.g. O2 on the iron of a dioxygenase) would bind only with the
    # metal: drop it as well.
    kept_metals = [k for k in kept if k["metal"]]
    missing = [m for m in cands + others if m["metal"] and id(m) not in
               {id(k) for k in kept_metals}]
    for c in [k for k in kept if not k["metal"] and not k["cofactor"]]:
        for el, p in zip(c["el"], c["xyz"]):
            if el not in ("O", "N", "S"):
                continue
            near = [m for m in missing if np.linalg.norm(p - m["xyz"][0])
                    <= metal_ideal(m["el"][0], el) + cfg.donor_tol]
            if not near or any(np.linalg.norm(p - k["xyz"][0]) <=
                               metal_ideal(k["el"][0], el) + cfg.donor_tol
                               for k in kept_metals):
                continue
            kept.remove(c)
            filled.discard((c["ccd"],) + c["site"])
            c["why"] = f"coordinates the position of {near[0]['tag']}, which was not placed"
            for k in c["coverage"]:
                best_reject.setdefault((c["ccd"],) + k, c["why"])
            rep.info(f"dropped {c['tag']} ({c['why']}; it would bind only with the metal)")
            break
    for c in kept:
        rep.info(c["kept_msg"])
    for w in warn:
        rep.review(w)

    # Empty sites with the copies that were considered for them, for review:
    # written to candidates.pdb (B-factor = site number, occupancy = rank).
    kept_ids = {id(k) for k in kept}

    def rank_key(c):
        cov = max(c["coverage"].values())
        return (c.get("stage", 0), len(failures(c, cov)), -len(c["donors"]),
                c["clashes"], -cov, -c["burial"])

    def candidates_note(ccd, ch, label, residues):
        pool = [c for c in cands + others if c["ccd"] == ccd and id(c) not in kept_ids
                and (ch, label) in c["coverage"]]
        idx = len(rep.candidates) + 1
        rep.candidates.append(dict(
            site=idx, ccd=ccd, chain=ch, label=label, residues=sorted(residues),
            cofactor=ccd in cofactor_ccds, cands=sorted(pool, key=rank_key)))
        if not pool:
            return f"; no candidates, site {idx} in candidates.tcl"
        return f"; {len(pool)} candidate(s) in candidates.pdb, site {idx}"

    # cofactors without annotated residues that got no copy at all
    for ccd, groups in site_res.items():
        if not all(lb.startswith(NOSITE) for lb in groups):
            continue
        if any(k["ccd"] == ccd for k in kept):
            continue
        why = next((w for k, w in best_reject.items() if k[0] == ccd), None) or \
            next((w for k, w in low_coord.items() if k[0] == ccd), None)
        why = (f"best copy did not fit: {why}" if why else
               "no transplanted copy" if not any(c["ccd"] == ccd for c in cands) else
               "no copy passed")
        lbl = sorted(groups)[0]
        chs = sorted({k[0] for c in cands + others if c["ccd"] == ccd for k in c["coverage"]})
        note = "".join(candidates_note(ccd, ch, lbl, []) for ch in chs or chains[:1])
        (rep.review if ccd in cofactor_ccds else rep.info)(
            f"{ccd} is annotated in UniProt but without binding-site residues; none "
            f"placed ({why}){note}")

    # annotated sites (per chain) that ended up without a ligand
    for ccd, groups in site_res.items():
        if ccd not in {c["ccd"] for c in cands} and ccd not in {k[3] for k in ligands}:
            continue  # nothing of this type was transplanted; reported elsewhere
        for label in groups:
            for ch in chains:
                if not groups[label] & chain_res.get(ch, set()):
                    continue                  # site not in this chain/segment
                sk = (ccd, ch, label)
                if sk in filled:
                    continue
                lb = f" site {label}" if label else ""
                if sk in low_coord:          # always worth a look, cofactor or not
                    rep.review(f"{ccd} binding site{lb} in chain {ch} left empty "
                               f"({low_coord[sk]})"
                               + candidates_note(ccd, ch, label, groups[label]))
                    continue
                # site sharing residues with a filled one: one copy serves both
                if covered.get(sk) in dropped:
                    covered.pop(sk)
                by = covered.get(sk) or next(
                    (k["tag"] for k in kept if k["ccd"] == ccd and k["site"][0] == ch
                     and overlapping(ccd, k["site"][1], label)), None)
                if by:
                    lb = f" site {label}" if label else ""
                    rep.info(f"{ccd} binding site{lb} in chain {ch} shares residues with "
                             f"a site filled by {by}; one copy placed")
                    continue
                why = (f"best copy did not fit: {best_reject[sk]}"
                       if sk in best_reject else "no transplanted copy touches it"
                       + ("; some were not transferred, see INFO" if fit_rmsd is not None
                          and getattr(cfg, "_transfer_drops", False) else ""))
                lb = f" site {label}" if label else ""
                msg = (f"{ccd} binding site{lb} in chain {ch} left empty ({why})"
                       + candidates_note(ccd, ch, label, groups[label]))
                # an annotated cofactor missing is a problem; a substrate left out
                # of an apo-like pocket is expected
                (rep.review if ccd in cofactor_ccds else rep.info)(msg)
    rep.data["candidate_sites"] = [
        {"site": c["site"], "ligand": c["ccd"], "chain": c["chain"],
         "label": "" if c["label"].startswith(NOSITE) else c["label"],
         "residues": c["residues"],
         "candidates": [{"rank": r, "copy": x["tag"], "why": x.get("why", "")}
                        for r, x in enumerate(c["cands"], 1)]}
        for c in rep.candidates]
    return kept


def write_candidates(outdir, rep):
    """candidates.pdb: copies considered for each empty site, in the model's
    frame (B-factor = site number, occupancy = rank, 1 = best), described in
    REMARK 250 lines, which also list empty sites without any candidate;
    candidates.tcl shows them in VMD next to prepped.pdb and prints the remarks."""
    if not rep.candidates:
        return
    uni = rep.data.get("uniprot") or {}
    head = f"{rep.name} - UniProt {rep.data.get('accession') or '?'}"
    if uni.get("protein_name"):
        head += f" ({uni['protein_name']})"
    lines, atoms_out, serial = [f"REMARK 250 {head}"], [], 0
    for s in rep.candidates:
        lab = f" site {s['label']}" if s["label"] and not s["label"].startswith(NOSITE) else ""
        where = (f"residues {','.join(map(str, s['residues']))}" if s["residues"]
                 else "no annotated residues")
        lines.append(f"REMARK 250 SITE {s['site']}: {s['ccd']}{lab} in chain {s['chain']} "
                     f"({where}){' - cofactor' if s['cofactor'] else ''}")
        if not s["cands"]:
            lines.append("REMARK 250   no candidate copies (nothing transplanted near the site)")
        for rank, c in enumerate(s["cands"], 1):
            cov = max(c["coverage"].values())
            extra = (f", {len(c['donors'])} donors" if c["metal"] else "") + \
                (f", moved {c['shift']:.1f} A" if c["shift"] >= 0.05 else "")
            lines.append(f"REMARK 250   rank {rank}: {c['tag']} - {c.get('why', '')} "
                         f"(clashes {c['clashes']}, burial {c['burial']:.0%}, "
                         f"coverage {cov:.0%}{extra})")
            for a in heavy(c["atoms"]) or c["atoms"]:
                serial += 1
                b = replace(a, record="HETATM", resname=s["ccd"], chain=s["chain"][:1],
                            resnum=min(9999, s["site"] * 100 + rank), icode="",
                            b=float(s["site"]))
                line = pdb_line(b, serial, f"S{s['site']:03d}"[:4])
                atoms_out.append(line[:54] + f"{rank:6.2f}" + line[60:])
    (Path(outdir) / "candidates.pdb").write_text(
        "\n".join(lines) + "\n" + "".join(atoms_out) + "END\n")
    tcl = ["# load from this folder: vmd -e candidates.tcl",
           "set fh [open candidates.pdb r]",
           "while {[gets $fh line] >= 0} {",
           "    if {[string match \"REMARK 250*\" $line]} {puts [string range $line 11 end]}",
           "}",
           "close $fh",
           "mol new prepped.pdb", "mol modstyle 0 top NewCartoon",
           "mol modselect 0 top protein"]
    for s in rep.candidates:
        if s["residues"]:
            tcl += ["mol addrep top",
                    "mol modstyle [expr [molinfo top get numreps]-1] top Licorice 0.3",
                    "mol modcolor [expr [molinfo top get numreps]-1] top "
                    + ("Name" if s["cands"] else "ColorID 1"),
                    f"mol modselect [expr [molinfo top get numreps]-1] top \"protein and "
                    f"chain {s['chain'][:1]} and resid {' '.join(map(str, s['residues']))}\"",
                    f"# site {s['site']}: {s['ccd']}"
                    + ("" if s["cands"] else " (no candidates; residues in red)")]
            tcl += ["mol addrep top",
                    "mol modstyle [expr [molinfo top get numreps]-1] top Licorice 0.1",
                    "mol modcolor [expr [molinfo top get numreps]-1] top "
                    + ("Name" if s["cands"] else "ColorID 1"),
                    f"mol modselect [expr [molinfo top get numreps]-1] top \"within 5 of protein and "
                    f"chain {s['chain'][:1]} and resid {' '.join(map(str, s['residues']))}\"",
                    f"# site {s['site']}: {s['ccd']}"
                    + ("" if s["cands"] else " (no candidates; residues in red)")]

    if atoms_out:
        tcl += ["mol new candidates.pdb", "mol modstyle 0 top VDW 0.5",
                "mol modmaterial 0 top GlassBubble", 
                "mol modcolor 0 top Beta",
                "# B-factor = site number (REMARK 250 in candidates.pdb), occupancy = rank"]
    (Path(outdir) / "candidates.tcl").write_text("\n".join(tcl) + "\n")


# --------------------------------------------------------------------------
# 3. Trimming
# --------------------------------------------------------------------------

def mature_spans(residues, chains, cfg):
    """Decide which UniProt mature chains of one model chain to build.
    residues: [(key, atoms)] of the model chain; chains: [(start, end)] from UniProt.
    Returns (spans to keep, message, is_problem)."""
    chains = sorted(chains)
    if len(chains) == 1:
        return chains, "", False
    if any(chains[i][1] >= chains[i + 1][0] for i in range(len(chains) - 1)):
        best = max(chains, key=lambda c: c[1] - c[0])
        return [best], (f"UniProt lists overlapping mature chains {chains} (alternative "
                        f"forms); kept the longest, {best[0]}-{best[1]}"), False
    # Links between mature chains: a disulfide, or a large interface - the latter
    # only for chains long enough that a contact means more than a peptide lying
    # on the precursor's surface.
    atoms_of = [[a for key, at in residues if s <= key[1] <= e for a in heavy(at)]
                for s, e in chains]
    present = [i for i, at in enumerate(atoms_of) if at]
    if not present:
        return None, f"none of the UniProt mature chains {chains} is in the model", True
    nres = {i: len({a.resnum for a in atoms_of[i]}) for i in present}
    links = {}
    for i in present:
        for j in present:
            if j <= i:
                continue
            sg_i = [a for a in atoms_of[i] if a.name == "SG"]
            sg_j = [a for a in atoms_of[j] if a.name == "SG"]
            if any(np.linalg.norm(np.subtract(p.xyz, q.xyz)) < cfg.ss_cutoff
                   for p in sg_i for q in sg_j):
                links[(i, j)] = "disulfide"
                continue
            small, big = (i, j) if nres[i] <= nres[j] else (j, i)
            if nres[small] < cfg.min_chain_interface_len:
                continue
            ts, tb = cKDTree(coords(atoms_of[small])), cKDTree(coords(atoms_of[big]))
            touching = {atoms_of[small][k].resnum for k, lst in
                        enumerate(ts.query_ball_tree(tb, 4.5)) if lst}
            frac = len(touching) / nres[small]
            if frac >= cfg.chain_interface_frac:
                links[(i, j)] = f"interface {frac:.0%}"

    def component(start):
        comp, todo = {start}, [start]
        while todo:
            i = todo.pop()
            for (a, b) in links:
                for x, y in ((a, b), (b, a)):
                    if x == i and y not in comp:
                        comp.add(y)
                        todo.append(y)
        return comp

    main = max(present, key=lambda i: nres[i])
    comp = component(main)
    rng = lambda i: f"{chains[i][0]}-{chains[i][1]}"
    how = "; ".join(f"{rng(a)}/{rng(b)}: {w}" for (a, b), w in links.items()
                    if a in comp and b in comp)
    kept = [chains[i] for i in sorted(comp)]
    dropped = [chains[i] for i in present if i not in comp]
    n_kept = sum(nres[i] for i in comp)
    if not dropped:
        if len(kept) == 1:
            return kept, "", False
        return kept, (f"mature protein of {len(kept)} chains held together ({how}); "
                      "built as separate segments, excised linkers removed"), False
    if n_kept >= cfg.main_chain_frac * sum(nres.values()):
        msg = f"kept mature chain(s) {kept}" + (f" ({how})" if how else "")
        return kept, (msg + f"; unlinked short chains {dropped} removed"
                      + ("; built as separate segments" if len(kept) > 1 else "")), False
    return None, (f"UniProt lists {len(chains)} mature chains {chains} that are not all "
                  "linked by disulfides or large interfaces in the model (polyprotein?); "
                  "mature boundaries not applied - choose the product to simulate"), True


_SEG_IDS = "ABCDEFGHIJKMNOPQRSTUVYZ0123456789abcdefghijklmnopqrstuvwxyz"  # no L/W/X


def trim_protein(protein, plddt, uni, cfg, rep):
    """Trim each model chain to its mature span(s) and confident termini.
    Mature chains that are built separately get their own chain IDs (one psfgen
    segment each).  Returns trimmed atoms and {chain: (first, last, n0, n1)}."""
    out, termini = [], {}
    by_chain = OrderedDict()
    for key, atoms in group_residues(protein).items():
        by_chain.setdefault(key[0], []).append((key, atoms))
    free_ids = [c for c in _SEG_IDS if c not in by_chain]
    uni_chains = uni["chains"] if uni else []
    spans_info = {}
    for ch, residues in by_chain.items():
        nums = [k[1] for k, _ in residues]
        spans, msg, problem = (mature_spans(residues, uni_chains, cfg) if uni_chains
                               else (None, "", False))
        if msg:
            (rep.review if problem else rep.info)(f"chain {ch}: {msg}")
        if not spans:
            spans = [(nums[0], nums[-1])]
            mature = False
        else:
            mature = True
        if mature and spans[0][0] > nums[0]:
            rep.info(f"chain {ch}: removed {spans[0][0] - nums[0]} residues before the "
                     "mature chain (signal/transit/propeptide/iMet)")
        for n_span, (lo, hi) in enumerate(spans):
            seg_ch = ch if n_span == 0 else free_ids.pop(0)
            idx = [i for i, n in enumerate(nums) if lo <= n <= hi]
            if not idx:
                rep.review(f"chain {ch}: no residues inside mature span {lo}-{hi}")
                continue
            conf = [plddt.get((ch, nums[i]), 100.0) >= cfg.plddt for i in idx]
            w = cfg.trim_window
            first = next((j for j in range(len(idx)) if all(conf[j:j + w])), 0)
            last = next((j for j in range(len(idx) - 1, -1, -1)
                         if all(conf[max(0, j - w + 1):j + 1])), len(idx) - 1)
            keep = idx[first:last + 1]
            n0, n1 = nums[keep[0]], nums[keep[-1]]
            n_native, c_native = n0 == lo, n1 == hi
            tag = f"chain {seg_ch}" + (f" (residues {lo}-{hi} of chain {ch})"
                                       if len(spans) > 1 else "")
            if not n_native:
                rep.info(f"{tag}: N-terminus trimmed to {n0} by pLDDT -> ACE cap")
            if not c_native:
                rep.info(f"{tag}: C-terminus trimmed to {n1} by pLDDT -> CT3 cap")
            low = [nums[i] for i in keep if plddt.get((ch, nums[i]), 100) < cfg.plddt]
            if low:
                rep.info(f"{tag}: {len(low)} internal residues below pLDDT "
                         f"{cfg.plddt} kept")
            first_res = residues[keep[0]][0][3]
            if n_native:
                fp = {"PRO": "PROP", "GLY": "GLYP"}.get(first_res, "NTER")
            else:
                fp = "ACP" if first_res == "PRO" else "ACE"
            termini[seg_ch] = (fp, "CTER" if c_native else "CT3", n0, n1)
            spans_info[seg_ch] = (ch, n0, n1)
            for i in keep:
                out.extend(replace(a, chain=seg_ch) for a in residues[i][1])
    rep.data["segments"] = {k: f"chain {v[0]} residues {v[1]}-{v[2]}"
                            for k, v in spans_info.items()}
    return out, termini


# --------------------------------------------------------------------------
# 4. Dowser++
# --------------------------------------------------------------------------

# Files Dowser++ creates while running; never copied into a per-run folder, so a
# stale copy in the installation cannot leak into (or, with symlinks, be
# overwritten by) a run.
DOWSER_TEMP = re.compile(r"^(box.*\.txt|config\.txt|AllSets\.pdb|AllScores\.txt|"
                         r"temp_PredictedWaters.*\.pdb|PredictedWaters\.pdb|"
                         r"PredictedInternal\.pdb|data\.pdb|reform2?\.pdb|placed\.pdb|"
                         r"refined\.pdb|sorted\.pdb|surface\.wat|new\.pdbqt|Set\d+\.pdbqt|"
                         r"dpp\.log)$")


def stage_dowser(src, dst, link):
    for root, _, files in os.walk(src):
        d = Path(dst) / os.path.relpath(root, src)
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            if Path(root) == Path(src) and DOWSER_TEMP.match(f):
                continue
            s = Path(root) / f
            if link:
                (d / f).symlink_to(s.resolve())
            else:
                shutil.copy2(s, d / f)


def _dowser_waters(path):
    out = []
    for a in read_pdb(path):
        if is_protein(a):
            continue
        if a.element == "O" and (is_water(a) or a.name in ("O", "OW", "OH2")):
            out.append(a)
    # Dowser++ numbers its waters 1..N in chain A: give them their own chain so
    # they can never be confused with protein residues of the same number
    return [replace(a, chain="W", resnum=i, icode="", resname="HOH")
            for i, a in enumerate(out, 1)]


def run_dowser(protein, lig_atoms, outdir, cfg, rep):
    import tempfile
    import time
    logdir = Path(outdir).resolve() / "dowser"
    logdir.mkdir(exist_ok=True)
    env = dict(os.environ)
    if cfg.dowser_path:
        env["PATH"] = cfg.dowser_path + os.pathsep + env.get("PATH", "")

    def attempt(atoms, label):
        work = Path(tempfile.mkdtemp(prefix="dowserpp_", dir=cfg.dowser_tmp))
        try:
            stage_dowser(cfg.dowser_dir, work, cfg.dowser_link)
            inp = work / "dowser_in.pdb"
            write_pdb(heavy(atoms), inp)
            before = {p.name for p in work.iterdir()}
            t0 = time.time()
            try:
                r = subprocess.run(["bash", cfg.dowser_script, str(inp)], cwd=work,
                                   env=env, capture_output=True, text=True,
                                   stdin=subprocess.DEVNULL,
                                   timeout=cfg.dowser_timeout or None)
                log_txt, rc = r.stdout + r.stderr, r.returncode
            except subprocess.TimeoutExpired:
                log_txt, rc = "TIMEOUT", -1
            (logdir / f"dowser_{label}.log").write_text(log_txt)
            mins = (time.time() - t0) / 60
            if "No buried cavities" in log_txt:
                return [], mins, "none"
            if cfg.dowser_output:
                outs = [work / cfg.dowser_output] if (work / cfg.dowser_output).exists() else []
            else:   # new .pdb files that contain waters
                outs = [p for p in work.iterdir() if p.suffix == ".pdb"
                        and p.name not in before and _dowser_waters(p)]
            if rc != 0 or not outs:
                return None, mins, f"exit code {rc}, no output file"
            out = max(outs, key=lambda p: p.stat().st_mtime)
            shutil.copy(out, logdir / out.name)
            if len(outs) > 1:
                rep.info(f"Dowser++ wrote several water files; used {out.name} "
                         "(set --dowser-output to choose)")
            return _dowser_waters(out), mins, out.name
        finally:
            if cfg.keep_dowser_tmp:
                rep.info(f"Dowser++ work folder kept: {work}")
            else:
                shutil.rmtree(work, ignore_errors=True)

    use_ligs = lig_atoms and not cfg.dowser_protein_only
    waters, mins, what = attempt(protein + (lig_atoms if use_ligs else []),
                                 "with_ligands" if use_ligs else "protein")
    if waters is None and use_ligs:
        rep.review(f"Dowser++ failed with cofactors present ({what}); retried "
                   "protein-only")
        waters, mins2, what = attempt(protein, "protein_only")
        mins += mins2
    if waters is None:
        rep.review(f"Dowser++ failed ({what}); no internal waters - see dowser/*.log")
        return []
    if what == "none":
        rep.info(f"Dowser++: no buried cavities, no internal waters ({mins:.1f} min)")
        return []

    # drop waters overlapping protein/cofactors or each other
    tree = cKDTree(coords(heavy(protein + lig_atoms)))
    kept = []
    for w in waters:
        if tree.query(w.xyz)[0] < cfg.water_overlap:
            continue
        if any(np.linalg.norm(np.subtract(w.xyz, k.xyz)) < 2.4 for k in kept):
            continue
        kept.append(w)
    rep.info(f"Dowser++: {len(kept)} internal waters kept ({len(waters) - len(kept)} "
             f"removed for overlap) from {what}, {mins:.1f} min")
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


def run_pdb2pqr(protein, waters, ligands, workdir, cfg, rep):
    """PDB2PQR + PROPKA on protein, internal waters and kept ligands.  PDB2PQR
    passes unknown HETATM residues on to PROPKA, which includes organic ligands
    and treats metal ions as point charges; the ligands themselves are left out
    of PDB2PQR's output, which is fine since only the protein is taken from it."""
    wd = (Path(workdir) / "pdb2pqr").resolve()
    wd.mkdir(exist_ok=True)
    inp = wd / "p2p_in.pdb"
    wat = [replace(w, record="HETATM", resname="HOH", name="O", chain="W", resnum=i)
           for i, w in enumerate(waters, 1)]
    # Own chain + numbering, so ligands can never be taken for protein residues.
    # Organic ligands get neutral residue names (LG1, LG2, ...): PDB2PQR would
    # otherwise match some by name to its own templates, assign partial charges
    # and abort on a non-integral total charge.  Metal ions keep their names,
    # which is how PROPKA recognises them and their charge.
    lig = [replace(a, record="HETATM", chain="L", resnum=k, icode="",
                   resname=l["ccd"] if l["metal"] else f"LG{k}")
           for k, l in enumerate(ligands, 1) for a in heavy(l["atoms"])]
    out_pdb, out_pqr = wd / "p2p_out.pdb", wd / "p2p_out.pqr"
    cmd = [cfg.pdb2pqr, "--ff=CHARMM", "--ffout=CHARMM",
           "--titration-state-method=propka", f"--with-ph={cfg.ph}",
           "--keep-chain", f"--pdb-output={out_pdb}", str(inp), str(out_pqr)]
    for attempt in ((lig, "with ligands"), ([], "without ligands")) if lig else (([], ""),):
        write_pdb(heavy(protein) + attempt[0] + wat, inp)
        out_pdb.unlink(missing_ok=True)
        r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True)
        (wd / "pdb2pqr_stdout.log").write_text(r.stdout + r.stderr)
        if r.returncode == 0 and out_pdb.exists():
            break
        if attempt[0]:
            (wd / "pdb2pqr_with_ligands_failed.log").write_text(r.stdout + r.stderr)
            rep.review("PDB2PQR failed with the ligands present; protonation computed "
                       "without them (see pdb2pqr/pdb2pqr_with_ligands_failed.log)")
    if r.returncode != 0 or not out_pdb.exists():
        raise RuntimeError(f"pdb2pqr failed (see {wd}/pdb2pqr_stdout.log)")
    logtxt = ""
    for f in (wd / "p2p_out.log", wd / "pdb2pqr_stdout.log"):
        if f.exists():
            logtxt += f.read_text()
    atoms = read_pdb(out_pdb)
    # PDB2PQR splits chains at backbone breaks and may relabel the pieces; it
    # keeps the residue order, so map its protein residues back to ours by order
    in_keys = list(dict.fromkeys((a.chain, a.resnum) for a in heavy(protein)))
    out_keys = list(dict.fromkeys((a.chain, a.resnum) for a in atoms
                                  if not is_water(a) and a.chain not in ("L", "W")))
    if out_keys != in_keys and len(out_keys) == len(in_keys) and \
            all(o[1] == i[1] for o, i in zip(out_keys, in_keys)):
        remap = dict(zip(out_keys, in_keys))
        atoms = [replace(a, chain=remap[(a.chain, a.resnum)][0])
                 if (a.chain, a.resnum) in remap else a for a in atoms]
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
    pts = [a for a in prot_heavy if is_donor(a) or
           (orig_resname.get((a.chain, a.resnum)), a.name) in COORD_ATOMS]
    if not pts:
        return []
    tree = cKDTree(coords(pts))
    sites = []
    for lig in ligands:
        for m in lig["atoms"]:
            if m.element not in METAL_ELEMENTS:
                continue
            reach = max(cfg.metal_cutoff, metal_reach(m.element, cfg))
            for i in tree.query_ball_point(m.xyz, reach):
                d = float(np.linalg.norm(np.subtract(m.xyz, pts[i].xyz)))
                if d <= max(cfg.metal_cutoff, metal_ideal(m.element, pts[i].element)
                            + cfg.donor_tol):
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

MASS_TO_ELEMENT = [(1.008, "H"), (10.81, "B"), (12.011, "C"), (14.007, "N"),
                   (15.999, "O"), (18.998, "F"), (22.99, "NA"), (24.305, "MG"),
                   (30.974, "P"), (32.06, "S"), (35.45, "CL"), (39.098, "K"),
                   (40.078, "CA"), (54.938, "MN"), (55.845, "FE"), (58.933, "CO"),
                   (58.693, "NI"), (63.546, "CU"), (65.38, "ZN"), (78.97, "SE"),
                   (79.904, "BR"), (112.41, "CD"), (126.9, "I")]
COVALENT_R = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07,
              "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39, "SE": 1.20, "B": 0.84,
              "FE": 1.32, "MG": 1.41, "ZN": 1.22, "CU": 1.32, "MN": 1.39, "CO": 1.26,
              "NI": 1.24}
_RTF_CACHE = {}


def load_rtf(paths):
    """{RESI name: (atom names, elements, heavy-atom bonds)} from topology/stream files."""
    key = tuple(str(p) for p in paths)
    if key in _RTF_CACHE:
        return _RTF_CACHE[key]
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"topology files not found: {missing[:3]}")
    texts = [Path(p).read_text(errors="replace") for p in paths]
    type_el = {}
    for txt in texts:
        for line in txt.splitlines():
            t = line.split("!")[0].split()
            if len(t) >= 4 and t[0].upper() == "MASS":
                if len(t) >= 5 and t[4].isalpha():
                    type_el[t[2].upper()] = t[4].upper()
                else:
                    m = float(t[3])
                    type_el[t[2].upper()] = min(MASS_TO_ELEMENT, key=lambda e: abs(e[0] - m))[1]
    res, cur = {}, None
    for txt in texts:
        for line in txt.splitlines():
            t = line.split("!")[0].split()
            if not t:
                continue
            kw = t[0].upper()[:4]
            if kw in ("RESI", "PRES"):
                cur = None
                if kw == "RESI" and len(t) > 1:
                    name = t[1].upper()
                    if name in res:      # psfgen keeps the FIRST definition; read and discard
                        cur = {"atoms": [], "el": {}, "bonds": set()}
                    else:
                        cur = res[name] = {"atoms": [], "el": {}, "bonds": set()}
            elif kw == "END":
                cur = None
            elif cur is not None and kw == "ATOM" and len(t) >= 3:
                name = t[1].upper()
                cur["atoms"].append(name)
                cur["el"][name] = type_el.get(t[2].upper(), re.sub(r"[^A-Z]", "", name)[:1])
            elif cur is not None and kw in ("BOND", "DOUB", "TRIP", "AROM"):
                names = [x.upper() for x in t[1:]]
                for a, b in zip(names[::2], names[1::2]):
                    if a[0] not in "+-" and b[0] not in "+-":
                        cur["bonds"].add((a, b))
    _RTF_CACHE[key] = res
    return res


TOPPAR_FILES = ("load_psfgen.tcl", "topologies_psfgen.txt", "ccd_aliases.json",
                "uniprot_states.json")


def _toppar_json(toppar, fname):
    key = ("json", str(Path(toppar).resolve()), fname)
    if key not in _RTF_CACHE:
        d = json.loads((Path(toppar).resolve() / fname).read_text())
        _RTF_CACHE[key] = {k: ([v] if isinstance(v, str) else list(v)) for k, v in d.items()}
    return _RTF_CACHE[key]


def charmm_aliases(toppar):
    """PDB code -> CHARMM residue names (toppar_prepare.py's CCD_ALIASES)."""
    return {k.upper(): v for k, v in _toppar_json(toppar, "ccd_aliases.json").items()}


def uniprot_states(toppar):
    """normalised UniProt ligand name -> CHARMM residue names (UNIPROT_STATES)."""
    return {norm_name(k): v for k, v in _toppar_json(toppar, "uniprot_states.json").items()}


def charmm_ligand(atoms, ccd, toppar, uniprot_names=()):
    """Find a CHARMM RESI matching this ligand's heavy-atom graph.
    Residues for the ligand's UniProt state (e.g. Fe(2+) -> FE2P, NADH -> NAI) are
    tried first, then the PDB code, then its aliases.
    Returns (resname, {pdb atom name: charmm atom name}, message, note)."""
    import networkx as nx
    from networkx.algorithms import isomorphism
    top = Path(toppar).resolve()
    names = [l.strip() for l in (top / "topologies_psfgen.txt").read_text().splitlines()
             if l.strip()]
    rtf = load_rtf([top / n for n in names])
    states = uniprot_states(toppar)
    prefer = list(dict.fromkeys(r for n in uniprot_names for r in states.get(norm_name(n), [])))
    la = heavy(atoms) or atoms
    g = nx.Graph()
    for i, a in enumerate(la):
        g.add_node(i, el=a.element.upper())
    xyz = coords(la)
    for i in range(len(la)):
        for j in range(i + 1, len(la)):
            ri = COVALENT_R.get(la[i].element.upper(), 0.8)
            rj = COVALENT_R.get(la[j].element.upper(), 0.8)
            if np.linalg.norm(xyz[i] - xyz[j]) < ri + rj + 0.45:
                g.add_edge(i, j)
    default = list(dict.fromkeys([ccd] + charmm_aliases(toppar).get(ccd, [])))
    tried = []
    for name in dict.fromkeys(prefer + default):
        r = rtf.get(name)
        if r is None:
            continue
        hv = [a for a in r["atoms"] if r["el"].get(a) != "H"]
        h = nx.Graph()
        for a in hv:
            h.add_node(a, el=r["el"].get(a, "X").upper())
        for a, b in r["bonds"]:
            if a in h and b in h:
                h.add_edge(a, b)
        tried.append(f"{name} ({len(hv)} heavy atoms vs {len(la)})")
        if len(hv) != len(la):
            continue
        gm = isomorphism.GraphMatcher(g, h, node_match=lambda x, y: x["el"] == y["el"])
        if gm.is_isomorphic():
            note = ""
            if len(prefer) > 1:
                note = (f"UniProt names {list(uniprot_names)} suggest several states "
                        f"{prefer}; used the first that matches")
            elif prefer and name in prefer:
                note = f"state from UniProt '{', '.join(uniprot_names)}'"
            elif prefer:
                note = (f"UniProt '{', '.join(uniprot_names)}' suggests {prefer[0]}, which "
                        "is not in the toppar or does not match; used the PDB code's residue")
            return name, {la[i].name: gm.mapping[i] for i in gm.mapping}, "", note
    if not tried:
        return None, {}, "no matching RESI in the toppar folder", ""
    return None, {}, "heavy-atom graph does not match " + ", ".join(tried), ""


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
    tcl = [f'# psfgen build for "{name}" - generated by af_prep.py',
           "package require psfgen", "resetpsf"]
    toptcl = Path(cfg.toppar).resolve() / "load_psfgen.tcl"
    tcl += [f"source {{{toptcl}}}", f'load_charmm_topologies "{toptcl.parent}"']
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
                f'    pdb "{f.name}"', "}", f'coordpdb "{f.name}" {seg(ch)}']

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
        rname, amap, why, note = \
            (None, {}, "needs a bonded model / dedicated parameters", "") \
            if ccd in NEEDS_BONDED else \
            charmm_ligand(lig["atoms"], ccd, cfg.toppar, lig.get("uniprot", ()))
        if rname:
            la = [replace(a, record="HETATM", resname=rname, chain="X", resnum=1,
                          icode="", name=amap.get(a.name, a.name))
                  for a in heavy(lig["atoms"])]
            write_pdb(la, f, sid)
            auto = "    auto none\n" if len(la) == 1 else ""
            tcl += [f"segment {sid} {{\n    first none\n    last none\n{auto}"
                    f'    pdb "{f.name}"\n}}', f'coordpdb "{f.name}" {sid}']
            lig_seg[id(lig)] = (sid, la, amap)
            msg = f"{ccd} built as CHARMM residue {rname} (segment {sid})"
            if note and ("suggest" in note):
                rep.review(f"{msg}; {note}")
            else:
                rep.info(msg + (f"; {note}" if note else ""))
        else:
            write_pdb(heavy(lig["atoms"]), f, sid)
            if ccd in NEEDS_BONDED:
                hint = ("needs a bonded model to its protein ligands, e.g. published "
                        "cluster parameters added with toppar_prepare.py --extra-stream")
            else:
                hint = ("add its topology to the toppar (e.g. toppar_prepare.py "
                        "--extra-stream with CGenFF output) or an alias to CCD_ALIASES "
                        "in toppar_prepare.py")
            msg = f"{ccd} NOT built ({why}) - written to build/{f.name}; {hint}"
            if ccd in NEEDS_BONDED or why.startswith("no matching RESI"):
                rep.needs_params(ccd, msg)       # parameterisation work
            else:
                rep.review(msg)                  # e.g. incomplete or distorted copy
            tcl += [f'# NEEDS_PARAMS: {ccd} in "{f.name}" (segment {sid}) not built']

    for his, lig in phem:
        if id(lig) in lig_seg:
            tcl.append(f"patch PHEM {seg(his[0])}:{his[1]} {lig_seg[id(lig)][0]}:1"
                       "  ;# VERIFY patch name/order in your heme toppar")

    if waters:
        f = bd / "water_internal.pdb"
        wa = [replace(w, record="HETATM", resname="TIP3", name="OH2", chain="W",
                      resnum=i, icode="", element="O")
              for i, w in enumerate(waters, 1)]
        write_pdb(wa, f, "DWAT")
        tcl += ["", "# Dowser++ internal waters",
                "segment DWAT {\n    first none\n    last none\n    auto none\n"
                f'    pdb "{f.name}"\n}}', f'coordpdb "{f.name}" DWAT']

    tcl += ["", "regenerate angles dihedrals", "guesscoord",
            f'writepsf "{name}.psf"', f'writepdb "{name}.pdb"']
    final = name
    if cfg.solvate:
        tcl += ["", "package require solvate",
                f'solvate "{name}.psf" "{name}.pdb" -t {cfg.padding} -o "{name}_solv"',
                "package require autoionize",
                f'autoionize -psf "{name}_solv.psf" -pdb "{name}_solv.pdb" '
                f'-sc {cfg.salt} -o "{name}_ion"']
        final = f"{name}_ion"
        tcl += ["", "# box dimensions for NAMD",
                f'set bm [mol new "{final}.psf"]', f'mol addfile "{final}.pdb" waitfor all',
                "set mm [measure minmax [atomselect $bm all]]",
                f'set fh [open "{final}_box.txt" w]',
                "puts $fh [vecsub [lindex $mm 1] [lindex $mm 0]]", "close $fh",
                "mol delete $bm"]

    eb = []
    for lig, m, p, d in extrabonds:
        if id(lig) not in lig_seg:
            continue
        sid, la, amap = lig_seg[id(lig)]
        mname = amap.get(m.name, m.name)
        eb.append((sid, 1, mname, seg(p.chain), p.resnum, p.name, f"{d:.3f}"))
    if eb:
        tcl += ["", "# NAMD extraBonds restraints for metal coordination",
                f'set m [mol new "{final}.psf"]', f'mol addfile "{final}.pdb" waitfor all',
                "set fh [open metal_extrabonds.txt w]",
                f"foreach {{s1 r1 n1 s2 r2 n2 r0}} {{{' '.join(' '.join(map(str, e)) for e in eb)}}} {{",
                "    set a [atomselect $m \"segname $s1 and resid $r1 and name $n1\"]",
                "    set b [atomselect $m \"segname $s2 and resid $r2 and name $n2\"]",
                f"    puts $fh \"bond [$a get index] [$b get index] {cfg.metal_k} $r0\"",
                "    $a delete; $b delete", "}", "close $fh"]
    tcl += ["", 'puts "AFPREP_BUILD_OK"', "exit"]
    (bd / "build.tcl").write_text("\n".join(tcl) + "\n")
    return bd


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def model_accession(model, cfg):
    """(accession, AlphaFold id, review notes) from header, file name or --accession."""
    notes = []
    acc, af_id = accession_from_name(model)
    head = accession_from_header(model)
    if len(head) > 1:
        notes.append(f"several accessions in header {head}; using {head[0]} "
                     "(multimer or chimera?)")
    if head:
        if acc and acc != head[0]:
            notes.append(f"file name says {acc} but header says {head[0]}; using header")
        if acc != head[0]:
            acc, af_id = head[0], f"AF-{head[0]}-F1"
    if cfg.accession:
        acc = cfg.accession
        af_id = af_id or f"AF-{acc}-F1"
    return acc, af_id, notes


def prefetch(cfg):
    """Download UniProt entries (and AlphaFill files with --alphafill auto) for
    all models before processing, with at most cfg.max_downloads connections at
    a time, so parallel workers read from the cache instead of all hitting the
    servers at once.  Failures are only counted here; the model's own lookup
    retries and reports them."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = {}
    for m in cfg.models:
        try:
            acc, af_id, _ = model_accession(Path(m), cfg)
        except Exception:  # noqa: BLE001 - reported again in process()
            continue
        if not acc:
            continue
        if not cfg.no_uniprot and not (cfg.cache / f"{acc}.json").exists():
            jobs[("uniprot", acc)] = lambda a=acc: fetch_uniprot(a, cfg.cache)
        if cfg.alphafill == "auto" and not (cfg.cache / f"{acc}.alphafill.cif").exists() \
                and not (cfg.cache / f"{acc}.alphafill.none").exists():
            jobs[("alphafill", acc)] = lambda a=acc, i=af_id: download_alphafill(
                a, i, cfg.alphafill_url, cfg.cache)
    if not jobs:
        return
    from concurrent.futures import as_completed
    if tqdm is None:
        log.warning("downloading %d UniProt/AlphaFill entries (%d at a time)",
                    len(jobs), cfg.max_downloads)
    failed = []
    with ThreadPoolExecutor(max_workers=cfg.max_downloads) as ex:
        futures = {ex.submit(f): k for k, f in jobs.items()}
        done = as_completed(futures)
        if tqdm is not None:
            done = tqdm(done, total=len(futures), desc="downloads", unit="entry")
        for fut in done:
            k = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{k[0]} {k[1]} ({exc})")
    if failed:
        log.warning("%d downloads failed after retries, e.g. %s", len(failed), failed[0])


def run_ligand_names(cfg):
    """{(UniProt ligand name, ChEBI ID)} over all models of this run whose
    UniProt entry is in the cache."""
    pairs = set()
    for m in cfg.models:
        try:
            acc, _, _ = model_accession(Path(m), cfg)
        except Exception:  # noqa: BLE001 - reported again in process()
            continue
        f = Path(cfg.cache) / f"{acc}.json" if acc else None
        if f and f.exists():
            try:
                pairs |= set(parse_uniprot(json.loads(f.read_text()))["chebi"].items())
            except (ValueError, KeyError, TypeError):
                continue
    return pairs


def resolve_chebi_ids(cfg):
    """Resolve every ChEBI ID of this run once, in the main process, into
    <cache>/chebi_ccd.json; the model workers only read that file.  Results
    already in the file are reused; failed lookups are retried next run."""
    if cfg.no_uniprot:
        return
    ids = sorted({c for _, c in run_ligand_names(cfg)})
    if not ids:
        return
    path = Path(cfg.cache) / CHEBI_CACHE_NAME
    todo = len(set(ids) - set(ligand_resolver.load_cache(path)))
    if todo and tqdm is None:
        log.warning("resolving %d ChEBI IDs to PDB codes", todo)
    progress = ((lambda it, total: tqdm(it, total=total, desc="ChEBI", unit="id"))
                if tqdm else None)
    results = ligand_resolver.resolve_many(ids, path, workers=min(cfg.max_downloads, 4),
                                           progress=progress)
    failed = sorted(c for c, r in results.items() if not r["complete"])
    if failed:
        log.warning("%d ChEBI ID(s) could not be fully resolved and will be retried "
                    "next run, e.g. %s: %s", len(failed), failed[0],
                    "; ".join(results[failed[0]]["notes"][:2]))
    _RESOLUTIONS.pop(str(cfg.cache), None)        # reread in this process


def process(model, cfg):
    model = Path(model)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", model.stem)
    outdir = Path(cfg.outdir) / name
    outdir.mkdir(parents=True, exist_ok=True)
    rep = Report(name)
    cfg._transfer_drops = False
    for nm, ccds in cfg.name_map.items():         # user-supplied UniProt name -> CCD
        NAME_TO_CCD_NORM[norm_name(nm)] = list(ccds)
    acc, af_id, notes = model_accession(model, cfg)
    for n in notes:
        rep.review(n)
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
        rep.review("no UniProt accession in header or file name (use --accession)")
    if uni:
        rep.data["uniprot"] = uni
        for m in uni["modified"]:
            if any(h in m["desc"].lower() for h in COVALENT_HINTS):
                rep.review(f"covalent cofactor/modification at {m['pos']}: "
                           f"{m['desc']} - needs a custom patch")
            elif m["type"] == "Glycosylation":
                rep.info(f"glycosylation at {m['pos']}: {m['desc']} (not modelled)")
    allowed, expected, site_res, unmapped, site_names, related = (
        allowed_ligands(uni, cfg.keep_ligands, cfg.ignore_names,
                        chebi_resolutions(cfg), rep) if uni
        else (set(cfg.keep_ligands), set(), {}, {}, {}, []))

    # --- 2. AlphaFill
    af_file = get_alphafill(acc, af_id, cfg.alphafill, cfg.alphafill_url, cfg.cache, rep)
    src = read_structure(af_file) if af_file else inp
    af_protein = [a for a in src if is_protein(a) and a.record == "ATOM"]
    ligs = OrderedDict((k, v) for k, v in group_residues(
        [a for a in src if not is_protein(a) and not is_water(a)]).items())
    fit_rmsd, rejected = {}, []
    protein = [a for a in inp if is_protein(a) and a.record == "ATOM"]
    if af_file:
        ca_in = {(a.chain, a.resnum): a.xyz for a in inp if a.name == "CA"}
        ca_af = {(a.chain, a.resnum): a.xyz for a in af_protein if a.name == "CA"}
        common = sorted(set(ca_in) & set(ca_af))
        dev = (np.sqrt(np.mean(np.sum((np.array([ca_in[k] for k in common]) -
                                       np.array([ca_af[k] for k in common])) ** 2, 1)))
               if common else float("nan"))
        same_len = len(common) == len(ca_in) == len(ca_af)
        rep.data["alphafill_vs_model_ca_rmsd"] = round(float(dev), 2)
        if cfg.alphafill_coords == "alphafill":
            protein = af_protein
            if not same_len or dev > 0.5:
                rep.review(f"AlphaFill protein differs from input model (CA RMSD "
                           f"{dev:.2f} A, same length: {same_len}); AlphaFill coordinates used")
        else:
            if not same_len or dev > 0.5:
                rep.info(f"AlphaFill protein differs from input model (CA RMSD {dev:.2f} A, "
                         f"same length: {same_len}); ligands moved onto the input model")
            ligs, fit_rmsd, rejected = transplant_ligands(ligs, af_protein, protein, cfg, rep)

    # --- 3. trim
    protein, termini = trim_protein(protein, plddt, uni, cfg, rep)
    orig_resname = {(a.chain, a.resnum): ("HIS" if a.resname in ("HSD", "HSE", "HSP")
                                          else a.resname) for a in protein}

    cofactor_ccds = {c for ccds in expected for c in ccds}
    # cofactors UniProt names without binding-site residues: pseudo-sites, one per
    # copy UniProt's note gives ("Binds 2 Mg(2+) ions per subunit"), else one
    if uni:
        counts = {}
        for co_name in uni["cofactors"]:
            m = rep.data.get("ligand_mapping", {}).get(co_name, {})
            for c in m.get("codes", []):
                counts[c] = max(counts.get(c, 1), uni["cofactor_counts"].get(co_name, 1))
        for c in sorted(cofactor_ccds - set(site_res)):
            site_res[c] = {f"{NOSITE}{k}": set() for k in range(1, counts.get(c, 1) + 1)}
    ligands = (select_ligands(protein, ligs, allowed, site_res, unmapped, related,
                              fit_rmsd, cofactor_ccds, cfg, rep, rejected)
               if ligs or rejected or site_res else [])
    for lig in ligands:                           # UniProt names of the site each one fills
        names = site_names.get(lig["ccd"], {})
        lig["uniprot"] = sorted(names.get(lig["site_label"], set())
                                or names.get("*cofactor*", set()))
    found = {l["ccd"] for l in ligands}
    for ccds in expected:
        if not found & set(ccds):
            msg = (f"annotated cofactor {'/'.join(ccds)} not placed - "
                   "consider co-folding (Boltz/Chai-1) or a homolog structure")
            # a site-level line already says what happened: keep this one as INFO
            explained = any(lvl == "REVIEW" and any(m.startswith(f"{c} ") for c in ccds)
                            for lvl, m in rep.flags)
            (rep.info if explained else rep.review)(msg)
    lig_atoms = [a for l in ligands for a in l["atoms"]]
    write_pdb(heavy(protein) + heavy(lig_atoms), outdir / "prepped.pdb")
    write_candidates(outdir, rep)

    # --- 4. Dowser++ internal waters
    waters = run_dowser(protein, lig_atoms, outdir, cfg, rep) if cfg.dowser_dir else []

    for lig in ligands:
        if lig.get("water_completed") and cfg.dowser_dir:
            el = lig["el"][0]
            n = sum(1 for w in waters if np.linalg.norm(np.subtract(w.xyz, lig["xyz"][0]))
                    <= metal_ideal(el, "O") + cfg.donor_tol)
            rep.info(f"{lig['tag']}: {n} Dowser++ water(s) at bonding distance"
                     + ("" if n else "; bulk water from solvation may still reach it"))

    # --- 5. PDB2PQR
    p2p_atoms, pkas, p2p_disu = run_pdb2pqr(protein, waters, ligands, outdir, cfg, rep)
    p2p_prot = [a for a in p2p_atoms if (a.chain, a.resnum) in orig_resname
                and a.chain not in ("W", "L") and not is_water(a)]
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

    if ligands and not any("PDB2PQR failed with the ligands" in m for _, m in rep.flags):
        rep.info(f"PROPKA included {len(ligands)} ligand(s): organic ligands with "
                 "PROPKA's ligand model, metal ions as point charges")

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
        r = subprocess.run([cfg.vmd, "-dispdev", "text"],
                           cwd=bd, capture_output=True, text=True,
                           input=(bd/"build.tcl").read_text(), timeout=cfg.vmd_timeout)
        (bd / "vmd.log").write_text(r.stdout + r.stderr)
        if not (bd / f"{name}.psf").exists():
            rep.review("VMD/psfgen build did not produce a PSF; see build/vmd.log")
        elif "MOLECULE DESTROYED BY FATAL ERROR" in r.stdout or \
                re.search(r"psfgen\) ERROR", r.stdout):
            rep.review("psfgen reported errors; see build/vmd.log")
    rep.write(outdir)
    return rep


def write_name_suggestions(cfg):
    """Collect, over all models of this run, the unmapped-name suggestions into
    <outdir>/ligand_name_suggestions.tsv and the ligands that could not be built
    for lack of parameters into <outdir>/needs_params.tsv, most frequent first."""
    from collections import defaultdict
    agg = defaultdict(lambda: [0, []])
    for m in cfg.models:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(m).stem)
        f = Path(cfg.outdir) / name / "report.json"
        if not f.exists():
            continue
        for uni_name, ccd, n in json.loads(f.read_text()).get("name_suggestions", []):
            agg[(uni_name, ccd)][0] += n
            agg[(uni_name, ccd)][1].append(name)
    params = defaultdict(list)
    for m in cfg.models:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(m).stem)
        f = Path(cfg.outdir) / name / "report.json"
        if f.exists():
            for code in json.loads(f.read_text()).get("needs_params", []):
                params[code].append(name)
    with open(Path(cfg.outdir) / "needs_params.tsv", "w") as fh:
        fh.write("ccd\tcopies\tn_models\tmodels\n")
        for code, models in sorted(params.items(), key=lambda x: -len(x[1])):
            fh.write(f"{code}\t{len(models)}\t{len(set(models))}\t"
                     f"{','.join(sorted(set(models)))}\n")
    out = Path(cfg.outdir) / "ligand_name_suggestions.tsv"
    with open(out, "w") as fh:
        fh.write("uniprot_name\tccd\tcopies\tn_models\tmodels\n")
        for (uni_name, ccd), (n, models) in sorted(agg.items(), key=lambda x: -x[1][0]):
            fh.write(f"{uni_name}\t{ccd}\t{n}\t{len(models)}\t{','.join(models)}\n")
    # loose matches (stereo-ambiguous, analogs) found in ChEBI-mapped sites
    rel = defaultdict(lambda: [0, []])
    for m in cfg.models:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(m).stem)
        f = Path(cfg.outdir) / name / "report.json"
        if f.exists():
            for uni_name, ccd, kind, n in json.loads(f.read_text()).get("related_ligands", []):
                rel[(uni_name, ccd, kind)][0] += n
                rel[(uni_name, ccd, kind)][1].append(name)
    with open(Path(cfg.outdir) / "related_ligands.tsv", "w") as fh:
        fh.write("uniprot_name\tccd\tkind\tcopies\tn_models\tmodels\n")
        for (uni_name, ccd, kind), (n, models) in sorted(rel.items(), key=lambda x: -x[1][0]):
            fh.write(f"{uni_name}\t{ccd}\t{kind}\t{n}\t{len(models)}\t{','.join(models)}\n")
    write_table_audit(cfg)


def write_table_audit(cfg):
    """<outdir>/ligand_table_audit.tsv: for every UniProt ligand name of this run
    that LIGAND_NAME_TO_CCD or --name-map covers, the table's codes next to the
    ChEBI-based ones.  Rows that agree can be removed from the table; the
    others are deliberate choices or table errors.  Disagreements come first."""
    resolutions = ligand_resolver.load_cache(Path(cfg.cache) / CHEBI_CACHE_NAME)
    overrides = {norm_name(k): (v, "name-map") for k, v in cfg.name_map.items()}
    rows = []
    for uni_name, chebi in sorted(run_ligand_names(cfg)):
        table, source = overrides.get(norm_name(uni_name)) or (
            LIGAND_NAME_TO_CCD_ORIG.get(norm_name(uni_name)), "table")
        if not table:
            continue
        r = resolutions.get(chebi)
        auto = r["codes"] if r else []
        status = r["status"] if r else "not resolved"
        only_t = sorted(set(table) - set(auto))
        only_a = sorted(set(auto) - set(table))
        verdict = ("same" if r and not only_t and not only_a
                   else "no ChEBI structure" if r and r["status"] == "generic"
                   else "unresolved" if not r else "differs")
        rows.append((verdict != "same", verdict, uni_name, chebi, source, ",".join(table),
                     ",".join(auto), ",".join(only_t), ",".join(only_a), status))
    with open(Path(cfg.outdir) / "ligand_table_audit.tsv", "w") as fh:
        fh.write("verdict\tuniprot_name\tchebi\tsource\ttable_codes\tchebi_codes\t"
                 "only_in_table\tonly_from_chebi\tchebi_status\n")
        for row in sorted(rows, key=lambda r: (not r[0], r[1], r[2].lower())):
            fh.write("\t".join(row[1:]) + "\n")


def _run_one(m, cfg):
    try:
        rep = process(m, cfg)
        return (Path(m).name, rep.status, rep.count("REVIEW"), rep.count("NEEDS_PARAMS"),
                ",".join(sorted(set(rep.data.get("needs_params", [])))))
    except Exception as exc:  # noqa: BLE001
        log.error("[%s] ERROR: %s", m, exc)
        return (Path(m).name, f"ERROR: {exc}", "-", "-", "")


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
    p.add_argument("--max-downloads", type=int, default=16,
                   help="parallel connections when downloading UniProt/AlphaFill entries")
    p.add_argument("--retries", type=int, default=6,
                   help="attempts per download (backoff 2, 4, 8, ... s with jitter)")
    p.add_argument("--alphafill-coords", choices=("model", "alphafill"), default="model",
                   help="'model': keep the input structure and move AlphaFill ligands onto "
                        "it by local superposition; 'alphafill': use AlphaFill's protein")
    p.add_argument("--fit-radius", type=float, default=10.0,
                   help="residues within this distance of a ligand define its local fit (A)")
    p.add_argument("--max-fit-rmsd", type=float, default=1.0,
                   help="drop ligands whose local backbone fit RMSD exceeds this (A)")
    p.add_argument("--pocket-radius", type=float, default=5.0,
                   help="residues with heavy atoms this close to a ligand form its pocket (A)")
    p.add_argument("--max-pocket-rmsd", type=float, default=1.5,
                   help="drop ligands whose pocket heavy-atom RMSD exceeds this (A)")
    p.add_argument("--contact-dist", type=float, default=4.0,
                   help="ligand-protein contact distance for --min-contact-frac (A)")
    p.add_argument("--min-contact-frac", type=float, default=0.7,
                   help="drop ligands keeping less than this fraction of AlphaFill's contacts")
    p.add_argument("--name-map", action="append", default=[], metavar="NAME=CCD[,CCD]",
                   help="map a UniProt ligand name to one or more CCD codes, replacing "
                        "the automatic (ChEBI-based) mapping, e.g. "
                        "\"(S)-lactate=2OP,LAC\" (repeatable)")
    p.add_argument("--ignore-ligand", action="append", default=[], metavar="NAME",
                   help="UniProt ligand name to ignore, in addition to IGNORED_LIGAND_NAMES "
                        "(repeatable)")
    p.add_argument("--keep-ligands", default="",
                   help="extra CCD codes to keep, comma-separated (e.g. ATP,ADP)")
    p.add_argument("--plddt", type=float, default=50.0, help="terminal trim threshold")
    p.add_argument("--main-chain-frac", type=float, default=0.8,
                   help="with several UniProt mature chains, keep only the longest if it "
                        "holds this fraction of their residues")
    p.add_argument("--min-chain-interface-len", type=int, default=30,
                   help="shorter mature chains count as linked only through a disulfide")
    p.add_argument("--chain-interface-frac", type=float, default=0.5,
                   help="mature chains count as one protein if linked by a disulfide or if "
                        "this fraction of the smaller chain's residues touch the other")
    p.add_argument("--trim-window", type=int, default=3)
    p.add_argument("--clash", type=float, default=2.2)
    p.add_argument("--metal-clash", type=float, default=1.6)
    p.add_argument("--max-clash", type=int, default=3,
                   help="ligand heavy atoms allowed closer than --clash to the protein")
    p.add_argument("--burial-dist", type=float, default=4.0,
                   help="a ligand atom is buried if a protein heavy atom is this close (A)")
    p.add_argument("--min-burial", type=float, default=0.5,
                   help="minimum fraction of buried ligand heavy atoms")
    p.add_argument("--min-site-coverage", type=float, default=0.5,
                   help="minimum fraction of the annotated site residues the ligand contacts")
    p.add_argument("--site-contact", type=float, default=4.0,
                   help="ligand-to-binding-site-residue contact distance (A)")
    p.add_argument("--overlap", type=float, default=2.0,
                   help="ligand copies closer than this are the same site")
    p.add_argument("--pdb2pqr", default="pdb2pqr")
    p.add_argument("--pka-window", type=float, default=1.0)
    p.add_argument("--metal-cutoff", type=float, default=2.8,
                   help="metal-donor distance for extraBonds restraints and overrides (A)")
    p.add_argument("--donor-tol", type=float, default=0.35,
                   help="a protein O/N/S counts as a metal donor up to the metal's typical "
                        "bond length plus this (A)")
    p.add_argument("--metal-shift", type=float, default=1.0,
                   help="largest move (A) when fitting a transplanted metal ion to its "
                        "donors; 0 disables")
    p.add_argument("--min-metal-donors", type=int, default=3,
                   help="metal ions with fewer protein donors are kept only if a kept "
                        "ligand bridges them (e.g. Mg.ATP) or, one donor short, if there is "
                        "room for water to complete them (flagged); otherwise dropped")
    p.add_argument("--min-partner-atoms", type=int, default=6,
                   help="heavy atoms a ligand needs to count as the partner that holds a "
                        "poorly coordinated metal ion (excludes O2, NO, CO, acetate ...)")
    p.add_argument("--denovo-shift", type=float, default=2.5,
                   help="how far (A) a metal ion placed from the annotated residues' donor "
                        "atoms may move from its starting point")
    p.add_argument("--metal-min-sep", type=float, default=3.0,
                   help="minimum distance between two metal ions kept in sharing sites (A)")
    p.add_argument("--metal-k", type=float, default=100.0,
                   help="extraBonds force constant, kcal/mol/A^2")
    p.add_argument("--cys-deprot-patch", default="CYSD",
                   help="patch name for deprotonated Cys in your toppar")
    p.add_argument("--ss-cutoff", type=float, default=2.5)
    p.add_argument("--ss-min", type=float, default=1.8)
    p.add_argument("--toppar", default="./toppar",
                   help="output folder of toppar_prepare.py")
    p.add_argument("--solvate", action="store_true")
    p.add_argument("--padding", type=float, default=12.0)
    p.add_argument("--salt", type=float, default=0.15)
    p.add_argument("--run-vmd", action="store_true")
    p.add_argument("--vmd", default="vmd")
    p.add_argument("--vmd-timeout", type=float, default=3600,
                   help="seconds before a VMD build is abandoned")
    p.add_argument("--dowser-dir", default="",
                   help="Dowser++ installation folder; enables internal-water placement")
    p.add_argument("--dowser-script", default="Dowser++.sh")
    p.add_argument("--dowser-output", default="",
                   help="file Dowser++ writes its final waters to (default: auto-detect)")
    p.add_argument("--dowser-link", action="store_true",
                   help="symlink instead of copy the Dowser++ files per run (large folders)")
    p.add_argument("--dowser-tmp", default=None,
                   help="where per-run Dowser++ folders are created (default: system tmp)")
    p.add_argument("--dowser-path", default="",
                   help="directories prepended to PATH for Dowser++ (e.g. R's bin), ':'-separated")
    p.add_argument("--dowser-timeout", type=float, default=0, help="seconds, 0 = none")
    p.add_argument("--dowser-protein-only", action="store_true",
                   help="give Dowser++ the protein without cofactors")
    p.add_argument("--keep-dowser-tmp", action="store_true")
    p.add_argument("--water-overlap", type=float, default=2.6,
                   help="drop Dowser waters closer than this to protein/cofactor atoms (A)")
    p.add_argument("-j", "--jobs", type=int, default=1,
                   help="models processed in parallel (Dowser++ uses one core per model)")
    p.add_argument("-v", "--verbose", action="store_true")
    cfg = p.parse_args(argv)
    cfg.keep_ligands = {s.strip().upper() for s in cfg.keep_ligands.split(",") if s.strip()}
    try:
        cfg.name_map = dict((k.strip(), [c.strip().upper() for c in v.split(",")
                                         if c.strip()])
                            for k, v in (m.rsplit("=", 1) for m in cfg.name_map))
    except ValueError:
        sys.exit("--name-map expects NAME=CCD or NAME=CCD,CCD")
    if any(not codes for codes in cfg.name_map.values()):
        sys.exit("--name-map expects at least one CCD code after '='")
    # names given explicitly with --name-map are never ignored
    cfg.ignore_names = ({norm_name(n) for n in IGNORED_LIGAND_NAMES | set(cfg.ignore_ligand)}
                        - {norm_name(n) for n in cfg.name_map})
    logging.basicConfig(level=logging.INFO if cfg.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
    cfg.cache = Path(cfg.cache or Path(cfg.outdir) / "cache")
    cfg.cache.mkdir(parents=True, exist_ok=True)
    if cfg.dowser_dir:
        d = Path(cfg.dowser_dir).resolve()
        if not (d / cfg.dowser_script).exists():
            sys.exit(f"{cfg.dowser_script} not found in {d}")
        cfg.dowser_dir = d
    missing = [f for f in TOPPAR_FILES if not (Path(cfg.toppar) / f).exists()]
    if missing:
        sys.exit(f"{cfg.toppar} lacks {', '.join(missing)} - run toppar_prepare.py "
                 "(a version that writes ccd_aliases.json and uniprot_states.json)")
    if shutil.which(cfg.pdb2pqr) is None:
        sys.exit(f"pdb2pqr executable '{cfg.pdb2pqr}' not found")

    global HTTP_RETRIES
    HTTP_RETRIES = max(1, cfg.retries)

    def bar(it, total, desc):
        return tqdm(it, total=total, desc=desc, unit="model") if tqdm else it

    from contextlib import nullcontext
    with (logging_redirect_tqdm() if tqdm else nullcontext()):  # log lines above the bars
        prefetch(cfg)
        resolve_chebi_ids(cfg)
        rows_by = {}
        if cfg.jobs > 1:
            # models are independent; each Dowser++ run uses its own copy of the
            # Dowser++ folder, so they can run side by side (one core each)
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=cfg.jobs) as ex:
                futs = {ex.submit(_run_one, m, cfg): m for m in cfg.models}
                for fut in bar(as_completed(futs), len(futs), "models"):
                    rows_by[futs[fut]] = fut.result()
        else:
            for m in bar(cfg.models, len(cfg.models), "models"):
                rows_by[m] = _run_one(m, cfg)
    rows = [rows_by[m] for m in cfg.models]
    write_name_suggestions(cfg)
    with open(Path(cfg.outdir) / "summary.tsv", "w") as fh:
        fh.write("model\tstatus\tn_review\tn_needs_params\tneeds_params\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    for r in rows:
        print("\t".join(map(str, r)))


if __name__ == "__main__":
    main()
