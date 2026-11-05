#!/usr/bin/env python3
"""
toppar_prepare.py -- fetch the CHARMM36 toppar release, apply CUFIX, and split
CHARMM stream (.str) files into plain RTF/PRM files for psfgen and NAMD.

What it does
  1. Downloads (or takes a local copy of) toppar_c36_*.tgz and unpacks it, untouched,
     to ./toppar.orig
  2. Runs every selected stream file through a small interpreter for the CHARMM
     commands that appear in them (set, if/then/else/endif, single-line if,
     goto/label, return).  This matters: e.g. the cholesterol stream holds two
     alternative models selected by an if-block, and toppar_all36_na_rna_modified
     has dead code after RETURN.  A naive line split would load both / too much.
  3. Writes each stream's topology and parameter sections, verbatim, to
     <out>/stream/<subdir>/<name>.rtf and .prm (several sections are merged into
     one file; section titles become comments; only one END at the end), and
     copies the base RTF/PRM files unchanged, so <out> (./toppar) is complete.
  4. Builds toppar_water_ions_cufix.{rtf,prm}: the stock water/ion file plus the
     CUFIX corrections.  Entries CUFIX supersedes are commented out in place,
     so the result does not depend on how a program handles duplicates.
  5. Writes load lists, all with paths relative to <out>:
       load_psfgen.tcl        proc load_charmm_topologies ?dir?   (default ../toppar)
       namd_parameters.conf   proc load_charmm_parameters ?dir?   (default ../toppar)
       topologies_psfgen.txt  topology files, psfgen order (first definition wins)
       topologies_charmm.txt  topology files, CHARMM order (last definition wins;
                              use this for CHARMM, ParmEd, OpenMM)
       parameters.txt         parameter files, CHARMM order
     and reports
       - RESI/PRES names defined in more than one file (and which one wins),
       - BOND/ANGLE/NONBONDED/NBFIX entries redefined across files with different
         values (the later file wins in CHARMM),
       - optionally, which of your PDB residue names have a CHARMM residue of
         the same name (a name match only -- check the atoms!).

Typical use
  python toppar_prepare.py                         # download, build ./toppar
  python toppar_prepare.py --archive toppar_c36_feb26.tgz --resnames ligands.txt
  python toppar_prepare.py --cufix toppar_water_ions_cufix.str   # use your own CUFIX file
  python toppar_prepare.py --extra-stream lig1.str lig2.str      # e.g. CGenFF output
  python toppar_prepare.py --set chol_model=2                    # pick the Suits cholesterol
  python toppar_prepare.py --verify                              # cross-check with ParmEd

Only the Python standard library is needed (ParmEd only for --verify).
"""

import argparse
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_URL = ("https://mackerell.umaryland.edu/download.php?"
               "filename=CHARMM_ff_params_files/toppar_c36_feb26.tgz")

# Base force-field files in CHARMM read order.
BASE_FILES = [
    ("top_all36_prot.rtf", "par_all36m_prot.prm"),
    ("top_all36_na.rtf", "par_all36_na.prm"),
    ("top_all36_carb.rtf", "par_all36_carb.prm"),
    ("top_all36_lipid.rtf", "par_all36_lipid.prm"),
    ("top_all36_cgenff_v4.6.rtf", "par_all36_cgenff_v4.6.prm"),
]
CGENFF_50 = ("top_all36_cgenff.rtf", "par_all36_cgenff.prm")   # CGenFF 5.0 in feb26

# Stream files (relative to toppar/), in CHARMM read order.  Deliberately left out:
#   *_for_new_psf_gen_code_2022.str  -- need CHARMM's new PSF code (LINEAR, NOANG on PRES)
#   *_ljpme.str                      -- LJ-PME variants, not for cutoff/switching runs
#   toppar_all36_lipid_list.str      -- only streams other files via 'unit 9'
#   toppar_all36_lipid_cationpi_wyf.str -- optional NBFIX set (add with --extra-stream)
#   carb_glycolipid, lipid_lps       -- many residues use atom names such as '2C1',
#                                       which psfgen reads as patch syntax and rejects
#   lignin, cphmd, drude, non_charmm/*, hbond
DEFAULT_STREAMS = [
    # Won's generic ions come first: CHARMM (and psfgen, which upper-cases names)
    # treat 'Tm3p' (thulium) and the nucleic-acid model compound 'TM3P' as the same
    # residue, and the more specific definition should be the one that survives.
    "stream/misc/toppar_ions_won.str",
    "stream/prot/toppar_all36_prot_arg0.str",
    "stream/prot/toppar_all36_prot_c36m_d_aminoacids.str",
    "stream/prot/toppar_all36_prot_fluoro_alkanes.str",
    "stream/prot/toppar_all36_prot_heme.str",
    "stream/prot/toppar_all36_prot_na_combined.str",
    "stream/prot/toppar_all36_prot_retinol.str",
    "stream/prot/toppar_all36_prot_model.str",
    "stream/prot/toppar_all36_prot_modify_res.str",
    "stream/prot/toppar_all36_prot_cofactors.str",
    "stream/prot/toppar_all36_prot_aldehydes.str",
    "stream/prot/toppar_all36_prot_pyridines.str",
    "stream/na/toppar_all36_na_nad_ppi.str",
    "stream/na/toppar_all36_na_rna_modified.str",
    "stream/na/toppar_all36_na_modifications.str",
    "stream/na/toppar_all36_na_model.str",
    "stream/carb/toppar_all36_carb_glycopeptide.str",
    "stream/carb/toppar_all36_carb_imlab.str",
    "stream/carb/toppar_all36_carb_model.str",
    "stream/lipid/toppar_all36_lipid_bacterial.str",
    "stream/lipid/toppar_all36_lipid_miscellaneous.str",
    "stream/lipid/toppar_all36_lipid_cholesterol.str",
    "stream/lipid/toppar_all36_lipid_sphingo.str",
    "stream/lipid/toppar_all36_lipid_detergent.str",
    "stream/lipid/toppar_all36_lipid_ether.str",
    "stream/lipid/toppar_all36_lipid_yeast.str",
    "stream/lipid/toppar_all36_lipid_inositol.str",
    "stream/lipid/toppar_all36_lipid_cardiolipin.str",
    "stream/lipid/toppar_all36_lipid_prot.str",
    "stream/lipid/toppar_all36_lipid_model.str",
]
WATER_IONS = "toppar_water_ions.str"
MARKER = "toppar_prepare.log"                               # identifies our output dir
BUNDLED_CUFIX = "non_charmm/toppar_ions_nbfix_uiuc.str"   # Yoo & Aksimentiev, 2019

# PDB chemical-component code -> CHARMM residue names, tried in order AFTER the
# PDB code itself.  af_prep.py reads these from ccd_aliases.json.
CCD_ALIASES = {
    # water and ions (toppar_water_ions)
    "HOH": ["TIP3"], "NA": ["SOD"], "K": ["POT"], "CL": ["CLA"], "CA": ["CAL"],
    "LI": ["LIT"], "RB": ["RUB"], "CS": ["CES"], "BA": ["BAR"],
    "ZN": ["ZN2"], "CD": ["CD2"],
    # metal ions (toppar_ions_won)
    "FE2": ["FE2P"], "FE": ["FE3P"], "CU1": ["CU1P"], "CU": ["CU2P"], "MN": ["MN2P"],
    "CO": ["CO2P"], "NI": ["NI2P"], "SR": ["SR2P"], "HG": ["HG2P"], "PB": ["PB2P"],
    "AL": ["AL3P"], "LA": ["LA3P"], "GD3": ["GD3P"], "EU3": ["EU3P"], "YT3": ["Y3P"],
    # cofactors whose CHARMM name differs from the PDB code
    "HEM": ["HEME"],
    "RET": ["RTAL"], "RTL": ["RTOL"],      # retinal, retinol
    "FDA": ["FADR"], "FNR": ["FMNR"],      # fully reduced FAD / FMN
    "CMO": ["CO"], "OXY": ["O2"],          # heme ligands: carbon monoxide, O2
}

# UniProt ligand name -> CHARMM residues for that charge/redox state, tried
# BEFORE the PDB code.  af_prep.py reads these from uniprot_states.json.
UNIPROT_STATES = {
    "Na(+)": ["SOD"], "K(+)": ["POT"], "Mg(2+)": ["MG"], "Ca(2+)": ["CAL"],
    "Zn(2+)": ["ZN2"], "Cd(2+)": ["CD2"], "Sr(2+)": ["SR2P"],
    "Fe(2+)": ["FE2P"], "Fe(3+)": ["FE3P"],
    "Cu(+)": ["CU1P"], "Cu(2+)": ["CU2P"],
    "Mn(2+)": ["MN2P"], "Co(2+)": ["CO2P"], "Ni(2+)": ["NI2P"],
    "NAD(+)": ["NAD"], "NADH": ["NAI"], "NADP(+)": ["NAP"], "NADPH": ["NDP"],
    "FAD": ["FAD"], "FADH2": ["FADR"], "FMN": ["FMN"], "FMNH2": ["FMNR"],
    "heme b": ["HEME"],
}

CTRL_NOOPS = {"BOMLEV", "BOMB", "WRNLEV", "WRNL", "PRNLEV", "PRNL", "ECHO",
              "IOFORMAT", "IOFO", "FASTER", "LABEL", "TIMER"}


class StreamError(RuntimeError):
    pass


@dataclass
class Section:
    kind: str            # 'rtf' or 'para'
    header: str
    lines: list
    start: int           # 1-based line numbers in the source file
    end: int
    source: str = ""


@dataclass
class StreamResult:
    path: Path
    sections: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    conditional: bool = False


# --------------------------------------------------------------------------- #
# CHARMM stream interpreter
# --------------------------------------------------------------------------- #
def strip_comment(line):
    """Drop a CHARMM '!' comment (a '!' inside double quotes is kept)."""
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch == "!" and not quoted:
            return line[:i]
    return line


def tokenize(code):
    return re.findall(r'"[^"]*"|\S+', code)


def substitute(code, variables):
    def rep(m):
        name = (m.group(1) or m.group(2)).upper()
        return variables.get(name, m.group(0))
    return re.sub(r"@\{(\w+)\}|@(\w+)", rep, code)


def _as_number(text):
    try:
        return float(text)
    except ValueError:
        return None


def eval_condition(tokens, variables, where, notes):
    if len(tokens) != 3:
        raise StreamError(f"{where}: cannot parse IF condition {' '.join(tokens)!r}")
    a, op, b = tokens

    def value(tok):
        if tok.startswith('"'):
            return tok.strip('"')
        if tok.upper() in variables:      # CHARMM also accepts a bare parameter name
            return variables[tok.upper()]
        return tok

    a, b, op = value(a), value(b), op.upper().strip(".")
    na, nb = _as_number(a), _as_number(b)
    if op in ("EQ", "=="):
        return na == nb if None not in (na, nb) else a.upper() == b.upper()
    if op in ("NE", "!=", "<>"):
        return na != nb if None not in (na, nb) else a.upper() != b.upper()
    if op == "AE":
        return None not in (na, nb) and abs(na - nb) < 1e-5
    if op in ("GT", "GE", "LT", "LE"):
        if None in (na, nb):
            notes.append(f"{where}: non-numeric comparison {a} {op} {b} treated as false")
            return False
        return {"GT": na > nb, "GE": na >= nb, "LT": na < nb, "LE": na <= nb}[op]
    raise StreamError(f"{where}: unknown IF operator {op!r}")


def interpret_stream(path, overrides=None):
    """Return the RTF/PARA sections CHARMM would actually read from a stream file."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    res = StreamResult(path)
    variables = {k.upper(): v for k, v in (overrides or {}).items()}
    locked = set(variables)          # command-line values beat 'set' in the file
    stack = []                       # one [active, taken] pair per open IF block
    i = 0

    def active():
        return all(frame[0] for frame in stack)

    def read_section(header_toks, start_idx):
        kind_tok = header_toks[1].upper() if len(header_toks) > 1 else ""
        if kind_tok.startswith("RTF"):
            kind = "rtf"
        elif kind_tok.startswith("PARA"):
            kind = "para"
        else:
            raise StreamError(f"{path.name}:{start_idx + 1}: unsupported READ {kind_tok!r}")
        if any(t.upper() in ("UNIT", "NAME") for t in header_toks[2:]):
            raise StreamError(f"{path.name}:{start_idx + 1}: READ from another file/unit "
                              "is not supported; list that file directly instead")
        j = start_idx + 1
        while j < len(lines):
            toks = strip_comment(lines[j]).split()
            if toks and toks[0].upper() == "END":
                return Section(kind, lines[start_idx].strip(), lines[start_idx + 1:j],
                               start_idx + 1, j + 1), j + 1
            j += 1
        raise StreamError(f"{path.name}:{start_idx + 1}: READ section without END")

    def simple_command(toks, where):
        """Execute a non-block command; return 'stop', ('goto', label) or None."""
        kw = toks[0].upper()
        if kw == "SET":
            if len(toks) >= 2 and toks[1].upper() not in locked:
                variables[toks[1].upper()] = toks[2].strip('"') if len(toks) > 2 else ""
        elif kw == "IF":
            res.conditional = True
            if len(toks) < 5:
                raise StreamError(f"{where}: cannot parse {' '.join(toks)!r}")
            if eval_condition(toks[1:4], variables, where, res.notes):
                return simple_command(toks[4:], where)
        elif kw in ("RETURN", "STOP"):
            return "stop"
        elif kw == "GOTO":
            res.conditional = True
            return ("goto", toks[1].upper())
        elif kw == "STREAM":
            res.notes.append(f"{where}: nested 'stream {' '.join(toks[1:])}' NOT followed")
        elif kw == "READ":
            raise StreamError(f"{where}: READ inside a single-line IF is not supported")
        elif kw not in CTRL_NOOPS:
            res.notes.append(f"{where}: ignored command {' '.join(toks)!r}")
        return None

    while i < len(lines):
        where = f"{path.name}:{i + 1}"
        code = strip_comment(lines[i]).strip()
        if not code or code.startswith("*"):
            i += 1
            continue
        toks = tokenize(substitute(code, variables))
        kw = toks[0].upper()

        if kw == "IF" and toks[-1].upper() == "THEN":
            res.conditional = True
            if active():
                taken = eval_condition(toks[1:-1], variables, where, res.notes)
                stack.append([taken, taken])
            else:
                stack.append([False, True])
            i += 1
            continue
        if kw == "ELSE":
            if not stack:
                raise StreamError(f"{where}: ELSE without IF")
            stack[-1][0] = not stack[-1][1]
            stack[-1][1] = True
            i += 1
            continue
        if kw == "ENDIF":
            if not stack:
                raise StreamError(f"{where}: ENDIF without IF")
            stack.pop()
            i += 1
            continue

        if kw == "READ":
            section, i = read_section(toks, i)
            if active():
                body = "\n".join(section.lines)
                if re.search(r"@\w", strip_comment(body)):
                    res.notes.append(f"{where}: section contains '@' substitutions; check it")
                res.sections.append(section)
            continue
        if not active():
            i += 1
            continue

        action = simple_command(toks, where)
        if action == "stop":
            break
        if isinstance(action, tuple):
            label = action[1]
            order = list(range(i + 1, len(lines))) + list(range(0, i))
            target = next((k for k in order
                           if [t.upper() for t in strip_comment(lines[k]).split()[:2]]
                           == ["LABEL", label]), None)
            if target is None:
                raise StreamError(f"{where}: GOTO {label}: label not found")
            i = target + 1
            continue
        i += 1
    return res


# --------------------------------------------------------------------------- #
# Writing split files
# --------------------------------------------------------------------------- #
def render(sections, kind, source_desc):
    """Merge sections of one kind into a single plain RTF or PRM text."""
    out = [f"! Generated by toppar_prepare.py from {source_desc}",
           "! Section bodies are copied verbatim; titles became comments.", ""]
    version_written = False
    for sec in (s for s in sections if s.kind == kind):
        out.append(f"! ---- {sec.header}   [{sec.source}: lines {sec.start}-{sec.end}]")
        in_title = True
        version_pending = kind == "rtf"
        for raw in sec.lines:
            stripped = raw.strip()
            if in_title and stripped.startswith("*"):
                out.append("! " + stripped)
                continue
            in_title = False
            code = strip_comment(stripped).strip()
            if version_pending and code:
                version_pending = False
                if re.fullmatch(r"\d+(\s+\d+)?", code):
                    out.append(raw if not version_written else "! " + raw)
                    version_written = True
                    continue
            out.append(raw.rstrip())
        out.append("")
    out.append("END")
    return "\n".join(out) + "\n"


PARAM_HEADERS = {
    "ATOMS": "ATOMS", "BONDS": "BONDS", "BOND": "BONDS", "ANGLES": "ANGLES",
    "ANGLE": "ANGLES", "ANGL": "ANGLES", "THETAS": "ANGLES",
    "DIHEDRALS": "DIHEDRALS", "DIHEDRAL": "DIHEDRALS", "DIHE": "DIHEDRALS", "PHI": "DIHEDRALS",
    "IMPROPER": "IMPROPER", "IMPROPERS": "IMPROPER", "IMPR": "IMPROPER", "IMPHI": "IMPROPER",
    "CMAP": "CMAP", "NONBONDED": "NONBONDED", "NONB": "NONBONDED", "NBONDS": "NONBONDED",
    "NBFIX": "NBFIX", "HBOND": "HBOND", "NBTHOLE": "NBTHOLE", "END": None,
}


def scan_params(lines):
    """Yield (line_index, section, key, values) for BONDS/ANGLES/NONBONDED/NBFIX
    entries.  Dihedrals, impropers and CMAP are multi-term and not keyed here."""
    section, continuation = None, False
    for idx, raw in enumerate(lines):
        code = strip_comment(raw).strip()
        if not code or code.startswith("*"):
            continue
        if continuation:
            continuation = code.endswith("-")
            continue
        toks = code.split()
        head = toks[0].upper()
        if head in PARAM_HEADERS:
            section = PARAM_HEADERS[head]
            continuation = code.endswith("-")
            continue
        try:
            if section == "BONDS":
                key = tuple(sorted(t.upper() for t in toks[:2]))
                vals = tuple(float(x) for x in toks[2:4])
            elif section == "ANGLES":
                types = tuple(t.upper() for t in toks[:3])
                key = min(types, types[::-1])
                vals = tuple(float(x) for x in toks[3:])
            elif section == "NONBONDED":
                key = (toks[0].upper(),)
                vals = tuple(float(x) for x in toks[1:])
            elif section == "NBFIX":
                key = tuple(sorted(t.upper() for t in toks[:2]))
                vals = tuple(float(x) for x in toks[2:])
            else:
                continue
        except (ValueError, IndexError):
            continue
        yield idx, section, key, vals


def multiterm_entries(lines):
    """Number of data lines in DIHEDRALS/IMPROPER/CMAP sections."""
    section, count = None, 0
    for raw in lines:
        code = strip_comment(raw).strip()
        if not code or code.startswith("*"):
            continue
        head = code.split()[0].upper()
        if head in PARAM_HEADERS:
            section = PARAM_HEADERS[head]
        elif section in ("DIHEDRALS", "IMPROPER", "CMAP"):
            count += 1
    return count


RTF_BLOCK_END = ("RESI", "PRES", "MASS", "DEFA", "AUTO", "DECL", "END")


def drop_earlier_duplicates(sections, log, label):
    """CHARMM keeps the LAST definition of a RESI/PRES, psfgen the FIRST.  Comment
    out earlier definitions of a name repeated within one file so both agree."""
    blocks = []                                  # (section index, start, stop, name)
    for si, sec in enumerate(sections):
        if sec.kind != "rtf":
            continue
        current = None
        for li, raw in enumerate(sec.lines):
            toks = strip_comment(raw).split()
            head = toks[0].upper()[:4] if toks else ""
            if head in RTF_BLOCK_END and current is not None:
                blocks.append((si, current[0], li, current[1]))
                current = None
            if head in ("RESI", "PRES") and len(toks) > 1:
                current = (li, toks[1].upper())
        if current is not None:
            blocks.append((si, current[0], len(sec.lines), current[1]))
    last = {name: (si, start) for si, start, _, name in blocks}
    for si, start, stop, name in blocks:
        if last[name] != (si, start):
            lines = sections[si].lines
            for li in range(start, stop):
                lines[li] = "! [superseded by a later definition] " + lines[li]
            log.append(f"{label}: {name} defined more than once; earlier definition "
                       f"(source line {sections[si].start + start + 1}) commented out")


def residues_in(rtf_lines):
    """Return [(kind, name, charge, comment)] for RESI/PRES lines."""
    found = []
    for raw in rtf_lines:
        toks = strip_comment(raw).split()
        if toks and toks[0].upper()[:4] in ("RESI", "PRES"):
            comment = raw.split("!", 1)[1].strip() if "!" in raw else ""
            found.append(("PRES" if toks[0].upper().startswith("PRES") else "RESI",
                          toks[1], toks[2] if len(toks) > 2 else "", comment))
    return found


def apply_cufix(stock, cufix, log):
    """Return the sections of water/ions + CUFIX, with superseded entries commented out."""
    cufix_resi = {r[1].upper() for s in cufix.sections if s.kind == "rtf"
                  for r in residues_in(s.lines)}
    if "TIP3" in cufix_resi:
        log.append(f"CUFIX: {cufix.path.name} redefines TIP3, so it is a complete replacement "
                   f"for {WATER_IONS}; using it INSTEAD of the stock file.  Stock-only content "
                   "(e.g. H3O, H2O2, the updated OH- LJ, Ca2+/SOD-ester NBFIXes) is not loaded.")
        return list(cufix.sections)

    stock_resi = {r[1].upper() for s in stock.sections if s.kind == "rtf"
                  for r in residues_in(s.lines)}
    clash = sorted(cufix_resi & stock_resi)
    if clash:
        raise StreamError(f"CUFIX overlay redefines residues {clash}; refusing to merge")

    new_keys = {}
    for s in cufix.sections:
        if s.kind == "para":
            for _, sec, key, vals in scan_params(s.lines):
                new_keys[(sec, key)] = vals
            if multiterm_entries(s.lines):
                log.append("CUFIX: WARNING overlay has dihedral/improper/CMAP entries; "
                           "these were not checked for overrides")

    merged = []
    for s in stock.sections:
        if s.kind != "para":
            merged.append(s)
            continue
        lines = list(s.lines)
        for idx, sec, key, vals in scan_params(s.lines):
            if (sec, key) in new_keys:
                new = new_keys[(sec, key)]
                lines[idx] = f"! [CUFIX] superseded: {s.lines[idx].strip()}"
                log.append(f"CUFIX: {sec:9s} {' '.join(key):14s} {vals} -> {new}")
        merged.append(Section(s.kind, s.header, lines, s.start, s.end, s.source))
    merged.extend(cufix.sections)
    return merged


# --------------------------------------------------------------------------- #
# Fetch / unpack
# --------------------------------------------------------------------------- #
def fetch_and_unpack(url, archive, dest, force):
    if (dest / "toppar_water_ions.str").exists() and not force:
        print(f"[fetch] {dest} already exists; reusing it (use --force to re-extract)")
        return
    if archive is None:
        archive = Path(url.rsplit("/", 1)[-1].split("=")[-1])
        if not archive.exists():
            print(f"[fetch] downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "toppar_prepare.py"})
            with urllib.request.urlopen(req) as r, open(archive, "wb") as f:
                f.write(r.read())
    print(f"[fetch] unpacking {archive}")
    with tarfile.open(archive) as tar:
        members = []
        for m in tar.getmembers():
            parts = Path(m.name).parts
            if m.name.startswith("/") or ".." in parts or not parts or parts[0] != "toppar":
                raise StreamError(f"refusing suspicious archive member {m.name!r}")
            if m.isfile() or m.isdir():
                members.append(m)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=dest.parent))
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(tmp, members=members, **kwargs)
    if dest.exists():
        shutil.rmtree(dest)
    (tmp / "toppar").rename(dest)
    tmp.rmdir()


# --------------------------------------------------------------------------- #
# Optional ParmEd cross-check
# --------------------------------------------------------------------------- #
def verify_with_parmed(pairs, base_rtfs):
    """Compare ParmEd's reading of each .str with its reading of our rtf+prm."""
    try:
        import parmed
        from parmed.charmm import CharmmParameterSet
    except ImportError:
        print("[verify] ParmEd not installed; skipping (pip install parmed)")
        return True
    import warnings
    warnings.filterwarnings("ignore")

    def digest(ps):
        res = {n: [(a.name, a.type, round(a.charge, 4)) for a in r.atoms]
               for n, r in ps.residues.items()}
        pat = {n: [(a.name, a.type, round(a.charge, 4)) for a in r.atoms]
               for n, r in ps.patches.items()}
        prm = {}
        for attr in ("bond_types", "angle_types", "urey_bradley_types", "improper_types",
                     "improper_periodic_types", "cmap_types", "nbfix_types"):
            prm[attr] = {k: str(v) for k, v in getattr(ps, attr).items()}
        prm["dihedral_types"] = {k: str(list(v)) for k, v in ps.dihedral_types.items()}
        prm["lj"] = {k: (t.epsilon, t.rmin, t.epsilon_14, t.rmin_14)
                     for k, t in ps.atom_types.items()}
        return res, pat, prm

    ok = True
    for src, rtf, prm, conditional in pairs:
        try:
            a = CharmmParameterSet(*base_rtfs)
            a.read_stream_file(str(src))
            b = CharmmParameterSet(*base_rtfs)
            if rtf:
                b.read_topology_file(str(rtf))
            if prm:
                b.read_parameter_file(str(prm))
            same = digest(a) == digest(b)
        except Exception as exc:                      # noqa: BLE001
            print(f"[verify] {src.name}: ParmEd failed ({type(exc).__name__}: {exc})")
            ok = False
            continue
        if same:
            print(f"[verify] {src.name}: identical")
        elif conditional:
            print(f"[verify] {src.name}: differs -- expected, the file has IF/GOTO logic "
                  "that ParmEd ignores (it reads every section)")
        else:
            print(f"[verify] {src.name}: DIFFERS")
            ok = False
    return ok


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--archive", type=Path, help="use this local .tgz instead of downloading")
    ap.add_argument("--toppar", type=Path, default=Path("toppar.orig"),
                    help="unpack the release here, unmodified [toppar.orig]")
    ap.add_argument("--out", type=Path, default=Path("toppar"),
                    help="write the ready-to-use files here [toppar]; a directory from an "
                         "earlier run is replaced, any other non-empty directory is left alone")
    ap.add_argument("--force", action="store_true", help="re-extract even if --toppar exists")
    ap.add_argument("--cufix", default="bundled",
                    help="'bundled' (<toppar>/%s), 'none', or a path to a CUFIX .str" % BUNDLED_CUFIX)
    ap.add_argument("--cgenff", choices=["4.6", "5.0"], default="4.6",
                    help="CGenFF base files [4.6].  Match the version your ligand .str files "
                         "were made with.  Note: with 5.0, toppar_all36_prot_cofactors.str "
                         "(read later) puts back ~40 older bond/angle values")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="CHARMM variable for stream conditionals, e.g. chol_model=2")
    ap.add_argument("--extra-stream", nargs="+", type=Path, default=[],
                    help="additional .str files (e.g. CGenFF ligands), read after the defaults")
    ap.add_argument("--resnames", type=Path,
                    help="file with whitespace-separated PDB residue codes for a coverage report")
    ap.add_argument("--verify", action="store_true", help="cross-check splits with ParmEd")
    args = ap.parse_args(argv)

    overrides = dict(s.split("=", 1) for s in args.set)
    toppar, out = args.toppar.resolve(), args.out.resolve()
    if out == toppar or out.is_relative_to(toppar) or toppar.is_relative_to(out):
        raise SystemExit("--out and --toppar must be separate directories")
    if out.exists() and any(out.iterdir()):
        if not (out / MARKER).exists():
            raise SystemExit(f"{out} exists and was not made by this script; move it away "
                             "or choose another --out")
        shutil.rmtree(out)
    fetch_and_unpack(args.url, args.archive, toppar, args.force)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    base = list(BASE_FILES)
    if args.cgenff == "5.0":
        base[-1] = CGENFF_50
    for rtf, prm in base:
        for f in (rtf, prm):
            if not (toppar / f).exists():
                raise SystemExit(f"missing base file {toppar / f}")

    # ---- interpret and split stream files ----------------------------------
    for rtf, prm in base:                        # unmodified copies
        for f in (rtf, prm):
            shutil.copy2(toppar / f, out / f)
    rtf_files = [(out / r, "base") for r, _ in base]
    prm_files = [(out / p, "base") for _, p in base]
    verify_pairs = []

    def split_one(src, dest_dir, sections_override=None, label=None):
        result = interpret_stream(src, overrides)
        for s in result.sections:
            s.source = src.name
        sections = sections_override if sections_override is not None else result.sections
        drop_earlier_duplicates(sections, log, src.name)
        for note in result.notes:
            log.append(f"note: {note}")
        stem = label or src.stem
        dest_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for kind, ext in (("rtf", ".rtf"), ("para", ".prm")):
            if any(s.kind == kind for s in sections):
                path = dest_dir / (stem + ext)
                path.write_text(render(sections, kind, src.name))
                written[kind] = path
        tag = "  [conditional]" if result.conditional else ""
        shown = src.relative_to(toppar) if src.is_relative_to(toppar) else src
        print(f"[split] {shown} -> {', '.join(p.name for p in written.values())}{tag}")
        return result, written

    stream_list = [toppar / s for s in DEFAULT_STREAMS] + [p.resolve() for p in args.extra_stream]
    for src in stream_list:
        if not src.exists():
            raise SystemExit(f"missing stream file {src}")
        sub = (Path("stream") / src.parent.name if src.is_relative_to(toppar / "stream")
               else Path("extra"))
        result, written = split_one(src, out / sub)
        if "rtf" in written:
            rtf_files.append((written["rtf"], src.name))
        if "para" in written:
            prm_files.append((written["para"], src.name))
        verify_pairs.append((src, written.get("rtf"), written.get("para"), result.conditional))

    # ---- water/ions + CUFIX, read last so its NBFIXes win ------------------
    stock = interpret_stream(toppar / WATER_IONS, overrides)
    for s in stock.sections:
        s.source = WATER_IONS
    if args.cufix.lower() == "none":
        sections, label = stock.sections, "toppar_water_ions"
    else:
        cpath = toppar / BUNDLED_CUFIX if args.cufix == "bundled" else Path(args.cufix).resolve()
        cufix = interpret_stream(cpath, overrides)
        for s in cufix.sections:
            s.source = cpath.name
        sections, label = apply_cufix(stock, cufix, log), "toppar_water_ions_cufix"
    drop_earlier_duplicates(sections, log, label)
    water_rtf, water_prm = out / f"{label}.rtf", out / f"{label}.prm"
    water_rtf.write_text(render(sections, "rtf", f"{WATER_IONS} + CUFIX" if "cufix" in label
                                else WATER_IONS))
    water_prm.write_text(render(sections, "para", f"{WATER_IONS} + CUFIX" if "cufix" in label
                                else WATER_IONS))
    print(f"[split] {WATER_IONS} (+CUFIX: {args.cufix}) -> {water_rtf.name}, {water_prm.name}")
    rtf_files.append((water_rtf, label))
    prm_files.append((water_prm, label))

    # ---- duplicate residues: CHARMM keeps the last, psfgen the first --------
    defined = defaultdict(list)
    resinfo = {}
    for path, _ in rtf_files:
        for kind, name, charge, comment in residues_in(path.read_text(errors="replace").splitlines()):
            defined[name.upper()].append(path.name)
            resinfo[name.upper()] = (kind, name, charge, comment, path.name)   # last wins
    dups = {n: files for n, files in defined.items() if len(files) > 1}
    for n, files in sorted(dups.items()):
        log.append(f"duplicate RESI/PRES {n:6s} in {' , '.join(files)}  -> CHARMM uses {files[-1]}")

    # ---- parameter entries redefined with different values -----------------
    seen = {}
    n_over = 0
    for path, _ in prm_files:
        for _, sec, key, vals in scan_params(path.read_text(errors="replace").splitlines()):
            k = (sec, key)
            if k in seen and seen[k][1] != vals and seen[k][0] != path.name:
                n_over += 1
                log.append(f"override {sec:9s} {' '.join(key):16s} {seen[k][0]} {seen[k][1]} "
                           f"-> {path.name} {vals}")
            seen[k] = (path.name, vals)
            
    # ---- load lists (paths relative to the output directory) ----------------
    (out / "ccd_aliases.json").write_text(json.dumps(CCD_ALIASES, indent=1) + "\n")
    (out / "uniprot_states.json").write_text(json.dumps(UNIPROT_STATES, indent=1) + "\n")

    def rel(paths):
        return [p.relative_to(out).as_posix() for p, _ in paths]

    psfgen_order = rel(reversed(rtf_files))      # psfgen keeps the FIRST definition
    charmm_order = rel(rtf_files)                # CHARMM/ParmEd/OpenMM keep the LAST
    param_order = rel(prm_files)

    def tcl_list(names):
        return "".join(f"        {n}\n" for n in names)

    psf = out / "load_psfgen.tcl"
    psf.write_text(
        "# Generated by toppar_prepare.py.\n"
        "#\n"
        "#   package require psfgen\n"
        "#   source /path/to/toppar/load_psfgen.tcl\n"
        "#   load_charmm_topologies                 ;# reads from ../toppar\n"
        "#   load_charmm_topologies /path/to/toppar ;# or from anywhere else\n"
        "#\n"
        "# psfgen keeps the FIRST definition of a duplicated RESI/PRES; CHARMM and\n"
        "# CHARMM-GUI keep the LAST.  The files are therefore listed in REVERSE CHARMM\n"
        "# order, so psfgen ends up with the same residues CHARMM would use.\n"
        "proc load_charmm_topologies {{toppar ../toppar}} {\n"
        "    foreach f {\n" + tcl_list(psfgen_order) + "    } {\n"
        "        topology [file join $toppar $f]\n"
        "    }\n"
        "}\n")
    namd = out / "namd_parameters.conf"
    namd.write_text(
        "# Generated by toppar_prepare.py.\n"
        "#\n"
        "#   source /path/to/toppar/namd_parameters.conf\n"
        "#   load_charmm_parameters                 ;# reads from ../toppar\n"
        "#   load_charmm_parameters /path/to/toppar ;# or from anywhere else\n"
        "#\n"
        "# CHARMM read order.  Entries redefined in a later file are listed in\n"
        "# toppar_prepare.log; compare with NAMD's DUPLICATE warnings at startup.\n"
        "proc load_charmm_parameters {{toppar ../toppar}} {\n"
        "    paraTypeCharmm on\n"
        "    foreach f {\n" + tcl_list(param_order) + "    } {\n"
        "        parameters [file join $toppar $f]\n"
        "    }\n"
        "}\n")
    (out / "topologies_psfgen.txt").write_text("\n".join(psfgen_order) + "\n")
    (out / "topologies_charmm.txt").write_text("\n".join(charmm_order) + "\n")
    (out / "parameters.txt").write_text("\n".join(param_order) + "\n")

    # ---- residue-name coverage --------------------------------------------
    if args.resnames:
        codes = args.resnames.read_text().split()
        tsv = ["pdb_code\tcharmm_name\tmatch\tkind\tcharge\tfile\tcomment"]
        matched = []
        for code in codes:
            c = code.upper()
            alias = next((a for a in CCD_ALIASES.get(c, []) if a in resinfo), None)
            hit, how = (c, "exact") if c in resinfo else (alias, "alias")
            if hit and hit in resinfo:
                kind, name, charge, comment, fname = resinfo[hit]
                tsv.append(f"{code}\t{name}\t{how}\t{kind}\t{charge}\t{fname}\t{comment}")
                matched.append(code)
        unmatched = [c for c in codes if c not in matched]
        tsv.append("")
        tsv.append("# no CHARMM residue of that name (needs CGenFF or other parameters):")
        tsv.extend(f"# {' '.join(unmatched[i:i + 20])}" for i in range(0, len(unmatched), 20))
        (out / "resname_coverage.tsv").write_text("\n".join(tsv) + "\n")
        print(f"[cover] {len(matched)}/{len(codes)} residue codes have a same-named CHARMM "
              f"residue (name match only -- check atoms!) -> resname_coverage.tsv")

    (out / MARKER).write_text("\n".join(log) + "\n")
    print(f"[done] {len(rtf_files)} topology and {len(prm_files)} parameter files; "
          f"{len(dups)} duplicated residue names, {n_over} parameter overrides "
          f"(details in {out / MARKER})")
    print(f"       psfgen: source {psf}; load_charmm_topologies ?dir?\n"
          f"       NAMD:   source {namd}; load_charmm_parameters ?dir?\n"
          f"       lists:  topologies_psfgen.txt, topologies_charmm.txt, parameters.txt")

    if args.verify:
        base_rtfs = [str(toppar / r) for r, _ in base]
        if not verify_with_parmed(verify_pairs, base_rtfs):
            return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StreamError as exc:
        sys.exit(f"error: {exc}")
