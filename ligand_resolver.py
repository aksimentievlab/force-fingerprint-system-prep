#!/usr/bin/env python3
"""
ligand_resolver.py - map ChEBI IDs, as cited by UniProt for binding-site
ligands and cofactors, to PDB Chemical Component Dictionary (CCD) codes.

Used by af_prep.py (which resolves every ChEBI ID of a run once, before the
models are processed), and usable on its own:

    python ligand_resolver.py CHEBI:30616 CHEBI:60344 --cache chebi_ccd.json

Method
------
UniProt usually cites the major microspecies at pH 7.3 (ATP(4-), CHEBI:30616),
while CCD mostly stores neutral forms. So each ChEBI ID is first expanded to its
"protonation family" via ChEBI's conjugate acid/base relations, keeping only
compounds with the same InChIKey skeleton (first block: connectivity, no
stereo or protonation). Then, from strongest to weakest:

  chebi_xref            CCD entry cross-references the cited ChEBI ID
  chebi_xref_conjugate  ... or another protonation state of it
  inchikey_exact        CCD InChIKey equals that of a family member; catches
                        metal complexes such as heme, which graph search from
                        an InChI misses (standard InChI disconnects metals)
  graph-relaxed-stereo  RCSB graph match that keeps chirality, confirmed by
                        InChIKey skeleton
  ---- the codes above are the result ("codes") ----
  graph-relaxed         graph match ignoring chirality, same skeleton: the
                        CCD entry may leave stereo undefined ("stereo_ambiguous")
  structure_similar     graph match with a different skeleton: an analog such
                        as ANP for ATP ("similar"); never a match

Each result has a status:
  resolved   at least one code
  generic    ChEBI has no usable structure (a class such as "a divalent metal
             cation", an R-group structure, or a cluster such as [2Fe-2S])
             and no CCD entry cross-references it
  no_match   ChEBI has a structure but no CCD entry matches it strictly
  failed     a service failed; not cached, so the next run retries

Only complete results are written to the cache file.

Validated live (Nov 2026) on hemoglobin, EGFR, carbonic anhydrase II, LDHA,
DHFR, adrenodoxin and GOT1 ligands: HEM, OXY, ATP, NAI, NAD, NAP, PLP, ZN, CO.

Requirements: requests.
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CHEBI_API = "https://www.ebi.ac.uk/chebi/backend/api/public"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"

CACHE_VERSION = 1
PUBCHEM_MIN_INTERVAL = 0.25   # PubChem allows at most 5 requests/second
CONJUGATE_DEPTH = 4           # max hops along conjugate acid/base relations
GRAPHQL_CHUNK = 200
MAX_SIMILAR = 20              # analogs kept per result (for reporting only)

METHOD_PRIORITY = {
    "chebi_xref": 0,
    "chebi_xref_conjugate": 1,
    "inchikey_exact": 2,
    "graph-relaxed-stereo": 3,
    "graph-relaxed": 4,
    "structure_similar": 5,
}
STRICT_METHODS = {"chebi_xref", "chebi_xref_conjugate", "inchikey_exact",
                  "graph-relaxed-stereo"}
GRAPH_METHODS = ("graph-relaxed-stereo", "graph-relaxed")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class ServiceError(Exception):
    """A request failed after retries. Never cached."""

    def __init__(self, service, detail, operation=None):
        label = f"{service}/{operation}" if operation else service
        super().__init__(f"{label}: {detail}")
        self.service = service


_local = threading.local()


def session():
    """One requests session per thread (sessions aren't guaranteed thread-safe).
    Retries rate limits, server errors, connection errors and timeouts."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "ligand_resolver.py"
        retry = Retry(total=5, backoff_factor=1.0,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET", "POST"])
        s.mount("https://", HTTPAdapter(max_retries=retry))
        _local.session = s
    return s


def get_json(service, operation, url, **kwargs):
    """GET returning parsed JSON, None for 404; ServiceError otherwise."""
    try:
        r = session().get(url, timeout=kwargs.pop("timeout", 30), **kwargs)
    except requests.RequestException as err:
        raise ServiceError(service, err, operation) from err
    if r.status_code == 404:
        return None
    if not r.ok:
        raise ServiceError(service, f"HTTP {r.status_code}", operation)
    try:
        return r.json()
    except ValueError as err:
        raise ServiceError(service, "invalid JSON", operation) from err


_pubchem_lock = threading.Lock()
_pubchem_last = [0.0]


def pubchem_throttle():
    """Space PubChem requests across all threads."""
    with _pubchem_lock:
        wait = _pubchem_last[0] + PUBCHEM_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _pubchem_last[0] = time.monotonic()


def rcsb_search(query, operation, return_type="mol_definition"):
    body = {"query": query, "return_type": return_type,
            "request_options": {"return_all_hits": True}}
    try:
        r = session().post(RCSB_SEARCH, json=body, timeout=60)
    except requests.RequestException as err:
        raise ServiceError("rcsb_search", err, operation) from err
    if r.status_code == 204:                     # no hits: empty body
        return []
    if not r.ok:
        raise ServiceError("rcsb_search", f"HTTP {r.status_code}", operation)
    hits = [h["identifier"] for h in r.json().get("result_set", [])]
    return [h for h in hits if not h.startswith("PRD_")]   # drop BIRD entries


def graphql(query, variables, operation):
    try:
        r = session().post(RCSB_GRAPHQL, json={"query": query, "variables": variables},
                           timeout=60)
    except requests.RequestException as err:
        raise ServiceError("rcsb_graphql", err, operation) from err
    try:
        payload = r.json()
    except ValueError:
        payload = None
    if payload and payload.get("errors"):
        raise ServiceError("rcsb_graphql", json.dumps(payload["errors"])[:300], operation)
    if not r.ok or payload is None:
        raise ServiceError("rcsb_graphql", f"HTTP {r.status_code}", operation)
    return payload["data"]


def text_terminal(service, attribute, operator, value):
    return {"type": "terminal", "service": service,
            "parameters": {"attribute": attribute, "operator": operator, "value": value}}


# --------------------------------------------------------------------------
# ChEBI (confirmed ChEBI 2.0 layout)
# --------------------------------------------------------------------------

def normalize_chebi(raw_id):
    """'ChEBI:CHEBI:30616', 'CHEBI:30616' or '30616' -> 'CHEBI:30616'."""
    if not raw_id:
        return None
    number = str(raw_id).rsplit(":", 1)[-1].strip()
    return f"CHEBI:{number}" if number.isdigit() else None


def skeleton(inchikey):
    """First InChIKey block: connectivity only, no stereo or protonation.
    ATP(4-) and neutral ATP share it; ATP and ADP don't."""
    return inchikey[:14] if inchikey else None


@lru_cache(maxsize=None)
def chebi_get(chebi_id):
    """ChEBI record; None if ChEBI has no such entry. Failures raise (not cached)."""
    return get_json("chebi", "compound_record", f"{CHEBI_API}/compound/{chebi_id}/")


def chebi_structure(record):
    """(InChI, InChIKey) of the compound's own structure, or (None, None).
    R-group structures are skipped: generic classes can't match a CCD entry."""
    if not isinstance(record, dict):
        return None, None
    default = record.get("default_structure") or {}
    if default.get("is_r_group"):
        return None, None
    inchi, key = default.get("standard_inchi"), default.get("standard_inchi_key")
    if isinstance(inchi, str) and inchi.startswith("InChI="):
        return inchi, key if isinstance(key, str) else None
    return None, None


def chebi_conjugates(record, own_id):
    """ChEBI IDs linked by 'is conjugate acid/base of'. Relation ends are bare
    integers equal to ChEBI accession numbers; the partner is the other end."""
    partners = set()
    if not isinstance(record, dict):
        return partners
    own = int(own_id.split(":")[1])
    rel = record.get("ontology_relations") or {}
    for direction in ("incoming_relations", "outgoing_relations"):
        for r in rel.get(direction) or []:
            if "conjugate" not in str(r.get("relation_type", "")).lower():
                continue
            for end in (r.get("init_id"), r.get("final_id")):
                if isinstance(end, int) and end != own:
                    partners.add(f"CHEBI:{end}")
    return partners


def protonation_family(chebi_id, notes):
    """Breadth-first walk over conjugate relations, limited to compounds with
    the cited compound's InChIKey skeleton. Returns (family, excluded, complete)."""
    family, excluded, complete = {chebi_id}, set(), True
    try:
        _, cited_key = chebi_structure(chebi_get(chebi_id))
    except ServiceError as err:
        notes.append(str(err))
        return family, [], False
    cited = skeleton(cited_key)

    def same_species(partner):
        if not cited:
            return True
        try:
            _, key = chebi_structure(chebi_get(partner))
        except ServiceError as err:
            notes.append(str(err))
            return None
        return key is None or skeleton(key) == cited

    frontier = [chebi_id]
    for _ in range(CONJUGATE_DEPTH):
        nxt = []
        for current in frontier:
            try:
                record = chebi_get(current)
            except ServiceError as err:
                notes.append(str(err))
                complete = False
                continue
            for partner in chebi_conjugates(record, current):
                if partner in family or partner in excluded:
                    continue
                verdict = same_species(partner)
                if verdict is None:
                    complete = False
                elif verdict:
                    family.add(partner)
                    nxt.append(partner)
                else:
                    excluded.add(partner)
        if not nxt:
            break
        frontier = nxt
    else:
        complete = False      # depth limit reached with compounds unexplored
        notes.append("protonation family truncated at the depth limit")
    return family, sorted(excluded), complete


# --------------------------------------------------------------------------
# PubChem fallback: neutralized parent, only when ChEBI gives no neutral form
# --------------------------------------------------------------------------

def pubchem_get(operation, path, **params):
    pubchem_throttle()
    return get_json("pubchem", operation, f"{PUBCHEM}/{path}", params=params)


def pubchem_with_parent(cid):
    """((inchi, inchikey), ...) for a CID and its neutralized PubChem parent."""
    cids = [cid]
    parent = pubchem_get("parent_lookup", f"compound/cid/{cid}/cids/JSON",
                         cids_type="parent")
    pc = (parent or {}).get("IdentifierList", {}).get("CID", [])
    if pc and pc[0] > 0 and pc[0] != cid:
        cids.append(pc[0])
    props = pubchem_get("properties", "compound/cid/"
                        + ",".join(map(str, cids)) + "/property/InChI,InChIKey/JSON")
    rows = (props or {}).get("PropertyTable", {}).get("Properties", [])
    return tuple((r["InChI"], r.get("InChIKey")) for r in rows if r.get("InChI"))


@lru_cache(maxsize=None)
def pubchem_from_inchi(inchi):
    pubchem_throttle()
    try:
        r = session().post(f"{PUBCHEM}/compound/inchi/cids/JSON",
                           data={"inchi": inchi}, timeout=30)
    except requests.RequestException as err:
        raise ServiceError("pubchem", err, "inchi_lookup") from err
    if r.status_code == 404:
        return ()
    if not r.ok:
        raise ServiceError("pubchem", f"HTTP {r.status_code}", "inchi_lookup")
    cids = [c for c in r.json().get("IdentifierList", {}).get("CID", []) if c > 0]
    return pubchem_with_parent(cids[0]) if cids else ()


def is_neutral(inchi):
    return "/q" not in inchi and "/p" not in inchi


def family_structures(chebi_id, family, notes):
    """[(inchi, inchikey)] for the family, cited compound first, plus PubChem's
    neutral parent when ChEBI gave no neutral form. Returns (structures, failed)."""
    out, failed = [], False
    for member in [chebi_id] + sorted(family - {chebi_id}):
        try:
            inchi, key = chebi_structure(chebi_get(member))
        except ServiceError as err:
            notes.append(str(err))
            failed = True
            continue
        if inchi:
            out.append((inchi, key))
    if out and not any(is_neutral(i) for i, _ in out):
        try:
            out.extend(pubchem_from_inchi(out[0][0]))
        except ServiceError as err:
            notes.append(str(err))
            failed = True
    return list(dict.fromkeys(out)), failed


# --------------------------------------------------------------------------
# CCD searches
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def ccd_by_chebi_xref(chebi_ids):
    values = []
    for c in chebi_ids:
        values += [c, c.split(":")[1]]          # storage format may vary
    query = {"type": "group", "logical_operator": "and", "label": "nested-attribute",
             "nodes": [text_terminal("text_chem", "rcsb_chem_comp_related.resource_name",
                                     "exact_match", "ChEBI"),
                       text_terminal("text_chem",
                                     "rcsb_chem_comp_related.resource_accession_code",
                                     "in", values)]}
    return tuple(rcsb_search(query, "chemical_xref"))


@lru_cache(maxsize=None)
def ccd_by_inchikey(inchikeys):
    query = text_terminal("text_chem", "rcsb_chem_comp_descriptor.InChIKey",
                          "in", list(inchikeys))
    return tuple(rcsb_search(query, "inchikey_match"))


@lru_cache(maxsize=None)
def ccd_by_structure(inchi, match_type):
    query = {"type": "terminal", "service": "chemical",
             "parameters": {"value": inchi, "type": "descriptor",
                            "descriptor_type": "InChI", "match_type": match_type}}
    return tuple(rcsb_search(query, "structure_match"))


_CCD_INFO, _CCD_LOCK = {}, threading.Lock()


def ccd_info(ccd_ids):
    """{ccd_id: {"name", "inchikey"}}, fetched in batches and cached."""
    query = """
    query($ids: [String!]!) {
      chem_comps(comp_ids: $ids) {
        chem_comp { id name }
        rcsb_chem_comp_descriptor { InChIKey }
      }
    }"""
    with _CCD_LOCK:
        missing = sorted(set(ccd_ids) - set(_CCD_INFO))
    for i in range(0, len(missing), GRAPHQL_CHUNK):
        data = graphql(query, {"ids": missing[i:i + GRAPHQL_CHUNK]}, "component_info")
        for item in data.get("chem_comps") or []:
            item = item or {}
            comp = item.get("chem_comp") or {}
            if comp.get("id"):
                desc = item.get("rcsb_chem_comp_descriptor") or {}
                with _CCD_LOCK:
                    _CCD_INFO[comp["id"]] = {"name": comp.get("name"),
                                             "inchikey": desc.get("InChIKey")}
    with _CCD_LOCK:
        return {c: _CCD_INFO.get(c, {}) for c in ccd_ids}


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def resolve_chebi(chebi_id):
    """Resolve one ChEBI ID. Never raises; failures give status 'failed'."""
    chebi_id = normalize_chebi(chebi_id)
    result = {"chebi": chebi_id, "name": None, "status": None, "codes": [],
              "methods": {}, "stereo_ambiguous": [], "similar": [], "names": {},
              "family": [], "excluded": [], "complete": True, "notes": []}
    notes = result["notes"]
    try:
        try:
            record = chebi_get(chebi_id)
        except ServiceError as err:
            notes.append(str(err))
            record, result["complete"] = None, False
        if record:
            result["name"] = record.get("ascii_name") or record.get("name")

        family, excluded, fam_complete = protonation_family(chebi_id, notes)
        result["family"], result["excluded"] = sorted(family), excluded
        result["complete"] &= fam_complete

        candidates = {}

        def add(codes, method):
            for c in codes:
                if c not in candidates or METHOD_PRIORITY[method] < METHOD_PRIORITY[candidates[c]]:
                    candidates[c] = method

        add(ccd_by_chebi_xref((chebi_id,)), "chebi_xref")
        conjugates = tuple(sorted(family - {chebi_id}))
        if conjugates:
            add(ccd_by_chebi_xref(conjugates), "chebi_xref_conjugate")

        structures, lookup_failed = family_structures(chebi_id, family, notes)
        result["complete"] &= not lookup_failed
        keys = sorted({k for _, k in structures if k})
        if keys:
            add(ccd_by_inchikey(tuple(keys)), "inchikey_exact")
        for match_type in GRAPH_METHODS:
            for inchi, _ in structures:
                add(ccd_by_structure(inchi, match_type), match_type)

        # Check graph matches against InChIKey skeletons: a different skeleton
        # means a related compound (analog), not the same one.
        info = ccd_info(list(candidates)) if candidates else {}
        query_skeletons = {skeleton(k) for k in keys}
        for code, method in list(candidates.items()):
            key = info.get(code, {}).get("inchikey")
            if (method in GRAPH_METHODS and key and query_skeletons
                    and skeleton(key) not in query_skeletons):
                candidates[code] = "structure_similar"

        order = sorted(candidates, key=lambda c: (METHOD_PRIORITY[candidates[c]], c))
        result["codes"] = [c for c in order if candidates[c] in STRICT_METHODS]
        result["methods"] = {c: candidates[c] for c in result["codes"]}
        result["stereo_ambiguous"] = [c for c in order if candidates[c] == "graph-relaxed"]
        result["similar"] = [c for c in order if candidates[c] == "structure_similar"][:MAX_SIMILAR]
        result["names"] = {c: info.get(c, {}).get("name")
                           for c in result["codes"] + result["stereo_ambiguous"]}

        if result["codes"]:
            result["status"] = "resolved"
        elif not result["complete"]:
            result["status"] = "failed"
        elif record is None:
            result["status"] = "no_match"
            notes.append("not found in ChEBI")
        elif not structures:
            result["status"] = "generic"
        else:
            result["status"] = "no_match"
    except ServiceError as err:            # RCSB is essential
        notes.append(str(err))
        result.update(status="failed", complete=False)
    except Exception as err:  # noqa: BLE001 - one bad ID must not stop a batch
        notes.append(f"unexpected error: {err!r}")
        result.update(status="failed", complete=False)
    return result


def load_cache(path):
    """{chebi_id: result} from a cache file; empty if missing or outdated."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return {}
    if data.get("version") != CACHE_VERSION:
        return {}
    return data.get("entries", {})


def save_cache(path, entries):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps({"version": CACHE_VERSION,
                               "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "entries": entries}, indent=1, sort_keys=True))
    os.replace(tmp, path)


def resolve_many(chebi_ids, cache_path=None, workers=4, refresh=False, progress=None):
    """Resolve ChEBI IDs, reusing and updating the cache file.

    Returns {chebi_id: result} for every valid ID. Complete results are
    saved; failed or partial ones are returned but not saved, so the next
    run retries them. `progress`, if given, wraps the result iterator
    (e.g. a tqdm factory taking (iterable, total))."""
    ids = sorted({c for c in map(normalize_chebi, chebi_ids) if c})
    cache = load_cache(cache_path) if cache_path else {}
    todo = [c for c in ids if refresh or c not in cache]
    fresh = {}
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            results = ex.map(resolve_chebi, todo)
            if progress:
                results = progress(results, len(todo))
            for r in results:
                fresh[r["chebi"]] = r
        for c, r in fresh.items():
            if r["complete"]:
                cache[c] = r
        if cache_path:
            save_cache(cache_path, cache)
    return {c: fresh.get(c) or cache.get(c) for c in ids}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("chebi_ids", nargs="+", help="e.g. CHEBI:30616")
    p.add_argument("--cache", help="cache file to read and update")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--refresh", action="store_true", help="ignore cached results")
    a = p.parse_args(argv)
    for c, r in resolve_many(a.chebi_ids, a.cache, a.workers, a.refresh).items():
        extra = []
        if r["stereo_ambiguous"]:
            extra.append("stereo-ambiguous " + ",".join(r["stereo_ambiguous"]))
        if r["similar"]:
            extra.append("similar " + ",".join(r["similar"][:5]))
        codes = ", ".join(f"{k} ({r['methods'][k]})" for k in r["codes"]) or "-"
        print(f"{c}  {r['name'] or ''}  [{r['status']}]  {codes}"
              + (f"  | {'; '.join(extra)}" if extra else "")
              + (f"  | notes: {'; '.join(r['notes'])}" if r["notes"] else ""))


if __name__ == "__main__":
    main()
