# AlphaFold -> CHARMM/NAMD system preparation

Two scripts, with one helper module, turn AlphaFold models into simulation-ready CHARMM36 systems for NAMD:

- **`toppar_prepare.py`** builds a clean, psfgen-ready copy of the CHARMM36 force field in `./toppar`.
- **`af_prep.py`** prepares each AlphaFold model in turn:
  - adds cofactors (AlphaFill transplants, checked against UniProt annotations) and internal waters (Dowser++);
  UniProt ligands are mapped to PDB codes through their ChEBI IDs by **`ligand_resolver.py`**, which must sit next to `af_prep.py`;
  - assigns protonation states (PDB2PQR/PROPKA), metal sites and disulfides;
  - builds a solvated, ionized PSF/PDB with VMD/psfgen.

Every model gets a report.
Anything that needs a human decision is flagged **REVIEW** instead of being silently guessed.

---

## 1. Setup

### 1.1 Repository and dataset

Clone with the dataset submodule:

```bash
git clone --recurse-submodules <this-repo-url>
cd <this-repo>
# or, in an existing clone:
git submodule update --init
```

The submodule `filtered_dataset` ([BaratiLab/force-fingerprint](https://github.com/BaratiLab/force-fingerprint)) provides the input structures.
Follow its instructions to generate the filtered set, so that the AlphaFold models end up in:

```
filtered_dataset/filtered_mf/structures/*.pdb
```

### 1.2 Environment

The minimal environment is a single micromamba/conda environment.
R is only needed for Dowser++.

```yaml
# environment.yml
name: dpfunc
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - scipy
  - networkx
  - gemmi
  - requests
  - r-base          # Dowser++ (water clustering step)
  - tqdm            # progress bars (optional)
  - pip
  - pip:
      - pdb2pqr>=3.6.2   # also installs propka
```

```bash
micromamba env create -f environment.yml     # or: conda env create -f environment.yml
```

Activate this environment before running either script.
Dowser++ finds R on the `PATH`, which the active environment provides.

Optional extra: `parmed`, used only by `toppar_prepare.py --verify`.

`af_prep.py` needs network access the first time it sees a protein or ligand:
UniProt and AlphaFill for each model, and, for the ligand mapping, ChEBI, PubChem and the RCSB search and data APIs.
Everything is cached in `<outdir>/cache`, so reruns work offline.

### 1.3 External programs

**VMD** (1.9.4 or later), with the psfgen, solvate and autoionize plugins it ships with.
It must be on `PATH` as `vmd`, or passed with `--vmd`.
It is run in text mode;
no display is needed.

**Dowser++**, for internal waters.
Run the install script **from the `extern` directory**:

```bash
cd extern
bash install-dowser-linux.sh
cd ..
```

This installs Dowser++ into `extern/dowserplusplus`.

### 1.4 Force field

```bash
python toppar_prepare.py
```

This downloads the CHARMM36 toppar release and writes `./toppar` (details in section 3).
Run it once, and again whenever you change the force-field selection or add ligand topologies.

---

## 2. Usage

### 2.1 Prepare the force field

```bash
python toppar_prepare.py                                # download, build ./toppar
python toppar_prepare.py --extra-stream lig1.str        # add e.g. CGenFF ligand topologies
python toppar_prepare.py --verify                       # cross-check the split files with ParmEd
```

### 2.2 Prepare the structures

```bash
python af_prep.py filtered_dataset/filtered_mf/structures/*.pdb \
    --outdir 1-prep \
    --ph 7.0 \
    --alphafill auto \
    --toppar ./toppar \
    --solvate --padding 12 --salt 0.15 \
    --dowser-dir "$(pwd)/extern/dowserplusplus" \
    --run-vmd \
    -j 64 -v
```

**Quick test on one structure.** Leave out `--dowser-dir` and `--run-vmd`, then inspect `report.txt` and `build/build.tcl` before running the full set:

```bash
python af_prep.py filtered_dataset/filtered_mf/structures/SOME_PROTEIN.pdb \
    --outdir test-prep --toppar ./toppar -v
```

### 2.3 Checking the results

- **`1-prep/summary.tsv`** has one line per model:
  its status, the number of REVIEW and NEEDS_PARAMS items, and the ligands that need parameters.
  The status is the most serious category present: `ERROR`, `REVIEW`, `NEEDS_PARAMS` or `OK`.
  REVIEW means something about the structure needs a decision;
  NEEDS_PARAMS means only that kept ligands could not be built for lack of CHARMM parameters.
- **`1-prep/needs_params.tsv`** lists those ligands over the whole run, most frequent first,
  so you can decide which are worth parameterizing.
- **`1-prep/<model>/report.txt`** lists, for each model, what was done and what needs attention.
  REVIEW items come first.
- **The built system** is in `1-prep/<model>/build/`.
  For simulation use:
  - `<model>_ion.psf` and `<model>_ion.pdb`;
  - `<model>_ion_box.txt`, the box size;
  - `metal_extrabonds.txt`, when the model has metal sites.

**Reruns.** Each run replaces a model's earlier output:
the build files (including an old PSF), `candidates.pdb` and the report are removed or rewritten,
also when the model now fails.
A model that fails with an error gets a report with status `ERROR` and the error message.
Input files must have distinct names, since the name decides the output folder;
`af_prep.py` stops if two inputs share one.

### 2.4 Running NAMD with the result

```tcl
# NAMD configuration excerpt
source  /path/to/toppar/namd_parameters.conf
load_charmm_parameters /path/to/toppar
structure    <model>_ion.psf
coordinates  <model>_ion.pdb
# cellBasisVector1/2/3 from <model>_ion_box.txt
extraBonds    on                     ;# only if metal_extrabonds.txt exists
extraBondsFile metal_extrabonds.txt
```

Minimize first, because psfgen guesses the hydrogen positions.
Then release positional restraints in stages: heavy atoms, then backbone, then C-alpha.

---

## 3. `toppar_prepare.py` in detail

CHARMM distributes its force field as base RTF/PRM files plus many *stream* files (`.str`).
Stream files mix topology and parameter sections with CHARMM commands (`set`, `if`/`else`, `goto`, `return`).
psfgen and NAMD can't interpret those commands, and naively splitting the files loads the wrong sections.
`toppar_prepare.py` produces a folder that psfgen, NAMD and `af_prep.py` can all read consistently.

### What it does

1. **Fetches the release.** It downloads `toppar_c36_*.tgz` (or uses `--archive`) and unpacks it, untouched, to `./toppar.orig`.
2. **Interprets the stream files.** A small interpreter evaluates `set`, block and single-line `if`, `goto`/`label` and `return`, exactly as CHARMM would, so only the sections CHARMM would actually read are kept.
   For example, only one of the two cholesterol models (chosen with `--set chol_model=N`) is loaded, and code after a `return` is ignored.
3. **Splits each stream.** Each stream's topology and parameter sections are written, verbatim, to `toppar/stream/<subdir>/<name>.rtf` and `.prm`.
   The base RTF/PRM files are copied unchanged.
4. **Applies CUFIX.** The water/ion file is merged with the CUFIX corrections into `toppar_water_ions_cufix.{rtf,prm}`.
   Entries that CUFIX replaces are commented out in place.
5. **Fixes known topology errors.**
   - It removes bonds listed twice within a residue.
     psfgen rejects these with `duplicate bond`, while CHARMM ignores them.
   - It corrects known typos, such as NDP (NADPH), whose phosphate oxygen O2N is otherwise left without a bond.
   - It logs any residue still left with an atom that has no bond.
6. **Resolves duplicate residue names consistently.** CHARMM keeps the *last* definition of a residue name, psfgen the *first*.
   Earlier duplicates within one file are commented out, and the psfgen load list is written in reverse CHARMM order, so psfgen ends up with the same residues CHARMM would use.
7. **Writes the load files:**

| File | Purpose |
|---|---|
| `load_psfgen.tcl` | `load_charmm_topologies ?dir?` for psfgen (used by `af_prep.py` builds) |
| `namd_parameters.conf` | `load_charmm_parameters ?dir?` for NAMD |
| `topologies_psfgen.txt` | topology files in psfgen order (first definition wins) |
| `topologies_charmm.txt` | topology files in CHARMM order (last wins), for CHARMM, ParmEd, OpenMM |
| `parameters.txt` | parameter files in CHARMM order |
| `ccd_aliases.json` | PDB ligand code -> CHARMM residue names, e.g. `ZN` -> `ZN2`, `HEM` -> `HEME` |
| `uniprot_states.json` | UniProt state name -> CHARMM residue, e.g. `Fe(2+)` -> `FE2P`, `NADH` -> `NAI` |
| `toppar_prepare.log` | everything changed or worth knowing: duplicate residues, parameter overrides, CUFIX replacements, bond fixes |

The two JSON tables are defined only in `toppar_prepare.py` (`CCD_ALIASES`, `UNIPROT_STATES`).
`af_prep.py` reads them from the toppar folder, so the naming always matches the topologies psfgen loads.

### Options

| Option | Meaning |
|---|---|
| `--archive FILE.tgz` | use a local toppar release instead of downloading |
| `--url URL` | download from another location |
| `--toppar DIR` | where the untouched release is unpacked (default `toppar.orig`) |
| `--out DIR` | output folder (default `toppar`); a folder from an earlier run is replaced |
| `--force` | unpack the release again |
| `--cufix bundled\|none\|FILE.str` | CUFIX source (default: the copy shipped with the release) |
| `--cgenff 4.6\|5.0` | CGenFF base version; match your ligand stream files |
| `--set NAME=VALUE` | set a CHARMM variable used in stream conditionals, e.g. `chol_model=2` |
| `--extra-stream FILE...` | additional stream files, e.g. CGenFF output for ligands |
| `--resnames FILE` | coverage report: which of your PDB residue codes have a CHARMM residue |
| `--verify` | compare each split against ParmEd's reading of the original stream |

### Deliberately excluded streams

Some streams are left out by default:
- files for CHARMM's newer PSF generator;
- LJ-PME variants;
- the lipid list file;
- streams whose atom names psfgen cannot parse;
- Drude, CpHMD and `non_charmm` files.

The full list and reasons are at the top of the script.
Use `--extra-stream` to add any of them.

---

## 4. `af_prep.py` in detail

### Pipeline (per model)

**1. Accession and UniProt.** The UniProt accession is read from the PDB header (`DBREF`, or the accession in `TITLE`), or from mmCIF `_struct_ref`, with an `AF-<acc>-F<n>` file name as fallback.
The UniProt entry (cached in `<outdir>/cache`) supplies:
- cofactors;
- binding sites (with ligand names and labels);
- the mature chain boundaries;
- modified residues: covalent cofactors and glycosylation are flagged.

Each ligand UniProt names in a cofactor comment or binding site comes with a ChEBI ID.
`ligand_resolver.py` maps these IDs to PDB (CCD) codes, once per run and before any model is processed,
and stores the results in `<outdir>/cache/chebi_ccd.json` (section 5).
`LIGAND_NAME_TO_CCD` in `af_prep.py` and `--name-map` override the automatic mapping;
`<outdir>/ligand_table_audit.tsv` compares every table entry used in a run with the automatic result, so entries that agree can be removed.
Ligands without a structure in ChEBI (generic classes such as "a divalent metal cation", or clusters such as [2Fe-2S]) are skipped,
unless the table maps them;
an unmapped cofactor is flagged REVIEW.
Each report lists the mapping it used and how each code was found.

**2. Cofactors and other ligands from AlphaFill.**

- **Moving ligands onto your model.** Transplanted ligands are moved onto *your* model by superposing the C-alpha atoms within 10 Å of each ligand.
  A ligand is transferred only if:
  - the local backbone fit is <= 1.0 Å;
  - the pocket's heavy atoms, including side chains, agree within <= 1.5 Å (pocket = residues near the ligand in either structure;
    symmetric Asp/Glu/Phe/Tyr/Arg atoms are handled);
  - your model keeps >= 70% of the ligand-protein contacts AlphaFill had.
- **Selecting what to keep.** A ligand is kept only if it touches a binding-site residue that UniProt annotates for that ligand, and it fits your model's pocket:
  - at most 3 ligand heavy atoms closer than 2.2 Å to a protein heavy atom;
  metal atoms, also inside clusters and hemes, only count below 1.6 Å, since they bond to protein N/O/S at 2.0-2.4 Å;
  - at least 50% of its heavy atoms buried (a protein heavy atom within 4 Å);
  - contact with at least 50% of the residues UniProt annotates for the site.

  An apo-like pocket, collapsed or opened up, fails these tests.
  At most one copy is kept per annotated site and chain.
  A site belongs to the UniProt ligand, not to one PDB code:
  when a ligand maps to several codes (Fe(2+): FE2/FE, NAD(+): NAD/NAI), a copy under any of them fills the site,
  and an empty site is reported once, as e.g. "FE2/FE binding site".
  Codes of different metals reached through a generic name ("a divalent metal cation") are not merged.
  Cofactors are selected first, so a substrate cannot take the place of the cofactor it needs.
  Copies of different ligands may not overlap (atoms closer than 2 Å);
  pairs involving a metal atom only count below 1.6 Å, since a metal bonds to ligand atoms at about 2 Å (e.g. O2 on the iron of a dioxygenase).
  Sites of the same ligand that share residues get a second copy only if both can coexist:
  for metal ions, at least 3 Å apart and each with protein donors of its own, as in binuclear sites;
  organic ligands never share.
  A site served by a copy in a neighbouring site is reported as such, not as empty.
- **Metal coordination.** A transplanted metal ion sits where it was in the homolog's crystal structure, while the side chains of your model may be a few tenths of an Å off.
  Each ion is therefore first moved, by at most 1 Å, to where its potential donors sit at typical bond lengths, without approaching other atoms;
  the move is kept only if it does not lose donors, and the report gives its size.
  Donors are protein atoms that can coordinate a metal (side-chain O and S, histidine N, backbone O)
  within the metal's typical bond length plus 0.35 Å,
  e.g. about 2.1 Å for Zn-N/O, 2.3 Å for Zn-S, 2.4 Å for Ca-O and 2.8 Å for K-O.
  The same distances define the extraBonds restraints.
  A metal ion held by fewer than 3 protein donors is usually completed by a substrate or nucleotide, as in Mg.ATP.
  It is kept only if a kept ligand bridges it (an O/N/S of that ligand within bonding distance of the ion);
  such a partner must have at least 6 heavy atoms (`--min-partner-atoms`), so that small molecules that bind only to the metal,
  such as O2, NO or acetate, do not count, and the ion then stays at its transplanted position, where it bonds to the partner.
  An ion one donor short (2 by default) is kept if there is room for a water molecule at bonding distance,
  since waters from solvation or Dowser++ can complete its coordination;
  it is flagged REVIEW as a warning, and with Dowser++ the report says whether one of its waters reaches the ion.
  Otherwise the ion is dropped and its site flagged REVIEW, whether or not UniProt calls it a cofactor:
  the ion may be substrate-bridged.
  Conversely, a small non-cofactor ligand (fewer heavy atoms than `--min-partner-atoms`, e.g. O2)
  that coordinates the position of a metal ion that was not placed is dropped too,
  since it would bind only with the metal there.
- **Metal sites without a usable copy.** An annotated metal site that is still empty (no copy, or every copy dropped)
  gets an ion placed from the donor atoms of its annotated residues:
  starting from the centroids of donor pairs and triples, the ion moves by up to 2.5 Å (`--denovo-shift`) to fit them,
  and is kept under the same coordination rules (no partner ligand counts here).
  This also works when AlphaFill has no file for the protein.
  The same tests apply to cofactors and to other annotated ligands (substrates, nucleotides);
  the only difference is the reporting: an annotated cofactor whose site stays empty is flagged REVIEW,
  any other ligand left out is noted as INFO.
- **Cofactors without annotated residues.** UniProt sometimes names a cofactor without binding-site positions.
  Copies of such a cofactor are kept if they pass the clash and burial tests and, for metal ions, have at least 3 protein donors.
  The number of copies is taken from UniProt's cofactor note ("Binds 2 Mg(2+) ions per subunit"), otherwise one per chain.
- **Candidates for empty sites.** When a binding site stays empty, or a cofactor without annotated residues cannot be placed,
  the copies that were considered are written to `candidates.pdb` in the model's frame:
  copies that failed the fit or coordination tests, copies rejected during transfer (placed with their local superposition),
  and copies within 8 Å of the site that did not touch an annotated residue.
  The B-factor column holds the site number and the occupancy column the rank within the site (1 = best);
  REMARK 250 lines describe each site and list every candidate with its fit values and the reason it was not kept.
  Empty sites without any candidate are listed too.
  The site's REVIEW line gives its site number.
  `candidates.tcl` prints the REMARK lines (headed by the model, UniProt accession and protein name),
  loads `prepped.pdb` with the annotated residues shown (in red for sites without candidates)
  and the candidates colored by site.
- **Reporting.**
  - When a cofactor's missing site is already explained by a site-level line (site left empty, ion dropped, no copy fitting),
  the summary line "annotated cofactor ... not placed" is noted as INFO instead of repeating the REVIEW.
  - Generic UniProt ligand names such as "substrate", "a protein" or "DNA" are ignored
  (`IGNORED_LIGAND_NAMES` in `af_prep.py`, extended with `--ignore-ligand`);
  a name given with `--name-map` is never ignored.
  - Transplants UniProt doesn't mention (crystallization additives and the like) are summarized in one line.
  - A transplant in the site of a ChEBI-mapped ligand that matches it only when stereochemistry is ignored
  (the CCD entry may leave stereo undefined) is not kept and is flagged REVIEW, with the `--name-map` fix spelled out.
  A transplant that is a related compound (an analog, such as ANP in an ATP site) is not kept and is noted as INFO.
  Both are collected in `<outdir>/related_ligands.tsv`.
  - A transplant in the site of a ligand that remains unmapped (no ChEBI ID, or no CCD entry matches it) is flagged REVIEW, with the `--name-map` fix spelled out,
  unless the pair is a known coincidence listed in `NOT_A_MAPPING` (an ion in a sugar site and the like).
  All such suggestions of a run are collected in `<outdir>/ligand_name_suggestions.tsv`;
  add the coincidental ones to `NOT_A_MAPPING` and the genuine ones to `LIGAND_NAME_TO_CCD`.

**3. Trimming.**
- Residues outside UniProt's mature chain are removed: signal peptide, propeptides, initiator Met.
- When UniProt lists several mature chains, the chains are first tested for links in the model:
  a disulfide, or an interface covering at least 50% of the smaller chain's residues
  (interfaces only count for chains of at least 30 residues; a short peptide may just lie on the precursor's surface).
  - Overlapping chains are alternative forms; the longest is kept.
  - The longest chain and all chains linked to it are kept.
  Each becomes its own segment with its own termini, and excised linkers are removed,
  as for chymotrypsin-like proteases or insulin.
  - Unlinked chains are dropped if what is kept holds at least 80% of the mature residues.
  - Otherwise the protein is likely a polyprotein of independent products:
  no mature boundaries are applied and the model is flagged REVIEW, so you can choose the product to simulate.
  - `report.json` lists the segments under `segments`.
- Disordered termini are trimmed by pLDDT (`--plddt`, `--trim-window`).
- Residue numbering stays UniProt numbering.
- Native termini get NTER/CTER (PROP/GLYP for an N-terminal Pro/Gly).
  Termini created by trimming get neutral caps, ACE/ACP and CT3.

**4. Internal waters (Dowser++, optional).**
- Dowser++ runs on the trimmed protein plus cofactors.
  If that fails, it is retried protein-only.
- Each model runs in its own temporary copy of the Dowser++ folder, so `-j` is safe.
- Waters overlapping protein, cofactors or each other are dropped.
  The rest are renumbered in chain W.

**5. Protonation.**
- PDB2PQR with PROPKA, at `--ph`.
- States are read from the hydrogens PDB2PQR adds: HSD/HSE/HSP, ASPP, GLUP, LSN.
- PDB2PQR's coordinates are kept, including its Asn/Gln/His flips.
- The kept cofactors and ligands are included:
  PROPKA accounts for organic ligands with its ligand model and treats metal ions as point charges.
  Organic ligands are renamed `LG1`, `LG2`, ... for PDB2PQR, so that none is mistaken for one of its own templates;
  should PDB2PQR still fail with the ligands present, it is rerun without them and the model is flagged.
- Residues with pKa near the pH are reported.

**6. Metal sites and disulfides.**
- Residues coordinating a metal through a side-chain atom are overridden:
  His tautomer from the coordinating ring nitrogen (the nearer one, if both are within reach of the same ion),
  Asp/Glu deprotonated, Cys deprotonated (`--cys-deprot-patch`, default `CYSD`).
  A residue that only coordinates through its backbone oxygen keeps its state.
- A His bonded to a heme iron gets the PHEM patch.
- Metal-ligand distances are written as NAMD extraBonds restraints.
  A restraint whose atoms are not found in the built system is skipped and flagged.
- Disulfides are found by SG-SG distance and cross-checked against PDB2PQR's own detection.

**7. Build.**

- **Topologies.** `build/build.tcl` loads the topologies via `load_psfgen.tcl`.
- **Protein and patches.** It builds one segment per chain (`PRO<chain>`) and applies the protonation, disulfide and terminal patches.
- **Cofactor residue names.** Each cofactor is matched to a CHARMM residue by its heavy-atom bond graph, which also gives the atom-name mapping.
  The candidates are tried in this order:
  1. the residue for its UniProt charge/redox state;
  2. the PDB code;
  3. its aliases.

  Cofactors without a matching CHARMM residue are written out and flagged, not built.
- **Waters, solvation and ions.** Internal waters go into segment `DWAT`.
  With `--solvate`, the system is solvated and ionized, and the box size is written.
  With `--run-vmd`, VMD runs the script, and psfgen errors or a missing PSF are flagged.

### Output layout

```
<outdir>/
|-- summary.tsv                    one line per model: status, counts, ligands needing parameters
|-- needs_params.tsv               ligands not built for lack of parameters, over all models
|-- ligand_name_suggestions.tsv    unmapped UniProt names with the codes found in their sites
|-- related_ligands.tsv            stereo-ambiguous matches and analogs found in mapped sites
|-- ligand_table_audit.tsv         LIGAND_NAME_TO_CCD / --name-map entries vs. the ChEBI-based mapping
|-- cache/                         UniProt / AlphaFill downloads, chebi_ccd.json (ligand mapping)
`-- <model>/
    |-- report.txt, report.json    what was done: REVIEW, then NEEDS_PARAMS, then INFO
    |-- prepped.pdb                trimmed protein + kept cofactors (heavy atoms)
    |-- candidates.pdb             copies considered for sites left empty (see below)
    |-- candidates.tcl             shows them in VMD: vmd -e candidates.tcl
    |-- dowser/                    Dowser++ logs and raw output
    |-- pdb2pqr/                   PDB2PQR input, output, PROPKA log
    `-- build/
        |-- build.tcl              psfgen script (rerun with: vmd -dispdev text < build.tcl)
        |-- protein_<chain>.pdb, lig_NN_<code>.pdb, water_internal.pdb
        |-- <model>.psf/.pdb       vacuum build
        |-- <model>_ion.psf/.pdb   solvated and ionized system
        |-- <model>_ion_box.txt    box dimensions (Å)
        |-- metal_extrabonds.txt   NAMD extraBonds file (metal sites only)
        `-- vmd.log
```

### Common REVIEW and NEEDS_PARAMS items

| Flag | What to do |
|---|---|
| AlphaFill ligand sits in a UniProt site with no code mapping | add `--name-map "NAME=CODE"` if you want it, or the pair to `NOT_A_MAPPING` if it is a coincidence |
| AlphaFill ligand matches only when stereochemistry is ignored | check the CCD entry; add `--name-map "NAME=CODE,CODE"` (the fix is spelled out) to keep it |
| UniProt cofactor has no CCD mapping | the reason is given (no ChEBI ID, no structure in ChEBI, no CCD match, lookup failed); add it to `LIGAND_NAME_TO_CCD` or use `--name-map` |
| NEEDS_PARAMS: ligand NOT built (no matching RESI) | add its topology (e.g. CGenFF via `toppar_prepare.py --extra-stream`) or an alias in `CCD_ALIASES` |
| NEEDS_PARAMS: cluster NOT built (bonded model needed) | add published parameters for the cluster and its bonds to the protein |
| REVIEW: ligand NOT built (heavy-atom graph does not match) | the transplanted copy is likely incomplete or distorted; check `build/lig_*.pdb` |
| annotated cofactor not placed / cofactor binding site left empty | AlphaFill had no copy that fits the model's pocket (the reason is given); consider a homolog structure or co-folding |
| UniProt suggests a state that isn't available | the cofactor was built from its PDB code; check its charge/redox state |
| covalent cofactor/modification | needs a custom patch (e.g. PLP-Lys) |
| Dowser++ failed | see `dowser/*.log`; the model is built without internal waters |
| psfgen reported errors | see `build/vmd.log` |

### Options

**Inputs and annotations**

| Option | Default | Meaning |
|---|---|---|
| `models` | | AlphaFold PDB/mmCIF files |
| `--outdir` | `af_prep_out` | output folder |
| `--toppar` | `./toppar` | output folder of `toppar_prepare.py` |
| `--cache` | `<outdir>/cache` | UniProt/AlphaFill cache |
| `--max-downloads` | 16 | parallel connections for UniProt/AlphaFill downloads, which are done before processing starts |
| `--retries` | 6 | attempts per download; dropped connections, timeouts and HTTP 429/5xx are retried with backoff (2, 4, 8, ... s) |
| `--accession` | | UniProt accession, for single-model runs without header information |
| `--no-uniprot` | | skip UniProt; with the binding-site rule, no cofactors are kept |
| `--name-map NAME=CODE[,CODE]` | | map a UniProt ligand name to one or more PDB codes, replacing the automatic mapping (repeatable) |
| `--keep-ligands A,B` | | extra PDB codes to allow |
| `--ignore-ligand NAME` | | UniProt ligand name to ignore, in addition to `IGNORED_LIGAND_NAMES` (repeatable) |

**AlphaFill**

| Option | Default | Meaning |
|---|---|---|
| `--alphafill` | `auto` | `auto` (download), a folder of AlphaFill `.cif` files, or `none` |
| `--alphafill-url` | | download URL template; verify it before batch runs |
| `--alphafill-coords` | `model` | `model`: move ligands onto your structure; `alphafill`: use AlphaFill's protein |
| `--fit-radius` | 10 Å | residues used for the local superposition |
| `--max-fit-rmsd` | 1.0 Å | maximum local backbone RMSD |
| `--pocket-radius` | 5 Å | residues forming the pocket |
| `--max-pocket-rmsd` | 1.5 Å | maximum pocket heavy-atom RMSD |
| `--contact-dist` | 4 Å | contact distance for `--min-contact-frac` |
| `--min-contact-frac` | 0.7 | minimum fraction of AlphaFill's contacts kept |
| `--site-contact` | 4 Å | ligand-binding-site contact distance |
| `--clash`, `--metal-clash`, `--max-clash` | 2.2 Å, 1.6 Å, 3 | clash distance and number of clashing atoms allowed |
| `--burial-dist`, `--min-burial` | 4 Å, 0.5 | fraction of ligand atoms that must be buried |
| `--min-site-coverage` | 0.5 | fraction of the annotated site residues the ligand must contact |
| `--overlap` | 2 Å | ligand copies closer than this occupy the same site |

**Trimming, protonation, metals, disulfides**

| Option | Default | Meaning |
|---|---|---|
| `--plddt`, `--trim-window` | 50, 3 | terminal trimming threshold and window |
| `--main-chain-frac` | 0.8 | fraction of the mature residues the kept chains must hold before unlinked chains are dropped |
| `--chain-interface-frac` | 0.5 | mature chains count as linked if this fraction of the smaller chain's residues touch the other |
| `--min-chain-interface-len` | 30 | shorter mature chains count as linked only through a disulfide |
| `--ph` | 7.0 | pH for PDB2PQR/PROPKA |
| `--pdb2pqr` | `pdb2pqr` | executable |
| `--pka-window` | 1.0 | report residues with pKa this close to the pH |
| `--metal-cutoff` | 2.8 Å | metal-donor distance for extraBonds restraints and residue overrides |
| `--donor-tol` | 0.35 Å | a protein atom counts as a metal donor up to the metal's typical bond length plus this |
| `--metal-shift` | 1 Å | largest move when fitting a transplanted metal ion to its donors; 0 disables |
| `--min-partner-atoms` | 6 | heavy atoms a ligand needs to count as the partner holding a poorly coordinated metal ion |
| `--min-metal-donors` | 3 | metal ions with fewer protein donors are kept only if a kept ligand bridges them, or, one donor short, if there is room for water (flagged) |
| `--denovo-shift` | 2.5 Å | how far an ion placed from the annotated residues' donor atoms may move from its starting point |
| `--metal-min-sep` | 3 Å | minimum distance between two metal ions in sites that share residues |
| `--metal-k` | 100 | extraBonds force constant (kcal/mol/Å^2) |
| `--cys-deprot-patch` | `CYSD` | patch for metal-bound cysteines |
| `--ss-cutoff`, `--ss-min` | 2.5 Å, 1.8 Å | SG-SG disulfide window |

**Dowser++**

| Option | Default | Meaning |
|---|---|---|
| `--dowser-dir` | | Dowser++ folder; enables internal waters |
| `--dowser-script` | `Dowser++.sh` | entry script |
| `--dowser-path` | | directories prepended to `PATH`, if R is not already on it |
| `--dowser-output` | auto | name of the final water file, if auto-detection picks the wrong one |
| `--dowser-link` | | symlink instead of copying the Dowser++ folder per run |
| `--dowser-tmp` | system tmp | where per-run folders are created |
| `--dowser-timeout` | none | seconds per run |
| `--dowser-protein-only` | | run Dowser++ without cofactors |
| `--keep-dowser-tmp` | | keep per-run folders for debugging |
| `--water-overlap` | 2.6 Å | minimum distance of a water to protein/cofactor atoms |

**Build and execution**

| Option | Default | Meaning |
|---|---|---|
| `--solvate`, `--padding`, `--salt` | off, 12 Å, 0.15 M | solvation and ionization |
| `--run-vmd`, `--vmd`, `--vmd-timeout` | off, `vmd`, 3600 s | run the build |
| `-j/--jobs` | 1 | models in parallel (Dowser++ uses one core per model) |
| `-v` | | verbose log |

---

## 5. `ligand_resolver.py` in detail

UniProt identifies ligands by ChEBI ID; AlphaFill and the PDB use Chemical Component Dictionary (CCD) codes.
`ligand_resolver.py` translates between the two.
`af_prep.py` calls it for every ChEBI ID of a run before processing the models;
it also works on its own:

```bash
python ligand_resolver.py CHEBI:30616 CHEBI:60344 --cache chebi_ccd.json
```

### Method

UniProt usually cites the major microspecies at pH 7.3 (e.g. ATP(4-), CHEBI:30616), while CCD mostly stores neutral forms.
Each ChEBI ID is therefore first expanded to its *protonation family*:
the compounds linked to it by ChEBI's "is conjugate acid/base of" relations,
restricted to those with the same InChIKey skeleton (connectivity only, no stereo or protonation).
CCD entries are then searched with these methods, from strongest to weakest:

| Method | Meaning |
|---|---|
| `chebi_xref` | the CCD entry cross-references the cited ChEBI ID |
| `chebi_xref_conjugate` | ... or another protonation state of it |
| `inchikey_exact` | the CCD InChIKey equals that of a family member; catches metal complexes such as heme, which a structure search misses |
| `graph-relaxed-stereo` | RCSB structure match that keeps chirality, confirmed by the InChIKey skeleton |
| `graph-relaxed` | structure match ignoring chirality, same skeleton: the CCD entry may leave stereo undefined (*stereo-ambiguous*); reported, not used |
| `structure_similar` | structure match with a different skeleton: an analog such as ANP for ATP (*similar*); reported, never a match |

Only the first four methods produce codes that `af_prep.py` uses.

### Result status

| Status | Meaning |
|---|---|
| `resolved` | at least one CCD code |
| `generic` | ChEBI has no usable structure (a class, an R-group structure, or a cluster) and no CCD entry cross-references it |
| `no_match` | ChEBI has a structure, but no CCD entry matches it strictly |
| `failed` | a web service failed; not cached, so the next run retries |

Only complete results are written to the cache file, so a temporary outage never leaves a wrong entry behind.
Requests are retried on rate limits and server errors, and PubChem is queried at most 5 times per second.

### Options

| Option | Default | Meaning |
|---|---|---|
| `chebi_ids` | | ChEBI IDs to resolve, e.g. `CHEBI:30616` |
| `--cache` | | cache file to read and update |
| `--workers` | 4 | parallel lookups |
| `--refresh` | | ignore cached results |

## 6. Known limitations

- **Single structures only.** Each model is a single AlphaFold prediction, and UniProt binding sites are applied to every chain (homomers).
- **Some cofactors have no parameters.** Covalent cofactors, iron-sulfur clusters, heme c, PLP, TPP, coenzyme A, biotin and molybdopterin have no standard CHARMM36 topology in this setup.
  They are flagged rather than built.
- **Charge and redox states come from annotation.** They are taken from UniProt names where available, otherwise from the PDB code of the transplanted ligand.
  Neither is a calculation.
- **Equivalent atoms are assigned arbitrarily.** Atoms that are equivalent in the bond graph, such as the terminal phosphate oxygens, may receive CHARMM names in a different order than in the crystal structure.
  This only matters where the topology protonates one of them.
- **Dowser++ places waters by an energy threshold,** not by equilibrium sampling.
  Marginal waters can leave during equilibration.
- **The AlphaFill URL template has not been verified.** If `--alphafill auto` fails, download the files and pass the folder instead.
- **The ligand mapping depends on external databases.** ChEBI relations, CCD cross-references and RCSB search results can be incomplete;
  a ligand that stays unmapped is reported, and `--name-map` or `LIGAND_NAME_TO_CCD` fixes it.
