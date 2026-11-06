# AlphaFold -> CHARMM/NAMD system preparation

Two scripts turn AlphaFold models into simulation-ready CHARMM36 systems for NAMD:

- **`toppar_prepare.py`** builds a clean, psfgen-ready copy of the CHARMM36 force field in `./toppar`.
- **`af_prep.py`** prepares each AlphaFold model in turn:
  - adds cofactors (AlphaFill transplants, checked against UniProt annotations) and internal waters (Dowser++);
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

- **`1-prep/summary.tsv`** has one line per model: `OK`, `REVIEW` (with the number of flags) or `ERROR`.
- **`1-prep/<model>/report.txt`** lists, for each model, what was done and what needs attention.
  REVIEW items come first.
- **The built system** is in `1-prep/<model>/build/`.
  For simulation use:
  - `<model>_ion.psf` and `<model>_ion.pdb`;
  - `<model>_ion_box.txt`, the box size;
  - `metal_extrabonds.txt`, when the model has metal sites.

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

**2. Cofactors and other ligands from AlphaFill.**

- **Moving ligands onto your model.** Transplanted ligands are moved onto *your* model by superposing the C-alpha atoms within 10 Å of each ligand.
  A ligand is transferred only if:
  - the local backbone fit is <= 1.0 Å;
  - the pocket's heavy atoms, including side chains, agree within <= 1.5 Å (pocket = residues near the ligand in either structure;
    symmetric Asp/Glu/Phe/Tyr/Arg atoms are handled);
  - your model keeps >= 70% of the ligand-protein contacts AlphaFill had.
- **Selecting what to keep.** A ligand is kept only if it touches a binding-site residue that UniProt annotates for that ligand, and it fits your model's pocket:
  - no ligand heavy atom closer than 2.2 Å to a protein heavy atom (1.6 Å for metal ions);
  - at least 50% of its heavy atoms buried (a protein heavy atom within 4 Å);
  - contact with at least 50% of the residues UniProt annotates for the site.

  An apo-like pocket, collapsed or opened up, fails these tests.
  At most one copy is kept per annotated site and chain.
  The same tests apply to cofactors and to other annotated ligands (substrates, nucleotides);
  the only difference is the reporting: an annotated cofactor whose site stays empty is flagged REVIEW,
  any other ligand left out is noted as INFO.
- **Reporting.**
  - Generic UniProt ligand names such as "substrate", "a protein" or "DNA" are ignored
  (`IGNORED_LIGAND_NAMES` in `af_prep.py`, extended with `--ignore-ligand`);
  a name given with `--name-map` is never ignored.
  - Transplants UniProt doesn't mention (crystallization additives and the like) are summarized in one line.
  - A transplant in an annotated site whose UniProt name has no code mapping is flagged REVIEW, with the `--name-map` fix spelled out.

**3. Trimming.**
- Residues outside UniProt's mature chain are removed: signal peptide, propeptides, initiator Met.
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
- Residues with pKa near the pH are reported.

**6. Metal sites and disulfides.**
- Residues coordinating a metal are overridden: His tautomer from the coordinating nitrogen, Asp/Glu deprotonated, Cys deprotonated (`--cys-deprot-patch`, default `CYSD`).
- A His bonded to a heme iron gets the PHEM patch.
- Metal-ligand distances are written as NAMD extraBonds restraints.
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
|-- summary.tsv                    one line per model
|-- cache/                         UniProt / AlphaFill downloads
`-- <model>/
    |-- report.txt, report.json    what was done; REVIEW items first
    |-- prepped.pdb                trimmed protein + kept cofactors (heavy atoms)
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

### Common REVIEW flags

| Flag | What to do |
|---|---|
| AlphaFill ligand sits in a UniProt site with no code mapping | add `--name-map "NAME=CODE"` if you want it |
| cofactor NOT built (no matching RESI) | add its topology (e.g. CGenFF via `toppar_prepare.py --extra-stream`) or an alias in `CCD_ALIASES` |
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
| `--name-map NAME=CODE` | | map a UniProt ligand name to a PDB code (repeatable) |
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
| `--clash`, `--metal-clash`, `--max-clash` | 2.2 Å, 1.6 Å, 0 | clash distance and number of clashing atoms allowed |
| `--burial-dist`, `--min-burial` | 4 Å, 0.5 | fraction of ligand atoms that must be buried |
| `--min-site-coverage` | 0.5 | fraction of the annotated site residues the ligand must contact |
| `--overlap` | 2 Å | ligand copies closer than this occupy the same site |

**Trimming, protonation, metals, disulfides**

| Option | Default | Meaning |
|---|---|---|
| `--plddt`, `--trim-window` | 50, 3 | terminal trimming threshold and window |
| `--ph` | 7.0 | pH for PDB2PQR/PROPKA |
| `--pdb2pqr` | `pdb2pqr` | executable |
| `--pka-window` | 1.0 | report residues with pKa this close to the pH |
| `--metal-cutoff` | 2.8 Å | metal-coordination distance |
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

## 5. Known limitations

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
