#! /bin/bash

    # --alphafill 2-prep.alphafill_cifs/ \

uvr af_prep.py filtered_dataset/filtered_mf/structures/ABC*.pdb \
    --outdir 2-prep \
    --ph 7.0 \
    --alphafill auto \
    --toppar toppar/top_all36_prot.rtf \
             toppar/toppar_water_ions.str.vmd \
             toppar/toppar_all36_prot_heme.str \
    --solvate --padding 12 --salt 0.15 \
    --run-vmd --gcmc  \
    --gcmc-toppar toppar/top_all36_prot.rtf \
                  toppar/par_all36m_prot.prm \
                  toppar/toppar_water_ions.str \
                  toppar/toppar_all36_prot_heme.str \
    -v

#  --vmd /usr/local/bin/vmd \
