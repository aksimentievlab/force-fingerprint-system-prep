#! /bin/bash

uvr af_prep.py filtered_dataset/filtered_mf/structures/ABC*.pdb \
    --outdir 2-prep \
    --ph 7.0 \
    --alphafill 2-prep.alphafill_cifs/ \
    --toppar toppar/top_all36_prot.rtf \
             toppar/toppar_water_ions.str \
             toppar/toppar_all36_prot_heme.str \
    --solvate --padding 12 --salt 0.15 \
    --run-vmd \
    -v

#  --vmd /usr/local/bin/vmd \
