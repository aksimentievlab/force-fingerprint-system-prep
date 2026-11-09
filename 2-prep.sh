#! /bin/bash

    # --alphafill 2-prep.alphafill_cifs/ \

eval "$(micromamba shell hook --shell bash)"
micromamba activate dpfunc

# REVIEW=$(awk '$2 ~ /REVIEW/ {print $1}' 2-prep.log |
# 	     sed 's:\(.*\):filtered_dataset/filtered_mf/structures/\1:')

# uvr af_prep.py filtered_dataset/filtered_mf/structures/{ACYP_ECOLI,ISCX_ECOLI,TFF3_HUMAN}.pdb \
# uvr af_prep.py $REVIEW \

uvr af_prep.py filtered_dataset/filtered_mf/structures/*.pdb \
    --outdir 2-prep \
    --ph 7.0 \
    --alphafill auto \
    --toppar ./toppar \
    --solvate --padding 12 --salt 0.15 \
    --dowser-dir $(pwd)/extern/dowserplusplus \
    --dowser-path "$(dirname $(micromamba run -n dpfunc which R))" \
    --run-vmd \
    -j 64 -v

    # --run-vmd --gcmc  \
    # --gcmc-toppar toppar/top_all36_prot.rtf \
    #               toppar/par_all36m_prot.prm \
    #               toppar/toppar_water_ions.str \
    #               toppar/toppar_all36_prot_heme.str \

#  --vmd /usr/local/bin/vmd \
