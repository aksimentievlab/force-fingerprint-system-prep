#! /bin/bash

    # --alphafill 2-prep.alphafill_cifs/ \

BASEDIR=1-prep

GIT_ID=$(git show -q --pretty='%H' | head -c 7)
OLD_GIT_ID=$(git show -q --pretty='%H' HEAD^ | head -c 7)
OUTDIR=$BASEDIR.$GIT_ID

rm -f $BASEDIR
ln -s $OUTDIR $BASEDIR

eval "$(micromamba shell hook --shell bash)"
micromamba activate dpfunc


if [[ -d $OUTDIR ]]; then
    PDBS=$(find $OUTDIR -name report.txt -exec bash -c "head -n 3 {} | grep -q REVIEW" \; -print |
	   sed "s:$OUTDIR/\(.*\)/report.txt:filtered_dataset/filtered_mf/structures/\1.pdb:")
else
    # OLDOUTDIR=$BASEDIR.$OLD_GIT_ID
    # PDBS=$(find $OLDOUTDIR -name report.txt -exec bash -c "head -n 3 {} | grep -q REVIEW" \; -print |
    # 	   sed "s:$OLDOUTDIR/\(.*\)/report.txt:filtered_dataset/filtered_mf/structures/\1.pdb:")
    PDBS=$(echo filtered_dataset/filtered_mf/structures/*.pdb)
fi
    
uvr af_prep.py $PDBS \
    --outdir $OUTDIR \
    --ph 7.0 \
    --alphafill auto \
    --toppar ./toppar \
    --solvate --padding 12 --salt 0.15 \
    --dowser-dir $(pwd)/extern/dowserplusplus \
    --run-vmd \
    -j 64 -v


#  --vmd /usr/local/bin/vmd \
