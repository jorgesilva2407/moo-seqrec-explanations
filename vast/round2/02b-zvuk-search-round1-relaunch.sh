#!/bin/bash
# Round 2, step 2b: re-launch of the round-1 pair that failed to start (usage: 02b.sh <offer id>).
H=6a7e4a862e2cca1060d3c957691aea11
ZR=AbyssWanderer/moo-seqrec-explanations-zvuk; ZM=AbyssWanderer/sasrec-zvuk
cfg() { local n="zvuk-h$1-l$2-d$3"; echo "train-with+--processed-name+processed_zvuk+--hidden-size+$1+--num-layers+$2+--dropout+$3+--precision+fp32+--grad-clip+1.0+--epochs+8+--patience+5+--val-subset+20000+--run-name+$n+--upload-to+$ZM+--upload-folder+search/$n+--upload-last"; }
job="fetch-processed+processed_zvuk+$ZR,$(cfg 128 2 0.3),$(cfg 128 3 0.1)"
env="-e SASREC_LABEL=z1r1-1 -e SASREC_PREPARE=none -e SASREC_RESULTS_REPO=$ZR -e SASREC_NEEDS=dataset:${ZR}:read,model:${ZM}:write -e SASREC_JOB=$job"
vastai create instance "$1" --template_hash $H --disk 40 --label z1r1-1 --env "$env" 2>&1 | tail -1
