#!/bin/bash
# Round 2, step 2: Zvuk SASRec search, round 1 (12 configs x 8 epochs, two per machine).
# Offer ids are host-specific: replace them with current ones (vastai search offers ...).
# One launch failed on its host and was re-run elsewhere with 02b.
H=6a7e4a862e2cca1060d3c957691aea11
ZR=AbyssWanderer/moo-seqrec-explanations-zvuk; ZM=AbyssWanderer/sasrec-zvuk
cfg() { local n="zvuk-h$1-l$2-d$3"; echo "train-with+--processed-name+processed_zvuk+--hidden-size+$1+--num-layers+$2+--dropout+$3+--precision+fp32+--grad-clip+1.0+--epochs+8+--patience+5+--val-subset+20000+--run-name+$n+--upload-to+$ZM+--upload-folder+search/$n+--upload-last"; }
offers=(58617187 53157042 57972513 57543041 58992539 57938582)
pairs=("128 2 0.1|128 2 0.2" "128 2 0.3|128 3 0.1" "128 3 0.2|128 3 0.3" "256 2 0.1|256 2 0.2" "256 2 0.3|256 3 0.1" "256 3 0.2|256 3 0.3")
for i in "${!offers[@]}"; do
  IFS='|' read -r a b <<< "${pairs[$i]}"
  job="fetch-processed+processed_zvuk+$ZR,$(cfg $a),$(cfg $b)"
  env="-e SASREC_LABEL=z1r1-$i -e SASREC_PREPARE=none -e SASREC_RESULTS_REPO=$ZR -e SASREC_NEEDS=dataset:${ZR}:read,model:${ZM}:write -e SASREC_JOB=$job"
  vastai create instance "${offers[$i]}" --template_hash $H --disk 40 --label "z1r1-$i" --env "$env" 2>&1 | tail -1
done
