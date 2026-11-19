#!/bin/bash
# Round 2, step 3: Zvuk SASRec search, round 2: the three best round-1 runs resumed to convergence.
# usage: 03-zvuk-search-round2.sh <offer id> <run name>; run for zvuk-h256-l3-d0.1,
# zvuk-h256-l2-d0.1 and zvuk-h256-l3-d0.2 (best validation NDCG@10 after 8 epochs).
H=348e439b735c447748415384361e4873
ZR=AbyssWanderer/moo-seqrec-explanations-zvuk; ZM=AbyssWanderer/sasrec-zvuk
offer=$1; name=$2
job="fetch-processed+processed_zvuk+$ZR,fetch-run+$ZM+search/$name+$name,train-with+--resume+--run-name+$name+--extend-epochs+40+--upload-to+$ZM+--upload-folder+final/$name+--upload-last"
env="-e SASREC_LABEL=z1r2-$name -e SASREC_PREPARE=none -e SASREC_RESULTS_REPO=$ZR -e SASREC_NEEDS=dataset:${ZR}:read,model:${ZM}:write -e SASREC_JOB=$job"
vastai create instance "$offer" --template_hash $H --disk 40 --label "z1r2-$name" --env "$env" 2>&1 | tail -1
