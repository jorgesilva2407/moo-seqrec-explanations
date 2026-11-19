#!/bin/bash
# Round 2, step 1: job definitions of the first machines (source this file, then pass an
# $*_ENV to `vastai create instance <offer> --template_hash <hash> --env "$X_ENV"`).
# Run from bash, not zsh: zsh reads "$VAR:write" as a variable modifier.
# Z0 = Zvuk preprocessing + 1-epoch smoke run; M1 = held-out MOEA tuning; M2 = (N,C)-NSGA-II and
# Shapley on main; M3 = redrawn instance set + exact front at n = 16.
# The Zvuk run was restarted by hand after `six` turned out missing (see 01b); image sha-d7c46c6
# includes it, so on that image Z0 runs as written.

ZR=AbyssWanderer/moo-seqrec-explanations-zvuk
MR=AbyssWanderer/moo-seqrec-explanations-ml20m
ZM=AbyssWanderer/sasrec-zvuk
PREP="prepare-data+--dataset+zvuk+--processed-name+processed_zvuk+--max-items+50000+--max-users+100000"
# Z0: preprocess once, share it, then a 1-epoch smoke run of the largest config (memory, time per epoch)
Z0_ENV="-e SASREC_LABEL=z0prep -e SASREC_PREPARE=none -e SASREC_RESULTS_REPO=$ZR -e SASREC_NEEDS=dataset:${ZR}:write,model:${ZM}:write -e SASREC_JOB=$PREP,push-processed+processed_zvuk+$ZR,train-with+--processed-name+processed_zvuk+--hidden-size+256+--num-layers+3+--dropout+0.1+--precision+fp32+--grad-clip+1.0+--epochs+1+--val-subset+5000+--run-name+smoke-h256-l3"
T="--instance-set+tune+--per-stratum+10+--seeds+0+1+2+--m+10+--pop-size+25"
M1_JOB="explain-run+--stage+tune-moead-tch-p5+$T+--methods+moead+--decomposition+tchebicheff+--partitions+5,explain-run+--stage+tune-moead-tch-p6+$T+--methods+moead+--decomposition+tchebicheff+--partitions+6,explain-run+--stage+tune-moead-pbi-p5+$T+--methods+moead+--decomposition+pbi+--partitions+5,explain-run+--stage+tune-moead-pbi-p6+$T+--methods+moead+--decomposition+pbi+--partitions+6,explain-run+--stage+tune-nsga3-p5+$T+--methods+nsga3+--partitions+5,explain-run+--stage+tune-nsga3-p6+$T+--methods+nsga3+--partitions+6,explain-run+--stage+tune-smsemoa-p25+--instance-set+tune+--per-stratum+10+--seeds+0+1+2+--m+10+--methods+smsemoa+--pop-size+25,explain-run+--stage+tune-smsemoa-p50+--instance-set+tune+--per-stratum+10+--seeds+0+1+2+--m+10+--methods+smsemoa+--pop-size+50"
M1_ENV="-e SASREC_LABEL=m1tune -e SASREC_RESULTS_REPO=$MR -e SASREC_JOB=$M1_JOB"
M2_ENV="-e SASREC_LABEL=m2main -e SASREC_RESULTS_REPO=$MR -e SASREC_JOB=explain-run+--stage+main+--methods+nsga2_nc+shapley+--pop-size+25"
M3_ENV="-e SASREC_LABEL=m3redraw -e SASREC_RESULTS_REPO=$MR -e SASREC_JOB=explain-run+--stage+redraw+--instance-set+redraw+--per-stratum+33+--seeds+0+1+2+--methods+nsga2+nsga2_seeded+greedy+ablation+scalar_ga+--pop-size+25,exact-front+--length+16+--methods+nsga2+random+greedy+moead+nsga3+smsemoa"
