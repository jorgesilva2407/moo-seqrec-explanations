#!/bin/bash
# Round 2, step 4: tuned MOEAs on main and on the redrawn set (ML-20M), and the Zvuk headline
# comparison in three shards. usage: OFFERS="<7 offer ids>" bash 04-...sh
# Chosen on the held-out users by total HV: MOEA/D Tchebicheff 5 partitions, NSGA-III 6, SMS-EMOA pop 50.
# Zvuk shard 2 lost its machine after its headline stage; its SMS-EMOA stage was re-run with the same
# z2exp job string restricted to that stage and --shard 2/3 (results resume from what was uploaded).
H=348e439b735c447748415384361e4873
MR=AbyssWanderer/moo-seqrec-explanations-ml20m; ZR=AbyssWanderer/moo-seqrec-explanations-zvuk; ZM=AbyssWanderer/sasrec-zvuk
launch() { vastai create instance "$1" --template_hash $H --disk 40 --label "$2" --env "$3" 2>&1 | tail -1; }
read -r -a O <<< "$OFFERS"
launch ${O[0]} m4moead  "-e SASREC_LABEL=m4moead -e SASREC_RESULTS_REPO=$MR -e SASREC_JOB=explain-run+--stage+main+--methods+moead+--decomposition+tchebicheff+--partitions+5+--pop-size+25"
launch ${O[1]} m5nsga3  "-e SASREC_LABEL=m5nsga3 -e SASREC_RESULTS_REPO=$MR -e SASREC_JOB=explain-run+--stage+main+--methods+nsga3+--partitions+6+--pop-size+25"
launch ${O[2]} m6sms    "-e SASREC_LABEL=m6sms -e SASREC_RESULTS_REPO=$MR -e SASREC_JOB=explain-run+--stage+main+--methods+smsemoa+--pop-size+50"
launch ${O[3]} m7redraw "-e SASREC_LABEL=m7redraw -e SASREC_RESULTS_REPO=$MR -e SASREC_JOB=explain-run+--stage+redraw-moea+--instance-set+redraw+--per-stratum+33+--seeds+0+1+2+--methods+smsemoa+--pop-size+50,explain-run+--stage+redraw-moea+--instance-set+redraw+--per-stratum+33+--seeds+0+1+2+--methods+moead+--decomposition+tchebicheff+--partitions+5+--pop-size+25"
Z="--processed-name+processed_zvuk+--hub-repo+$ZM+--hub-file+final/zvuk-h256-l3-d0.1/best.pt"
for s in 0 1 2; do
  bench=""; [ $s = 0 ] && bench="explain-bench+$Z+--per-stratum+5+--results-repo+$ZR,"
  job="fetch-processed+processed_zvuk+$ZR,${bench}explain-run+--stage+main+$Z+--per-stratum+15+--main-per-stratum+15+--seeds+0+1+2+--methods+nsga2+nsga2_seeded+greedy+ablation+scalar_ga+--pop-size+25+--shard+$s/3,explain-run+--stage+main-sms+$Z+--per-stratum+15+--main-per-stratum+15+--seeds+0+1+2+--methods+smsemoa+--pop-size+50+--shard+$s/3"
  launch ${O[$((4+s))]} "z2exp-$s" "-e SASREC_LABEL=z2exp-$s -e SASREC_PREPARE=none -e SASREC_RESULTS_REPO=$ZR -e SASREC_NEEDS=dataset:${ZR}:write,model:${ZM}:read -e SASREC_JOB=$job"
done

# Check 3 (robustness to the choice of Zvuk model): the same comparison with the second-best model,
# on 15 instances, launched as label z3alt:
#   Z=--processed-name+processed_zvuk+--hub-repo+$ZM+--hub-file+final/zvuk-h256-l3-d0.2/best.pt
#   SASREC_JOB=fetch-processed+processed_zvuk+$ZR,explain-run+--stage+alt-model+$Z+--per-stratum+5+--main-per-stratum+5+--seeds+0+1+2+--methods+nsga2+nsga2_seeded+greedy+ablation+scalar_ga+--pop-size+25,explain-run+--stage+alt-model-sms+$Z+--per-stratum+5+--main-per-stratum+5+--seeds+0+1+2+--methods+smsemoa+--pop-size+50
