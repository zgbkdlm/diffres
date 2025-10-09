#!/bin/bash


mkdir -p ./gms/results


for mc_id in {0..999};
do
  python ./gms/baselines.py --mc_id=$mc_id --method=multinomial
done &

for mc_id in {0..999};
do
  python ./gms/baselines.py --mc_id=$mc_id --method=systematic
done &

for mc_id in {0..999};
do
  python ./gms/baselines.py --mc_id=$mc_id --method=stratified
done
