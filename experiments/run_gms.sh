#!/bin/bash


mkdir -p ./gms/results


for mc_id in {0..99};
do
  python ./gms/baselines.py --mc_id=$mc_id --method=multinomial
  python ./gms/gumbel.py --mc_id=$mc_id --tau=0.2
  python ./gms/gumbel.py --mc_id=$mc_id --tau=0.4
  python ./gms/gumbel.py --mc_id=$mc_id --tau=0.8
  python ./gms/soft.py --mc_id=$mc_id --alpha=0.2
  python ./gms/soft.py --mc_id=$mc_id --alpha=0.4
  python ./gms/soft.py --mc_id=$mc_id --alpha=0.8
  python ./gms/diffusion.py --mc_id=$mc_id --a=-0.5 --T=3. --nsteps=8 --integrator='euler'
  python ./gms/diffusion.py --mc_id=$mc_id --a=-0.5 --T=3. --nsteps=32 --integrator='euler'
  python ./gms/diffusion.py --mc_id=$mc_id --a=-0.5 --T=3. --nsteps=64 --integrator='euler'
  python ./gms/ot.py --mc_id=$mc_id --eps=0.2
  python ./gms/ot.py --mc_id=$mc_id --eps=0.8
  python ./gms/ot.py --mc_id=$mc_id --eps=1.
done
