#!/bin/bash

XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=.10

if [[ $ON_SERVER == "yes" ]]; then
  cd $WRKDIR/diffres
  source .venv/bin/activate
  cd experiments
fi

mkdir -p ./lgssm/results

# Reference methods
python ./lgssm/baselines.py --id_l=0 --id_u=99 --method=multinomial &
python ./lgssm/gumbel.py --id_l=0 --id_u=99 --tau=0.1 &
python ./lgssm/gumbel.py --id_l=0 --id_u=99 --tau=0.3 &
python ./lgssm/gumbel.py --id_l=0 --id_u=99 --tau=0.5 &
python ./lgssm/soft.py --id_l=0 --id_u=99 --alpha=0.5 &
python ./lgssm/soft.py --id_l=0 --id_u=99 --alpha=0.7 &
python ./lgssm/soft.py --id_l=0 --id_u=99 --alpha=0.9
python ./lgssm/ot.py --id_l=0 --id_u=99 --eps=0.4 &
python ./lgssm/ot.py --id_l=0 --id_u=99 --eps=0.8 &
python ./lgssm/ot.py --id_l=0 --id_u=99 --eps=1.6

# Diffusion
for T in 1. 2. 3.
do
  for dsteps in 4 8 16 32
  do
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-1. --T=$T --dsteps=$dsteps --integrator='euler' &
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-1. --T=$T --dsteps=$dsteps --integrator='euler' --sde
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-1. --T=$T --dsteps=$dsteps --integrator='lord_and_rougemont' &
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-1. --T=$T --dsteps=$dsteps --integrator='lord_and_rougemont' --sde
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-1. --T=$T --dsteps=$dsteps --integrator='jentzen_and_kloeden' &
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-1. --T=$T --dsteps=$dsteps --integrator='jentzen_and_kloeden' --sde &
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-1. --T=$T --dsteps=$dsteps --integrator='tweedie' --sde
  done
done
