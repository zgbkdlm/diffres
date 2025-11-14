#!/bin/bash

XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=.10

if [[ $ON_SERVER == "yes" ]]; then
  cd $WRKDIR/diffres
  source .venv/bin/activate
  cd experiments
fi

mkdir -p ./gms/results

nsamples=10000

# Reference methods
python ./gms/baselines.py --id_l=0 --id_u=99 --method=multinomial --nsamples=$nsamples &
python ./gms/gumbel.py --id_l=0 --id_u=99 --tau=0.2 --nsamples=$nsamples &
python ./gms/gumbel.py --id_l=0 --id_u=99 --tau=0.4 --nsamples=$nsamples &
python ./gms/gumbel.py --id_l=0 --id_u=99 --tau=0.8 --nsamples=$nsamples &
python ./gms/soft.py --id_l=0 --id_u=99 --alpha=0. --nsamples=$nsamples &
python ./gms/soft.py --id_l=0 --id_u=99 --alpha=0.2 --nsamples=$nsamples &
python ./gms/soft.py --id_l=0 --id_u=99 --alpha=0.4 --nsamples=$nsamples &
python ./gms/soft.py --id_l=0 --id_u=99 --alpha=0.8 --nsamples=$nsamples
python ./gms/ot.py --id_l=0 --id_u=99 --eps=0.3 --nsamples=$nsamples &
python ./gms/ot.py --id_l=0 --id_u=99 --eps=0.6 --nsamples=$nsamples &
python ./gms/ot.py --id_l=0 --id_u=99 --eps=0.8 --nsamples=$nsamples &
python ./gms/ot.py --id_l=0 --id_u=99 --eps=0.9 --nsamples=$nsamples

# Diffusion
for T in 1. 2. 3.
do
  for nsteps in 8 32 128
  do
    python ./gms/diffusion.py --id_l=0 --id_u=99 --nsamples=$nsamples --a=-1. --T=$T --nsteps=$nsteps --integrator='euler' &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --nsamples=$nsamples --a=-1. --T=$T --nsteps=$nsteps --integrator='euler' --sde &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --nsamples=$nsamples --a=-1. --T=$T --nsteps=$nsteps --integrator='lord_and_rougemont' &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --nsamples=$nsamples --a=-1. --T=$T --nsteps=$nsteps --integrator='lord_and_rougemont' --sde &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --nsamples=$nsamples --a=-1. --T=$T --nsteps=$nsteps --integrator='jentzen_and_kloeden' &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --nsamples=$nsamples --a=-1. --T=$T --nsteps=$nsteps --integrator='jentzen_and_kloeden' --sde &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --nsamples=$nsamples --a=-1. --T=$T --nsteps=$nsteps --integrator='tweedie' --sde
  done
done
