#!/bin/bash

XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=.10

if [[ $ON_SERVER == "yes" ]]; then
  cd $WRKDIR/diffres
  source .venv/bin/activate
  cd experiments
fi

mkdir -p ./gms/results

# Reference methods
python ./gms/baselines.py --id_l=0 --id_u=99 --method=multinomial &
python ./gms/gumbel.py --id_l=0 --id_u=99 --tau=0.2 &
python ./gms/gumbel.py --id_l=0 --id_u=99 --tau=0.4 &
python ./gms/gumbel.py --id_l=0 --id_u=99 --tau=0.8 &
python ./gms/soft.py --id_l=0 --id_u=99 --alpha=0. &
python ./gms/soft.py --id_l=0 --id_u=99 --alpha=0.2 &
python ./gms/soft.py --id_l=0 --id_u=99 --alpha=0.4 &
python ./gms/soft.py --id_l=0 --id_u=99 --alpha=0.8
python ./gms/ot.py --id_l=0 --id_u=99 --eps=0.3 &
python ./gms/ot.py --id_l=0 --id_u=99 --eps=0.8

# Diffusion
for T in 3. 4. 5.
do
  for nsteps in 8 32 128
  do
    python ./gms/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='euler' &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='euler' --sde &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='lord_and_rougemont' &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='lord_and_rougemont' --sde &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='jentzen_and_kloeden' &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='jentzen_and_kloeden' --sde &
    python ./gms/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='tweedie' --sde
  done
done
