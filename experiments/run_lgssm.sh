#!/bin/bash

XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=.10

if [[ $ON_SERVER == "yes" ]]; then
  cd $WRKDIR/diffres/experiments
fi

mkdir -p ./lgssm/results

# Reference methods
python ./lgssm/baselines.py --id_l=0 --id_u=99 --method=multinomial &
python ./lgssm/gumbel.py --id_l=0 --id_u=99 --tau=0.2 &
python ./lgssm/gumbel.py --id_l=0 --id_u=99 --tau=0.4 &
python ./lgssm/gumbel.py --id_l=0 --id_u=99 --tau=0.8 &
python ./lgssm/soft.py --id_l=0 --id_u=99 --alpha=0.2 &
python ./lgssm/soft.py --id_l=0 --id_u=99 --alpha=0.4 &
python ./lgssm/soft.py --id_l=0 --id_u=99 --alpha=0.8
python ./lgssm/ot.py --id_l=0 --id_u=99 --eps=0.4 &
python ./lgssm/ot.py --id_l=0 --id_u=99 --eps=0.8 &
python ./lgssm/ot.py --id_l=0 --id_u=99 --eps=1.6

# Diffusion
for T in 3. 4. 5.
do
  for nsteps in 8 32 128
  do
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='euler'
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='euler' --sde
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='lord_and_rougemont'
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='lord_and_rougemont' --sde
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='jentzen_and_kloeden'
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='jentzen_and_kloeden' --sde
    python ./lgssm/diffusion.py --id_l=0 --id_u=99 --a=-0.5 --T=$T --nsteps=$nsteps --integrator='tweedie' --sde
  done
done
