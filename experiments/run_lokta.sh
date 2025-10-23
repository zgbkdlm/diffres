#!/bin/bash

XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=.30

if [[ $ON_SERVER == "yes" ]]; then
  cd $WRKDIR/diffres
  source .venv/bin/activate
  cd experiments
fi

mkdir -p ./lokta/results

# Reference methods
for mc_id in {0..9}
do
  python ./lokta/gumbel.py --mc_id=$mc_id --tau=0.2 &
  python ./lokta/gumbel.py --mc_id=$mc_id --tau=0.4 &
  python ./lokta/gumbel.py --mc_id=$mc_id --tau=0.6 &
  python ./lokta/soft.py --mc_id=$mc_id --alpha=0. &
  python ./lokta/soft.py --mc_id=$mc_id --alpha=0.5 &
  python ./lokta/soft.py --mc_id=$mc_id --alpha=0.7 &
  python ./lokta/soft.py --mc_id=$mc_id --alpha=0.9
  python ./lokta/ot.py --mc_id=$mc_id --eps=0.5 &
  python ./lokta/ot.py --mc_id=$mc_id --eps=1.0 &
  python ./lokta/ot.py --mc_id=$mc_id --eps=1.5
done

# Diffusion
for mc_id in {0..9}
do
  for T in 1. 2.
  do
    for dsteps in 4 8 16
    do
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-0.5 --T=$T --dsteps=$dsteps --integrator='euler' &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-0.5 --T=$T --dsteps=$dsteps --integrator='euler' --sde &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-0.5 --T=$T --dsteps=$dsteps --integrator='lord_and_rougemont' &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-0.5 --T=$T --dsteps=$dsteps --integrator='lord_and_rougemont' --sde &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-0.5 --T=$T --dsteps=$dsteps --integrator='jentzen_and_kloeden' &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-0.5 --T=$T --dsteps=$dsteps --integrator='jentzen_and_kloeden' --sde &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-0.5 --T=$T --dsteps=$dsteps --integrator='tweedie' --sde
    done
  done
done