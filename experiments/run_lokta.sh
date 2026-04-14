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
for mc_id in {0..19}
do
  python ./lokta/others.py --mc_id=$mc_id --r="gumbel" --tau=0.1 &
  python ./lokta/others.py --mc_id=$mc_id --r="gumbel" --tau=0.3 &
  python ./lokta/others.py --mc_id=$mc_id --r="gumbel" --tau=0.5 &
  python ./lokta/others.py --mc_id=$mc_id --r="soft" --alpha=0.5 &
  python ./lokta/others.py --mc_id=$mc_id --r="soft" --alpha=0.7 &
  python ./lokta/others.py --mc_id=$mc_id --r="soft" --alpha=0.9
  python ./lokta/others.py --mc_id=$mc_id --r="ot" --eps=0.5 &
  python ./lokta/others.py --mc_id=$mc_id --r="ot" --eps=1.0 &
  python ./lokta/others.py --mc_id=$mc_id --r="ot" --eps=1.5
done

# Diffusion
for mc_id in {0..19}
do
  for T in 1. 2.
  do
    for dsteps in 4 8 16
    do
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-1. --T=$T --dsteps=$dsteps --integrator='euler' &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-1. --T=$T --dsteps=$dsteps --integrator='euler' --sde &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-1. --T=$T --dsteps=$dsteps --integrator='lord_and_rougemont' &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-1. --T=$T --dsteps=$dsteps --integrator='lord_and_rougemont' --sde &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-1. --T=$T --dsteps=$dsteps --integrator='jentzen_and_kloeden' &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-1. --T=$T --dsteps=$dsteps --integrator='jentzen_and_kloeden' --sde &
      python ./lokta/diffusion.py --mc_id=$mc_id --a=-1. --T=$T --dsteps=$dsteps --integrator='tweedie' --sde
    done
  done
done