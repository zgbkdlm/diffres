#!/bin/bash

if [[ $ON_SERVER == "yes" ]]; then
  cd $WRKDIR/diffres
  source .venv/bin/activate
  cd experiments
fi

mkdir -p ./profiling_time/results

python ./profiling_time/compare_time.py --integrator='jentzen_and_kloeden'
