#!/bin/bash

XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ $ON_SERVER == "yes" ]]; then
  cd $WRKDIR/diffres
  source .venv/bin/activate
  cd experiments
fi

mkdir -p ./cifar10/checkpoints

mc_id=$1
method=$2

python ./cifar10/run/py --mc_id=$mc_id --r=$method