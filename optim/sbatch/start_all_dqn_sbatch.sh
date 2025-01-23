#!/usr/bin/env bash

sbatch dqn/optim_taxi.sl
sleep .5
sbatch dqn/optim_cartpole.sl
sleep .5
