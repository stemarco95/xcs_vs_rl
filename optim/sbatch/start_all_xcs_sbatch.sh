#!/usr/bin/env bash

sbatch xcs/optim_taxi.sl
sleep .5
sbatch xcs/optim_cartpole.sl
sleep .5
