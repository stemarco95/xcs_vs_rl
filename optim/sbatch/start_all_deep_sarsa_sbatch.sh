#!/usr/bin/env bash

sbatch deep_sarsa/optim_taxi.sl
sleep .5
sbatch deep_sarsa/optim_cartpole.sl
sleep .5
sbatch deep_sarsa/optim_cliffwalking.sl
sleep .5
