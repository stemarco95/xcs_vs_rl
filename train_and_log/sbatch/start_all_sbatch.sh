#!/usr/bin/env bash

sh start_all_deep_sarsa_sbatch.sh
sleep .5
sh start_all_dqn_sbatch.sh
sleep .5
sh start_all_xcs_sbatch.sh
sleep .5
sh start_all_q_learning_sbatch.sh
sleep .5
sh start_all_sarsa_sbatch.sh