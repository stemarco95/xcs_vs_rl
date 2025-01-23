# Evaluating Adaptive Systems: A Comparative Study of XCS and Advanced RL Methods in Noisy Multi-Step Environments

Code for the paper: **Evaluating Adaptive Systems: A Comparative Study of XCS and Advanced RL Methods in Noisy Multi-Step Environments**

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Optimization](#optimization)
  - [Training](#training)
- [Results](#results)
  - [Optimization Results](#optimization-results)
  - [Training Results](#training-results)

---

## Installation

To set up the environment, clone the repository:

```bash
git clone https://github.com/stemarco95/xcs_vs_rl.git
cd xcs_vs_rl
```

---

## Usage

### Optimization

To start the parameter optimization process:

1. Navigate to the `optim/` directory.

2. Use the `main.py` script, providing a configuration string from `optim/sbatch/configs.py`. These configurations are defined as strings in Python dictionaries. You only need to copy one string per run.

3. Example command to run the optimization:

```bash
python3 optim/main.py "{\"seed\": 21, \"agent\":{\"type\": \"deep_sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]}, {\"name\": \"epsilon\", \"values\": [0.20, 0.25, 0.30, 0.35, 0.40]}, {\"name\": \"gamma\", \"values\": [0.85, 0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"blackjack\", \"parameter\": {\"iterations\": 1000, \"natural\": true, \"det_prob_state\": 1.00}}}"
```

4. Results will be saved in the `optim/results` directory as JSON files. The files are organized by the structure:
   - `agent/instance/benchmark.json`
     - `agent`: Specifies the RL agent (e.g., DQN).
     - `instance`: Describes the problem instance (e.g., state noise and action noise probabilities).
     - `benchmark.json`: Contains the benchmark results (e.g., cartpole.json).

### Training

After generating optimized parameters, use them to create training configurations:

1. Run the script `train_and_log/sbatch/create_configs.py`. This script will generate training configuration files based on the optimization results.

2. The generated configuration files will be stored in:

   - `train_and_log/sbatch/configs/agent/benchmark.json`
     - These are structured similarly to the optimization results.
     - Note: Some benchmarks may require splitting into multiple files due to their complexity.

3. Start the training process by running the `main.py` script in the `train_and_log/` directory, providing the path to a specific configuration file:

```bash
python3 train_and_log/main.py "train_and_log/sbatch/configs/deep_sarsa/blackjack.json"
```

---

## Results

The results of both optimization and training can be found in their respective directories:

#### Optimization Results

- Located in `optim/results`.
- Organized as `agent/instance/benchmark.json`, where:
  - `agent`: Specifies the RL agent (e.g., DQN).
  - `instance`: Details the problem instance (e.g., state and action noise probabilities).
  - `benchmark.json`: Includes benchmark results (e.g., cartpole.json).

| Agent | Benchmark | State Noise Prob. | Action Noise Prob. / Natural | Alpha | Beta | Epsilon | Gamma | Population Size | E0 |
|-------|-----------|----------------|-------------------------|-------|------|---------|-------|----------------|----|
| deep_sarsa | blackjack | 1.0 | True | 0.001 | N/A | 0.35 | 0.9 | N/A | N/A |
| deep_sarsa | blackjack | 1.0 | False | 0.005 | N/A | 0.2 | 0.95 | N/A | N/A |
| deep_sarsa | blackjack | 0.95 | True | 0.0005 | N/A | 0.2 | 0.95 | N/A | N/A |
| deep_sarsa | blackjack | 0.95 | False | 0.01 | N/A | 0.25 | 0.9 | N/A | N/A |
| deep_sarsa | blackjack | 0.9 | True | 0.01 | N/A | 0.35 | 0.9 | N/A | N/A |
| deep_sarsa | blackjack | 0.9 | False | 0.005 | N/A | 0.35 | 0.99 | N/A | N/A |
| deep_sarsa | cartpole | 1.0 | 1.0 | 0.001 | N/A | 0.25 | 0.99 | N/A | N/A |
| deep_sarsa | cartpole | 1.0 | 0.95 | 0.001 | N/A | 0.3 | 0.95 | N/A | N/A |
| deep_sarsa | cartpole | 1.0 | 0.9 | 0.001 | N/A | 0.4 | 0.99 | N/A | N/A |
| deep_sarsa | cartpole | 0.95 | 1.0 | 0.001 | N/A | 0.35 | 0.99 | N/A | N/A |
| deep_sarsa | cartpole | 0.95 | 0.95 | 0.01 | N/A | 0.25 | 0.99 | N/A | N/A |
| deep_sarsa | cartpole | 0.95 | 0.9 | 0.0001 | N/A | 0.35 | 0.95 | N/A | N/A |
| deep_sarsa | cartpole | 0.9 | 1.0 | 0.0005 | N/A | 0.2 | 0.99 | N/A | N/A |
| deep_sarsa | cartpole | 0.9 | 0.95 | 0.005 | N/A | 0.35 | 0.99 | N/A | N/A |
| deep_sarsa | cartpole | 0.9 | 0.9 | 0.001 | N/A | 0.25 | 0.99 | N/A | N/A |
| deep_sarsa | cliffwalking | 1.0 | 1.0 | 0.001 | N/A | 0.35 | 0.99 | N/A | N/A |
| deep_sarsa | cliffwalking | 1.0 | 0.95 | 0.005 | N/A | 0.25 | 0.99 | N/A | N/A |
| deep_sarsa | cliffwalking | 1.0 | 0.9 | 0.01 | N/A | 0.25 | 0.99 | N/A | N/A |
| deep_sarsa | cliffwalking | 0.95 | 1.0 | 0.001 | N/A | 0.3 | 0.95 | N/A | N/A |
| deep_sarsa | cliffwalking | 0.95 | 0.95 | 0.01 | N/A | 0.2 | 0.99 | N/A | N/A |
| deep_sarsa | cliffwalking | 0.95 | 0.9 | 0.0001 | N/A | 0.2 | 0.99 | N/A | N/A |
| deep_sarsa | cliffwalking | 0.9 | 1.0 | 0.005 | N/A | 0.3 | 0.99 | N/A | N/A |
| deep_sarsa | cliffwalking | 0.9 | 0.95 | 0.001 | N/A | 0.2 | 0.99 | N/A | N/A |
| deep_sarsa | cliffwalking | 0.9 | 0.9 | 0.01 | N/A | 0.35 | 0.99 | N/A | N/A |
| deep_sarsa | frozenlake4x4 | 1.0 | 1.0 | 0.005 | N/A | 0.4 | 0.9 | N/A | N/A |
| deep_sarsa | frozenlake4x4 | 1.0 | 0.95 | 0.01 | N/A | 0.3 | 0.95 | N/A | N/A |
| deep_sarsa | frozenlake4x4 | 1.0 | 0.9 | 0.0001 | N/A | 0.35 | 0.9 | N/A | N/A |
| deep_sarsa | frozenlake4x4 | 0.95 | 1.0 | 0.01 | N/A | 0.2 | 0.9 | N/A | N/A |
| deep_sarsa | frozenlake4x4 | 0.95 | 0.95 | 0.01 | N/A | 0.2 | 0.85 | N/A | N/A |
| deep_sarsa | frozenlake4x4 | 0.95 | 0.9 | 0.0005 | N/A | 0.3 | 0.9 | N/A | N/A |
| deep_sarsa | frozenlake4x4 | 0.9 | 1.0 | 0.01 | N/A | 0.3 | 0.9 | N/A | N/A |
| deep_sarsa | frozenlake4x4 | 0.9 | 0.95 | 0.001 | N/A | 0.3 | 0.85 | N/A | N/A |
| deep_sarsa | frozenlake4x4 | 0.9 | 0.9 | 0.0005 | N/A | 0.4 | 0.9 | N/A | N/A |
| deep_sarsa | frozenlake8x8 | 1.0 | 1.0 | 0.0001 | N/A | 0.3 | 0.9 | N/A | N/A |
| deep_sarsa | frozenlake8x8 | 1.0 | 0.95 | 0.001 | N/A | 0.3 | 0.9 | N/A | N/A |
| deep_sarsa | frozenlake8x8 | 1.0 | 0.9 | 0.0001 | N/A | 0.3 | 0.95 | N/A | N/A |
| deep_sarsa | frozenlake8x8 | 0.95 | 1.0 | 0.001 | N/A | 0.2 | 0.85 | N/A | N/A |
| deep_sarsa | frozenlake8x8 | 0.95 | 0.95 | 0.005 | N/A | 0.3 | 0.9 | N/A | N/A |
| deep_sarsa | frozenlake8x8 | 0.95 | 0.9 | 0.001 | N/A | 0.35 | 0.9 | N/A | N/A |
| deep_sarsa | frozenlake8x8 | 0.9 | 1.0 | 0.001 | N/A | 0.25 | 0.95 | N/A | N/A |
| deep_sarsa | frozenlake8x8 | 0.9 | 0.95 | 0.0001 | N/A | 0.2 | 0.95 | N/A | N/A |
| deep_sarsa | frozenlake8x8 | 0.9 | 0.9 | 0.001 | N/A | 0.35 | 0.9 | N/A | N/A |
| deep_sarsa | taxi | 1.0 | 1.0 | 0.0005 | N/A | 0.3 | 0.99 | N/A | N/A |
| deep_sarsa | taxi | 1.0 | 0.95 | 0.001 | N/A | 0.3 | 0.99 | N/A | N/A |
| deep_sarsa | taxi | 1.0 | 0.9 | 0.001 | N/A | 0.4 | 0.9 | N/A | N/A |
| deep_sarsa | taxi | 0.95 | 1.0 | 0.001 | N/A | 0.35 | 0.95 | N/A | N/A |
| deep_sarsa | taxi | 0.95 | 0.95 | 0.001 | N/A | 0.2 | 0.99 | N/A | N/A |
| deep_sarsa | taxi | 0.95 | 0.9 | 0.001 | N/A | 0.4 | 0.95 | N/A | N/A |
| deep_sarsa | taxi | 0.9 | 1.0 | 0.001 | N/A | 0.2 | 0.95 | N/A | N/A |
| deep_sarsa | taxi | 0.9 | 0.95 | 0.001 | N/A | 0.35 | 0.95 | N/A | N/A |
| deep_sarsa | taxi | 0.9 | 0.9 | 0.001 | N/A | 0.35 | 0.95 | N/A | N/A |
| dqn | blackjack | 1.0 | True | 0.0005 | N/A | 0.3 | 0.99 | N/A | N/A |
| dqn | blackjack | 1.0 | False | 0.01 | N/A | 0.2 | 0.9 | N/A | N/A |
| dqn | blackjack | 0.95 | True | 0.0005 | N/A | 0.4 | 0.9 | N/A | N/A |
| dqn | blackjack | 0.95 | False | 0.01 | N/A | 0.3 | 0.85 | N/A | N/A |
| dqn | blackjack | 0.9 | True | 0.05 | N/A | 0.35 | 0.95 | N/A | N/A |
| dqn | blackjack | 0.9 | False | 0.0001 | N/A | 0.3 | 0.99 | N/A | N/A |
| dqn | cartpole | 1.0 | 1.0 | 0.0005 | N/A | 0.4 | 0.95 | N/A | N/A |
| dqn | cartpole | 1.0 | 0.95 | 0.0005 | N/A | 0.35 | 0.99 | N/A | N/A |
| dqn | cartpole | 1.0 | 0.9 | 0.0005 | N/A | 0.3 | 0.99 | N/A | N/A |
| dqn | cartpole | 0.95 | 1.0 | 0.001 | N/A | 0.3 | 0.99 | N/A | N/A |
| dqn | cartpole | 0.95 | 0.95 | 0.01 | N/A | 0.35 | 0.85 | N/A | N/A |
| dqn | cartpole | 0.95 | 0.9 | 0.0001 | N/A | 0.25 | 0.99 | N/A | N/A |
| dqn | cartpole | 0.9 | 1.0 | 0.005 | N/A | 0.4 | 0.99 | N/A | N/A |
| dqn | cartpole | 0.9 | 0.95 | 0.005 | N/A | 0.2 | 0.95 | N/A | N/A |
| dqn | cartpole | 0.9 | 0.9 | 0.0001 | N/A | 0.25 | 0.85 | N/A | N/A |
| dqn | cliffwalking | 1.0 | 1.0 | 0.01 | N/A | 0.3 | 0.99 | N/A | N/A |
| dqn | cliffwalking | 1.0 | 0.95 | 0.005 | N/A | 0.4 | 0.9 | N/A | N/A |
| dqn | cliffwalking | 1.0 | 0.9 | 0.001 | N/A | 0.35 | 0.95 | N/A | N/A |
| dqn | cliffwalking | 0.95 | 1.0 | 0.01 | N/A | 0.25 | 0.95 | N/A | N/A |
| dqn | cliffwalking | 0.95 | 0.95 | 0.0001 | N/A | 0.35 | 0.95 | N/A | N/A |
| dqn | cliffwalking | 0.95 | 0.9 | 0.001 | N/A | 0.35 | 0.99 | N/A | N/A |
| dqn | cliffwalking | 0.9 | 1.0 | 0.005 | N/A | 0.3 | 0.99 | N/A | N/A |
| dqn | cliffwalking | 0.9 | 0.95 | 0.0001 | N/A | 0.2 | 0.99 | N/A | N/A |
| dqn | cliffwalking | 0.9 | 0.9 | 0.0005 | N/A | 0.2 | 0.99 | N/A | N/A |
| dqn | frozenlake4x4 | 1.0 | 1.0 | 0.005 | N/A | 0.3 | 0.85 | N/A | N/A |
| dqn | frozenlake4x4 | 1.0 | 0.95 | 0.001 | N/A | 0.2 | 0.85 | N/A | N/A |
| dqn | frozenlake4x4 | 1.0 | 0.9 | 0.001 | N/A | 0.2 | 0.95 | N/A | N/A |
| dqn | frozenlake4x4 | 0.95 | 1.0 | 0.01 | N/A | 0.3 | 0.85 | N/A | N/A |
| dqn | frozenlake4x4 | 0.95 | 0.95 | 0.0001 | N/A | 0.2 | 0.9 | N/A | N/A |
| dqn | frozenlake4x4 | 0.95 | 0.9 | 0.005 | N/A | 0.25 | 0.9 | N/A | N/A |
| dqn | frozenlake4x4 | 0.9 | 1.0 | 0.01 | N/A | 0.25 | 0.9 | N/A | N/A |
| dqn | frozenlake4x4 | 0.9 | 0.95 | 0.0005 | N/A | 0.3 | 0.95 | N/A | N/A |
| dqn | frozenlake4x4 | 0.9 | 0.9 | 0.001 | N/A | 0.25 | 0.95 | N/A | N/A |
| dqn | frozenlake8x8 | 1.0 | 1.0 | 0.0001 | N/A | 0.25 | 0.95 | N/A | N/A |
| dqn | frozenlake8x8 | 1.0 | 0.95 | 0.0005 | N/A | 0.35 | 0.95 | N/A | N/A |
| dqn | frozenlake8x8 | 1.0 | 0.9 | 0.0005 | N/A | 0.3 | 0.99 | N/A | N/A |
| dqn | frozenlake8x8 | 0.95 | 1.0 | 0.0005 | N/A | 0.4 | 0.95 | N/A | N/A |
| dqn | frozenlake8x8 | 0.95 | 0.95 | 0.0005 | N/A | 0.25 | 0.85 | N/A | N/A |
| dqn | frozenlake8x8 | 0.95 | 0.9 | 0.001 | N/A | 0.2 | 0.95 | N/A | N/A |
| dqn | frozenlake8x8 | 0.9 | 1.0 | 0.0001 | N/A | 0.2 | 0.9 | N/A | N/A |
| dqn | frozenlake8x8 | 0.9 | 0.95 | 0.001 | N/A | 0.35 | 0.95 | N/A | N/A |
| dqn | frozenlake8x8 | 0.9 | 0.9 | 0.0001 | N/A | 0.35 | 0.95 | N/A | N/A |
| dqn | taxi | 1.0 | 1.0 | 0.001 | N/A | 0.2 | 0.95 | N/A | N/A |
| dqn | taxi | 1.0 | 0.95 | 0.005 | N/A | 0.3 | 0.95 | N/A | N/A |
| dqn | taxi | 1.0 | 0.9 | 0.005 | N/A | 0.4 | 0.95 | N/A | N/A |
| dqn | taxi | 0.95 | 1.0 | 0.0005 | N/A | 0.3 | 0.95 | N/A | N/A |
| dqn | taxi | 0.95 | 0.95 | 0.005 | N/A | 0.25 | 0.95 | N/A | N/A |
| dqn | taxi | 0.95 | 0.9 | 0.005 | N/A | 0.2 | 0.9 | N/A | N/A |
| dqn | taxi | 0.9 | 1.0 | 0.001 | N/A | 0.3 | 0.85 | N/A | N/A |
| dqn | taxi | 0.9 | 0.95 | 0.005 | N/A | 0.35 | 0.9 | N/A | N/A |
| dqn | taxi | 0.9 | 0.9 | 0.005 | N/A | 0.3 | 0.85 | N/A | N/A |
| q_learning | blackjack | 1.0 | True | 0.1 | N/A | 0.3 | 0.9 | N/A | N/A |
| q_learning | blackjack | 1.0 | False | 0.066 | N/A | 0.25 | 0.85 | N/A | N/A |
| q_learning | blackjack | 0.95 | True | 0.005 | N/A | 0.2 | 0.9 | N/A | N/A |
| q_learning | blackjack | 0.95 | False | 0.066 | N/A | 0.35 | 0.85 | N/A | N/A |
| q_learning | blackjack | 0.9 | True | 0.005 | N/A | 0.4 | 0.85 | N/A | N/A |
| q_learning | blackjack | 0.9 | False | 0.005 | N/A | 0.25 | 0.9 | N/A | N/A |
| q_learning | cartpole | 1.0 | 1.0 | 0.066 | N/A | 0.25 | 0.99 | N/A | N/A |
| q_learning | cartpole | 1.0 | 0.95 | 0.066 | N/A | 0.25 | 0.9 | N/A | N/A |
| q_learning | cartpole | 1.0 | 0.9 | 0.066 | N/A | 0.3 | 0.95 | N/A | N/A |
| q_learning | cartpole | 0.95 | 1.0 | 0.1 | N/A | 0.2 | 0.9 | N/A | N/A |
| q_learning | cartpole | 0.95 | 0.95 | 0.1 | N/A | 0.35 | 0.9 | N/A | N/A |
| q_learning | cartpole | 0.95 | 0.9 | 0.033 | N/A | 0.4 | 0.99 | N/A | N/A |
| q_learning | cartpole | 0.9 | 1.0 | 0.1 | N/A | 0.4 | 0.95 | N/A | N/A |
| q_learning | cartpole | 0.9 | 0.95 | 0.1 | N/A | 0.2 | 0.9 | N/A | N/A |
| q_learning | cartpole | 0.9 | 0.9 | 0.1 | N/A | 0.35 | 0.85 | N/A | N/A |
| q_learning | cliffwalking | 1.0 | 1.0 | 0.066 | N/A | 0.4 | 0.9 | N/A | N/A |
| q_learning | cliffwalking | 1.0 | 0.95 | 0.1 | N/A | 0.3 | 0.9 | N/A | N/A |
| q_learning | cliffwalking | 1.0 | 0.9 | 0.066 | N/A | 0.35 | 0.85 | N/A | N/A |
| q_learning | cliffwalking | 0.95 | 1.0 | 0.033 | N/A | 0.4 | 0.95 | N/A | N/A |
| q_learning | cliffwalking | 0.95 | 0.95 | 0.066 | N/A | 0.25 | 0.9 | N/A | N/A |
| q_learning | cliffwalking | 0.95 | 0.9 | 0.066 | N/A | 0.35 | 0.9 | N/A | N/A |
| q_learning | cliffwalking | 0.9 | 1.0 | 0.033 | N/A | 0.3 | 0.85 | N/A | N/A |
| q_learning | cliffwalking | 0.9 | 0.95 | 0.1 | N/A | 0.25 | 0.9 | N/A | N/A |
| q_learning | cliffwalking | 0.9 | 0.9 | 0.1 | N/A | 0.4 | 0.95 | N/A | N/A |
| q_learning | frozenlake4x4 | 1.0 | 1.0 | 0.033 | N/A | 0.35 | 0.9 | N/A | N/A |
| q_learning | frozenlake4x4 | 1.0 | 0.95 | 0.005 | N/A | 0.25 | 0.85 | N/A | N/A |
| q_learning | frozenlake4x4 | 1.0 | 0.9 | 0.033 | N/A | 0.3 | 0.95 | N/A | N/A |
| q_learning | frozenlake4x4 | 0.95 | 1.0 | 0.066 | N/A | 0.35 | 0.99 | N/A | N/A |
| q_learning | frozenlake4x4 | 0.95 | 0.95 | 0.005 | N/A | 0.2 | 0.85 | N/A | N/A |
| q_learning | frozenlake4x4 | 0.95 | 0.9 | 0.033 | N/A | 0.35 | 0.85 | N/A | N/A |
| q_learning | frozenlake4x4 | 0.9 | 1.0 | 0.033 | N/A | 0.2 | 0.95 | N/A | N/A |
| q_learning | frozenlake4x4 | 0.9 | 0.95 | 0.033 | N/A | 0.4 | 0.85 | N/A | N/A |
| q_learning | frozenlake4x4 | 0.9 | 0.9 | 0.066 | N/A | 0.3 | 0.9 | N/A | N/A |
| q_learning | frozenlake8x8 | 1.0 | 1.0 | 0.066 | N/A | 0.4 | 0.95 | N/A | N/A |
| q_learning | frozenlake8x8 | 1.0 | 0.95 | 0.005 | N/A | 0.4 | 0.9 | N/A | N/A |
| q_learning | frozenlake8x8 | 1.0 | 0.9 | 0.01 | N/A | 0.3 | 0.95 | N/A | N/A |
| q_learning | frozenlake8x8 | 0.95 | 1.0 | 0.005 | N/A | 0.2 | 0.95 | N/A | N/A |
| q_learning | frozenlake8x8 | 0.95 | 0.95 | 0.033 | N/A | 0.4 | 0.99 | N/A | N/A |
| q_learning | frozenlake8x8 | 0.95 | 0.9 | 0.01 | N/A | 0.2 | 0.85 | N/A | N/A |
| q_learning | frozenlake8x8 | 0.9 | 1.0 | 0.005 | N/A | 0.4 | 0.85 | N/A | N/A |
| q_learning | frozenlake8x8 | 0.9 | 0.95 | 0.033 | N/A | 0.4 | 0.85 | N/A | N/A |
| q_learning | frozenlake8x8 | 0.9 | 0.9 | 0.066 | N/A | 0.25 | 0.95 | N/A | N/A |
| q_learning | taxi | 1.0 | 1.0 | 0.01 | N/A | 0.3 | 0.85 | N/A | N/A |
| q_learning | taxi | 1.0 | 0.95 | 0.1 | N/A | 0.35 | 0.99 | N/A | N/A |
| q_learning | taxi | 1.0 | 0.9 | 0.1 | N/A | 0.4 | 0.99 | N/A | N/A |
| q_learning | taxi | 0.95 | 1.0 | 0.1 | N/A | 0.3 | 0.99 | N/A | N/A |
| q_learning | taxi | 0.95 | 0.95 | 0.066 | N/A | 0.4 | 0.95 | N/A | N/A |
| q_learning | taxi | 0.95 | 0.9 | 0.1 | N/A | 0.2 | 0.85 | N/A | N/A |
| q_learning | taxi | 0.9 | 1.0 | 0.066 | N/A | 0.4 | 0.85 | N/A | N/A |
| q_learning | taxi | 0.9 | 0.95 | 0.066 | N/A | 0.35 | 0.85 | N/A | N/A |
| q_learning | taxi | 0.9 | 0.9 | 0.066 | N/A | 0.4 | 0.95 | N/A | N/A |
| sarsa | blackjack | 1.0 | True | 0.033 | N/A | 0.35 | 0.85 | N/A | N/A |
| sarsa | blackjack | 1.0 | False | 0.005 | N/A | 0.2 | 0.85 | N/A | N/A |
| sarsa | blackjack | 0.95 | True | 0.005 | N/A | 0.2 | 0.9 | N/A | N/A |
| sarsa | blackjack | 0.95 | False | 0.033 | N/A | 0.2 | 0.85 | N/A | N/A |
| sarsa | blackjack | 0.9 | True | 0.005 | N/A | 0.2 | 0.99 | N/A | N/A |
| sarsa | blackjack | 0.9 | False | 0.066 | N/A | 0.2 | 0.9 | N/A | N/A |
| sarsa | cartpole | 1.0 | 1.0 | 0.1 | N/A | 0.25 | 0.95 | N/A | N/A |
| sarsa | cartpole | 1.0 | 0.95 | 0.066 | N/A | 0.3 | 0.95 | N/A | N/A |
| sarsa | cartpole | 1.0 | 0.9 | 0.1 | N/A | 0.3 | 0.95 | N/A | N/A |
| sarsa | cartpole | 0.95 | 1.0 | 0.066 | N/A | 0.3 | 0.95 | N/A | N/A |
| sarsa | cartpole | 0.95 | 0.95 | 0.1 | N/A | 0.2 | 0.9 | N/A | N/A |
| sarsa | cartpole | 0.95 | 0.9 | 0.33 | N/A | 0.35 | 0.95 | N/A | N/A |
| sarsa | cartpole | 0.9 | 1.0 | 0.033 | N/A | 0.25 | 0.99 | N/A | N/A |
| sarsa | cartpole | 0.9 | 0.95 | 0.1 | N/A | 0.4 | 0.95 | N/A | N/A |
| sarsa | cartpole | 0.9 | 0.9 | 0.066 | N/A | 0.2 | 0.99 | N/A | N/A |
| sarsa | cliffwalking | 1.0 | 1.0 | 0.1 | N/A | 0.35 | 0.95 | N/A | N/A |
| sarsa | cliffwalking | 1.0 | 0.95 | 0.1 | N/A | 0.2 | 0.9 | N/A | N/A |
| sarsa | cliffwalking | 1.0 | 0.9 | 0.066 | N/A | 0.35 | 0.99 | N/A | N/A |
| sarsa | cliffwalking | 0.95 | 1.0 | 0.1 | N/A | 0.3 | 0.99 | N/A | N/A |
| sarsa | cliffwalking | 0.95 | 0.95 | 0.005 | N/A | 0.25 | 0.95 | N/A | N/A |
| sarsa | cliffwalking | 0.95 | 0.9 | 0.033 | N/A | 0.25 | 0.99 | N/A | N/A |
| sarsa | cliffwalking | 0.9 | 1.0 | 0.033 | N/A | 0.4 | 0.95 | N/A | N/A |
| sarsa | cliffwalking | 0.9 | 0.95 | 0.066 | N/A | 0.2 | 0.99 | N/A | N/A |
| sarsa | cliffwalking | 0.9 | 0.9 | 0.1 | N/A | 0.25 | 0.99 | N/A | N/A |
| sarsa | frozenlake4x4 | 1.0 | 1.0 | 0.033 | N/A | 0.3 | 0.85 | N/A | N/A |
| sarsa | frozenlake4x4 | 1.0 | 0.95 | 0.066 | N/A | 0.2 | 0.9 | N/A | N/A |
| sarsa | frozenlake4x4 | 1.0 | 0.9 | 0.066 | N/A | 0.25 | 0.9 | N/A | N/A |
| sarsa | frozenlake4x4 | 0.95 | 1.0 | 0.033 | N/A | 0.2 | 0.99 | N/A | N/A |
| sarsa | frozenlake4x4 | 0.95 | 0.95 | 0.066 | N/A | 0.25 | 0.9 | N/A | N/A |
| sarsa | frozenlake4x4 | 0.95 | 0.9 | 0.01 | N/A | 0.25 | 0.99 | N/A | N/A |
| sarsa | frozenlake4x4 | 0.9 | 1.0 | 0.066 | N/A | 0.25 | 0.85 | N/A | N/A |
| sarsa | frozenlake4x4 | 0.9 | 0.95 | 0.1 | N/A | 0.25 | 0.95 | N/A | N/A |
| sarsa | frozenlake4x4 | 0.9 | 0.9 | 0.066 | N/A | 0.35 | 0.99 | N/A | N/A |
| sarsa | frozenlake8x8 | 1.0 | 1.0 | 0.005 | N/A | 0.35 | 0.85 | N/A | N/A |
| sarsa | frozenlake8x8 | 1.0 | 0.95 | 0.1 | N/A | 0.35 | 0.99 | N/A | N/A |
| sarsa | frozenlake8x8 | 1.0 | 0.9 | 0.033 | N/A | 0.25 | 0.99 | N/A | N/A |
| sarsa | frozenlake8x8 | 0.95 | 1.0 | 0.033 | N/A | 0.3 | 0.85 | N/A | N/A |
| sarsa | frozenlake8x8 | 0.95 | 0.95 | 0.033 | N/A | 0.4 | 0.95 | N/A | N/A |
| sarsa | frozenlake8x8 | 0.95 | 0.9 | 0.033 | N/A | 0.3 | 0.9 | N/A | N/A |
| sarsa | frozenlake8x8 | 0.9 | 1.0 | 0.01 | N/A | 0.25 | 0.95 | N/A | N/A |
| sarsa | frozenlake8x8 | 0.9 | 0.95 | 0.005 | N/A | 0.3 | 0.99 | N/A | N/A |
| sarsa | frozenlake8x8 | 0.9 | 0.9 | 0.005 | N/A | 0.4 | 0.95 | N/A | N/A |
| sarsa | taxi | 1.0 | 1.0 | 0.1 | N/A | 0.25 | 0.95 | N/A | N/A |
| sarsa | taxi | 1.0 | 0.95 | 0.1 | N/A | 0.2 | 0.95 | N/A | N/A |
| sarsa | taxi | 1.0 | 0.9 | 0.1 | N/A | 0.3 | 0.95 | N/A | N/A |
| sarsa | taxi | 0.95 | 1.0 | 0.1 | N/A | 0.2 | 0.99 | N/A | N/A |
| sarsa | taxi | 0.95 | 0.95 | 0.1 | N/A | 0.2 | 0.95 | N/A | N/A |
| sarsa | taxi | 0.95 | 0.9 | 0.1 | N/A | 0.35 | 0.99 | N/A | N/A |
| sarsa | taxi | 0.9 | 1.0 | 0.1 | N/A | 0.2 | 0.95 | N/A | N/A |
| sarsa | taxi | 0.9 | 0.95 | 0.1 | N/A | 0.35 | 0.99 | N/A | N/A |
| sarsa | taxi | 0.9 | 0.9 | 0.1 | N/A | 0.4 | 0.99 | N/A | N/A |
| xcs | blackjack | 1.0 | True | N/A | 0.1 | 0.3 | 0.85 | 704 | 0.02 |
| xcs | blackjack | 1.0 | False | N/A | 0.1 | 0.4 | 0.9 | 1350 | 0.015 |
| xcs | blackjack | 0.95 | True | N/A | 0.1 | 0.25 | 0.85 | 1350 | 0.015 |
| xcs | blackjack | 0.95 | False | N/A | 0.2 | 0.3 | 0.99 | 1350 | 0.01 |
| xcs | blackjack | 0.9 | True | N/A | 0.35 | 0.2 | 0.85 | 1350 | 0.01 |
| xcs | blackjack | 0.9 | False | N/A | 0.1 | 0.4 | 0.85 | 2000 | 0.01 |
| xcs | cartpole | 1.0 | 1.0 | N/A | 0.1 | 0.3 | 0.99 | 10000 | 0.1 |
| xcs | cartpole | 1.0 | 0.95 | N/A | 0.1 | 0.3 | 0.99 | 10000 | 0.1 |
| xcs | cartpole | 1.0 | 0.9 | N/A | 0.1 | 0.3 | 0.99 | 10000 | 0.1 |
| xcs | cartpole | 0.95 | 1.0 | N/A | 0.1 | 0.3 | 0.99 | 10000 | 0.1 |
| xcs | cartpole | 0.95 | 0.95 | N/A | 0.1 | 0.3 | 0.99 | 10000 | 0.1 |
| xcs | cartpole | 0.95 | 0.9 | N/A | 0.1 | 0.3 | 0.99 | 10000 | 0.1 |
| xcs | cartpole | 0.9 | 1.0 | N/A | 0.1 | 0.3 | 0.99 | 10000 | 0.1 |
| xcs | cartpole | 0.9 | 0.95 | N/A | 0.1 | 0.3 | 0.99 | 10000 | 0.1 |
| xcs | cartpole | 0.9 | 0.9 | N/A | 0.1 | 0.3 | 0.99 | 10000 | 0.1 |
| xcs | cliffwalking | 1.0 | 1.0 | N/A | 0.35 | 0.35 | 0.95 | 2000 | 0.2 |
| xcs | cliffwalking | 1.0 | 0.95 | N/A | 0.35 | 0.35 | 0.95 | 2000 | 0.2 |
| xcs | cliffwalking | 1.0 | 0.9 | N/A | 0.1 | 0.3 | 0.95 | 2000 | 0.1 |
| xcs | cliffwalking | 0.95 | 1.0 | N/A | 0.35 | 0.2 | 0.95 | 2000 | 0.01 |
| xcs | cliffwalking | 0.95 | 0.95 | N/A | 0.05 | 0.2 | 0.99 | 2000 | 0.05 |
| xcs | cliffwalking | 0.95 | 0.9 | N/A | 0.2 | 0.2 | 0.95 | 2000 | 0.2 |
| xcs | cliffwalking | 0.9 | 1.0 | N/A | 0.05 | 0.2 | 0.95 | 2000 | 0.05 |
| xcs | cliffwalking | 0.9 | 0.95 | N/A | 0.2 | 0.25 | 0.85 | 1000 | 0.05 |
| xcs | cliffwalking | 0.9 | 0.9 | N/A | 0.3 | 0.4 | 0.85 | 2000 | 0.1 |
| xcs | frozenlake4x4 | 1.0 | 1.0 | N/A | 0.2 | 0.35 | 0.9 | 500 | 0.001 |
| xcs | frozenlake4x4 | 1.0 | 0.95 | N/A | 0.1 | 0.25 | 0.85 | 1000 | 0.005 |
| xcs | frozenlake4x4 | 1.0 | 0.9 | N/A | 0.2 | 0.2 | 0.85 | 500 | 0.02 |
| xcs | frozenlake4x4 | 0.95 | 1.0 | N/A | 0.3 | 0.2 | 0.85 | 500 | 0.01 |
| xcs | frozenlake4x4 | 0.95 | 0.95 | N/A | 0.35 | 0.35 | 0.99 | 500 | 0.001 |
| xcs | frozenlake4x4 | 0.95 | 0.9 | N/A | 0.1 | 0.4 | 0.85 | 500 | 0.01 |
| xcs | frozenlake4x4 | 0.9 | 1.0 | N/A | 0.3 | 0.3 | 0.95 | 1000 | 0.001 |
| xcs | frozenlake4x4 | 0.9 | 0.95 | N/A | 0.05 | 0.35 | 0.95 | 500 | 0.001 |
| xcs | frozenlake4x4 | 0.9 | 0.9 | N/A | 0.2 | 0.3 | 0.9 | 500 | 0.02 |
| xcs | frozenlake8x8 | 1.0 | 1.0 | N/A | 0.5 | 0.3 | 0.99 | 500 | 0.1 |
| xcs | frozenlake8x8 | 1.0 | 0.95 | N/A | 0.5 | 0.3 | 0.99 | 500 | 0.1 |
| xcs | frozenlake8x8 | 1.0 | 0.9 | N/A | 0.5 | 0.3 | 0.99 | 500 | 0.1 |
| xcs | frozenlake8x8 | 0.95 | 1.0 | N/A | 0.5 | 0.3 | 0.99 | 500 | 0.1 |
| xcs | frozenlake8x8 | 0.95 | 0.95 | N/A | 0.5 | 0.3 | 0.99 | 500 | 0.1 |
| xcs | frozenlake8x8 | 0.95 | 0.9 | N/A | 0.5 | 0.3 | 0.99 | 500 | 0.1 |
| xcs | frozenlake8x8 | 0.9 | 1.0 | N/A | 0.5 | 0.3 | 0.99 | 500 | 0.1 |
| xcs | frozenlake8x8 | 0.9 | 0.95 | N/A | 0.5 | 0.3 | 0.99 | 500 | 0.1 |
| xcs | frozenlake8x8 | 0.9 | 0.9 | N/A | 0.5 | 0.3 | 0.99 | 500 | 0.1 |
| xcs | taxi | 1.0 | 1.0 | N/A | 0.1 | 0.3 | 0.99 | 1250 | 0.1 |
| xcs | taxi | 1.0 | 0.95 | N/A | 0.1 | 0.3 | 0.99 | 1250 | 0.1 |
| xcs | taxi | 1.0 | 0.9 | N/A | 0.1 | 0.3 | 0.99 | 1250 | 0.1 |
| xcs | taxi | 0.95 | 1.0 | N/A | 0.1 | 0.3 | 0.99 | 1250 | 0.1 |
| xcs | taxi | 0.95 | 0.95 | N/A | 0.1 | 0.3 | 0.99 | 1250 | 0.1 |
| xcs | taxi | 0.95 | 0.9 | N/A | 0.1 | 0.3 | 0.99 | 1250 | 0.1 |
| xcs | taxi | 0.9 | 1.0 | N/A | 0.1 | 0.3 | 0.99 | 1250 | 0.1 |
| xcs | taxi | 0.9 | 0.95 | N/A | 0.1 | 0.3 | 0.99 | 1250 | 0.1 |
| xcs | taxi | 0.9 | 0.9 | N/A | 0.1 | 0.3 | 0.99 | 1250 | 0.1 |


#### Training Results

- Located in `train_and_log/results`.
- For each evaluation episode, metrics such as the episode score, the number of steps taken, and whether the episode was successful are stored.
- Organized as `benchmark/agent/instance/results.csv`, where:
  - `benchmark`: Names the benchmark (e.g., cartpole).
  - `agent`: Specifies the RL agent (e.g., DQN).
  - `instance`: Details the problem instance (e.g., state and action noise probabilities).
  - `results.csv` : Includes training metrics.

