CONFIGURATIONS = {
    "deep_sarsa": [
        "{\"seed\": 21, \"agent\":{\"type\": \"deep_sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"taxi\", \"parameter\": {\"iterations\": 750, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"deep_sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"blackjack\", \"parameter\": {\"iterations\": 1000, \"natural\": true, \"det_prob_state\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"deep_sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"cartpole\", \"parameter\": {\"iterations\": 750,  \"encoding_type\": \"decimal\", \"discretization_bins\": 10, \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"deep_sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"frozenlake\", \"parameter\": {\"iterations\": 1000, \"desc_size\": 4, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"deep_sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"frozenlake\", \"parameter\": {\"iterations\": 1000, \"desc_size\": 8, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"deep_sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"cliffwalking\", \"parameter\": {\"iterations\": 500, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
    ],

    "dqn": [
        "{\"seed\": 21, \"agent\":{\"type\": \"dqn\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"taxi\", \"parameter\": {\"iterations\": 750, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"dqn\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"blackjack\", \"parameter\": {\"iterations\": 1000, \"natural\": true, \"det_prob_state\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"dqn\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"cartpole\", \"parameter\": {\"iterations\": 750,  \"encoding_type\": \"decimal\", \"discretization_bins\": 10, \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"dqn\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"frozenlake\", \"parameter\": {\"iterations\": 1000, \"desc_size\": 4, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"dqn\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"frozenlake\", \"parameter\": {\"iterations\": 1000, \"desc_size\": 8, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"dqn\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.0005, 0.00075, 0.001, 0.0025, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"cliffwalking\", \"parameter\": {\"iterations\": 1000, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
    ],

    "q_learning": [
        "{\"seed\": 21, \"agent\":{\"type\": \"q_learning\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.5, 0.1, 0.05, 0.001]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"taxi\", \"parameter\": {\"iterations\": 1000, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"q_learning\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.1, 0.066, 0.033, 0.01, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"blackjack\", \"parameter\": {\"iterations\": 1000, \"natural\": true, \"det_prob_state\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"q_learning\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.5, 0.1, 0.05, 0.001]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"cartpole\", \"parameter\": {\"iterations\": 1000,  \"encoding_type\": \"binary\", \"discretization_bins\": 10, \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"q_learning\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.1, 0.066, 0.033, 0.01, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"frozenlake\", \"parameter\": {\"iterations\": 1000, \"desc_size\": 4, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"q_learning\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.1, 0.066, 0.033, 0.01, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"frozenlake\", \"parameter\": {\"iterations\": 1000, \"desc_size\": 8, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"q_learning\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.1, 0.066, 0.033, 0.01, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"cliffwalking\", \"parameter\": {\"iterations\": 1000, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
    ],

    "sarsa": [
        "{\"seed\": 21, \"agent\":{\"type\": \"sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.5, 0.1, 0.05, 0.001]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"taxi\", \"parameter\": {\"iterations\": 1000, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.1, 0.066, 0.033, 0.01, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"blackjack\", \"parameter\": {\"iterations\": 1000, \"natural\": true, \"det_prob_state\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.5, 0.1, 0.05, 0.001]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"cartpole\", \"parameter\": {\"iterations\": 1000,  \"encoding_type\": \"binary\", \"discretization_bins\": 10, \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.1, 0.066, 0.033, 0.01, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"frozenlake\", \"parameter\": {\"iterations\": 1000, \"desc_size\": 4, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.1, 0.066, 0.033, 0.01, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"frozenlake\", \"parameter\": {\"iterations\": 1000, \"desc_size\": 8, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"sarsa\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"alpha\", \"values\": [0.1, 0.066, 0.033, 0.01, 0.005]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"cliffwalking\", \"parameter\": {\"iterations\": 1000, \"encoding_type\": \"one_hot\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}"
    ],

    "xcs": [
        "{\"seed\": 21, \"agent\":{\"type\": \"xcs\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"beta\", \"values\": [0.05, 0.1, 0.2, 0.3, 0.35]}, {\"name\": \"e0\", \"values\": [0.01, 0.05, 0.1, 0.2]}, {\"name\": \"pop_size\", \"values\": [500, 1250, 1000]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"taxi\", \"parameter\": {\"iterations\": 1000, \"encoding_type\": \"gray_code\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"xcs\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"beta\", \"values\": [0.05, 0.1, 0.2, 0.3, 0.35]}, {\"name\": \"e0\", \"values\": [0.005, 0.010, 0.015, 0.020, 0.025]}, {\"name\": \"pop_size\", \"values\": [704, 1350, 1000]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"blackjack\", \"parameter\": {\"iterations\": 1000, \"encoding_type\": \"gray_code\", \"natural\": true, \"det_prob_state\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"xcs\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"beta\", \"values\": [0.05, 0.1, 0.2, 0.3, 0.35]}, {\"name\": \"e0\", \"values\": [0.001, 0.005, 0.010, 0.015, 0.020]}, {\"name\": \"pop_size\", \"values\": [5000, 10000, 10000]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"cartpole\", \"parameter\": {\"iterations\": 1000,  \"encoding_type\": \"gray_code\", \"discretization_bins\": 10, \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"xcs\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"beta\", \"values\": [0.05, 0.1, 0.2, 0.3, 0.35]}, {\"name\": \"e0\", \"values\": [0.001, 0.005, 0.010, 0.015, 0.020]}, {\"name\": \"pop_size\", \"values\": [16, 500, 1000]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"frozenlake\", \"parameter\": {\"iterations\": 1000, \"desc_size\": 4, \"encoding_type\": \"gray_code\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"xcs\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"beta\", \"values\": [0.05, 0.1, 0.2, 0.3, 0.35]}, {\"name\": \"e0\", \"values\": [0.001, 0.005, 0.010, 0.015, 0.020]}, {\"name\": \"pop_size\", \"values\": [64, 500, 1000]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"frozenlake\", \"parameter\": {\"iterations\": 1000, \"desc_size\": 8, \"encoding_type\": \"gray_code\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
        "{\"seed\": 21, \"agent\":{\"type\": \"xcs\", \"parameter\": {}}, \"optim_parameter\": [{\"name\": \"beta\", \"values\": [0.05, 0.1, 0.2, 0.3, 0.35]}, {\"name\": \"e0\", \"values\": [0.01, 0.05, 0.1, 0.2]}, {\"name\": \"pop_size\", \"values\": [500, 1000, 1000]}, {\"name\": \"epsilon\", \"values\": [0.25, 0.30, 0.35]}, {\"name\": \"gamma\", \"values\": [0.90, 0.95, 0.99]}], \"environment\": {\"type\": \"cliffwalking\", \"parameter\": {\"iterations\": 1000, \"encoding_type\": \"gray_code\", \"det_prob_state\": 1.00, \"det_prob_action\": 1.00}}}",
    ],
}
