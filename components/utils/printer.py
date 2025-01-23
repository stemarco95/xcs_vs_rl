import os
from datetime import datetime


def print_best_param_and_value(best_param, best_value):
    print_double_separation_line()
    print("-------------BEST RESULTS-------------")
    print('Best parameters:', best_param)
    print('Best value:', best_value)
    print_double_separation_line()


def print_double_separation_line():
    print("-----------------------------------------------------------------------")
    print("-----------------------------------------------------------------------")


def print_separation_line():
    print("-----------------------------------------------------------------------")


def print_train_run_info(agent_info, env_info, current_conf, num_confs):
    print(f"\nEXPERIMENT INFORMATION TRAIN [CONF: {current_conf}/{num_confs} | DATE: {datetime.now().strftime('%d/%m/%Y %H:%M')}]")
    print_separation_line()
    for key, value in agent_info.items():
        print(f"{key}: {value}")

    print()

    for key, value in env_info.items():
        print(f"{key}: {value}")

    print_double_separation_line()
    print()


def print_train_eval_values(current_iteration, training_iterations, avg_eval_last, avg_eval_overall, eval_itime, succeeded_episodes, trainings_status):
    trainings_status = "Run" if trainings_status else "Done"
    print(f"\t##########INFO ({current_iteration}/{training_iterations} iterations done)##########")
    print(f"\tEvaluation | Last avg eval score: {avg_eval_last:.2f} | Overall avg score: {avg_eval_overall:.2f} | Avg time per episode (eval): {eval_itime:.7f}s | Successful episodes: {succeeded_episodes} | Trainings status: {trainings_status}\n")
