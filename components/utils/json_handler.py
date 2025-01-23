import os
import json
import shutil


def create_single_conf(file_name: str) -> None:
    data = []

    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)


def write(file_name, value):
    try:
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return

    data.append(value)

    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def print_from_folder(folder_path):
    try:
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

        json_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)

                    if isinstance(data, list):
                        for item in data:
                            print(item)
                    else:
                        print(f"Error: File '{json_file}' does not contain a list.")

            except json.JSONDecodeError:
                print(f"Error: File '{json_file}' is not a valid JSON file.")
            except Exception as e:
                print(f"Error reading file '{json_file}': {e}")

    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"Error accessing folder '{folder_path}': {e}")


def delete(folder):
    try:
        shutil.rmtree(folder)
    except OSError as e:
        print(f"Error while deleting folder '{folder}': {e}")


def save_optim_run_info(file_path, entry_number, agent_info, env_info):
    value_string = f"\nEXPERIMENT INFORMATION OPTIMIZING (TRIAL {entry_number})\n"
    value_string += get_separation_line()

    for key, value in agent_info.items():
        value_string += f"{key}: {value}\n"

    value_string += "\n"

    for key, value in env_info.items():
        value_string += f"{key}: {value}\n"

    value_string += get_double_separation_line()

    write(file_path, value_string)


def save_optim_values(file_path, current_iteration, training_iterations, avg_train_last, avg_eval_last, avg_train_overall, avg_eval_overall, succeeded_episodes, indent=False):
    indent_str = "\t" if indent else ""

    value_string = f"{indent_str}##########INFO ({current_iteration}/{training_iterations} iterations done)##########\n"
    value_string += f"{indent_str}Training | Last avg train score: {avg_train_last:.2f} | Overall avg score: {avg_train_overall:.2f} \n"
    value_string += f"{indent_str}Evaluation | Last avg eval score: {avg_eval_last:.2f} | Overall avg score: {avg_eval_overall:.2f} | Successful episodes: {succeeded_episodes}\n"

    write(file_path, value_string)


def save_result(file_path, entry_number, current_iteration, run_index):
    indent_str = "\t" if run_index > 0 else ""
    run_indices = {1: 'First', 2: 'Second', 3: 'Third', 4: 'Fourth', 5: 'Fifth', 6: 'Sixth', 7: 'Seventh'}

    value_string = f"{indent_str}{run_indices[run_index]} run of trial {entry_number} finished after {current_iteration} iteration(s)!\n"

    write(file_path, value_string)


def save_run_number_of_trial(file_path, entry_number, run_index):
    indent_str = "\t" if run_index > 0 else ""

    value_string = f"{indent_str}Started {run_index}. run of trial {entry_number}!\n"

    write(file_path, value_string)


def get_double_separation_line(indent_str=""):
    return (f"{indent_str}---------------------------------------------\n"
            f"{indent_str}---------------------------------------------\n")


def get_separation_line(indent_str=""):
    return f"{indent_str}---------------------------------------------\n"


def save_dict_as_json(data_dict, file_path):
    try:
        with open(file_path, mode='w', encoding='utf-8') as json_file:
            json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
        print(f"File successfully saved at '{file_path}'.")
    except Exception as e:
        print(f"Error saving the file: {e}")
