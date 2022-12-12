import pytest
import os

def testPlyFolder():
    
    path_to_data = os.getcwd() + "/tests/data/"
    list_of_data = os.listdir(path_to_data)
    chosen_data_folder = f"{path_to_data}/{list_of_data[0]}"
    all_files_in_data_folder = os.listdir(chosen_data_folder)
    assert 'original_ply' not in all_files_in_data_folder


def testInputValidation(capsys):
    path_to_data = os.getcwd() + "/tests/data/"
    list_of_data = os.listdir(path_to_data)
    chosen_data_folder = f"{path_to_data}/{list_of_data[0]}"
    current_folder_with_data = f"{chosen_data_folder}"

    temp_list_of_filenames_in_ply_folder = next(os.walk(current_folder_with_data))[2]

    
    list_of_filenames_in_ply_folder = []
    
    for i in range(len(temp_list_of_filenames_in_ply_folder)):
        list_of_filenames_in_ply_folder.append(os.path.splitext(temp_list_of_filenames_in_ply_folder[i])[0])

    index_of_all_data  = []
    for i in range(len(list_of_filenames_in_ply_folder)):
        index_of_all_data.append(i) 


    value_of_input_for_data = 12
    if value_of_input_for_data < len(list_of_filenames_in_ply_folder):
        print("Maybe smth else? That's list of what left")
    else:
        print("you entered inccorent number")
    
    captured = capsys.readouterr()
    assert captured.out == "you entered inccorent number\n"
