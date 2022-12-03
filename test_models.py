#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import torch
import pathlib
import shutil
from plyfile import PlyData, PlyElement
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score


# Dataset
from datasets.NPM3D import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model


    path_to_results = os.getcwd() + "/results"
    list_of_logs = os.listdir(path_to_results)
    print("Choose Training Log: Enter a number of Log")
    print(list_of_logs)

    for i in range(len(list_of_logs)):
        print(f"{i} {list_of_logs[i]}")

    while True:
        input_number = input()
        try:
            value_of_input_for_log = int(input_number)
            if value_of_input_for_log < len(list_of_logs):
                print(f"You choosed log: {value_of_input_for_log}")                
                break
            print(f"you entered inccorent number")
        except ValueError:
            print("Please enter a number")
            pass
        
        
    
    chosen_log = f"results/{list_of_logs[value_of_input_for_log]}"

    path_to_data = os.getcwd() + "/data"
    list_of_data = os.listdir(path_to_data)


    print("Choose Data folder: That's directory with name of data files. Please, specify one of them")
    print(list_of_data)

    for i in range(len(list_of_data)):
        print(f"{i} {list_of_data[i]}")

    while True:
        input_number = input()
        try:
            value_of_input_for_data = int(input_number)
            if value_of_input_for_data < len(list_of_data):
                print(f"You choosed data folder: {value_of_input_for_data}")                
                break
            print(f"you entered inccorent number")
        except ValueError:
            print("Please enter a number")
            pass

    chosen_data_folder = f"{path_to_data}/{list_of_data[value_of_input_for_data]}"
    print(chosen_data_folder)

    all_files_in_data_folder = os.listdir(chosen_data_folder)

    if 'original_ply' not in all_files_in_data_folder:
        print('You have not specified original_ply folder in data folder. All .ply must be in that folder. I will create it by next line of code and add all you ply files in there')
        os.mkdir(os.path.join(chosen_data_folder, 'original_ply'))
    

    list_of_filenames_in_data_folder = next(os.walk(chosen_data_folder))[2]
    print(list_of_filenames_in_data_folder)

    for i in range(len(list_of_filenames_in_data_folder)):
        current_file = list_of_filenames_in_data_folder[i]
        file_extension = pathlib.Path(current_file).suffix
        if file_extension == '.ply':
            
            print(current_file)
            str = f"{chosen_data_folder}/original_ply/"
            print(str)
            shutil.move(f"{chosen_data_folder}/{current_file}",f"{chosen_data_folder}/original_ply/{current_file}")
    
    print("Please specify what files you want to be test files")
    print("Write at least one number")
    print("if you add enough files - write -1")    

    current_folder_with_data = f"{chosen_data_folder}/original_ply/"

    temp_list_of_filenames_in_ply_folder = next(os.walk(current_folder_with_data))[2]
    list_of_filenames_in_ply_folder = []
    
    for i in range(len(temp_list_of_filenames_in_ply_folder)):
        list_of_filenames_in_ply_folder.append(os.path.splitext(temp_list_of_filenames_in_ply_folder[i])[0])

    copy_list = list_of_filenames_in_ply_folder.copy()

    for i in range(len(list_of_filenames_in_ply_folder)):
        print(f"{i} {list_of_filenames_in_ply_folder[i]}")

    is_at_least_one_file_added = False

    index_of_test_data = []
    index_of_all_data  = []
    test_point_cloud_names = []
    for i in range(len(list_of_filenames_in_ply_folder)):
        index_of_all_data.append(i)



    while True:
        input_number = input()
        try:
            value_of_input_for_data = int(input_number)
            if value_of_input_for_data == -1:
                if is_at_least_one_file_added:
                    print("OK you choosed enough files")
                    break
                else:
                    print("You need to choose at least one file to test")
            else:
                if value_of_input_for_data < len(list_of_filenames_in_ply_folder):
                    item = list_of_filenames_in_ply_folder[value_of_input_for_data]
                    print(f"You choosed file: {item}")
                    test_point_cloud_names.append(item)
                    index_of_test_data.append(copy_list.index(item))
                    list_of_filenames_in_ply_folder.remove(item)
                    print("Maybe smth else? That's list of what left")
                    is_at_least_one_file_added = True
                    if len(list_of_filenames_in_ply_folder) == 0:
                        print("You choosed all files!")
                        break
                else:
                    print(f"you entered inccorent number")
        except ValueError:
            print("Please enter a number")
            pass





    
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = -1

    # Choose to test on validation or test split
    on_val = False



    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    #config.batch_num = 3
    #config.in_radius = 4
    config.validation_size = 200
    config.input_threads = 10


    print()
    print('Data Preparation')
    print('****************')

    point_cloud_names = copy_list
    train_point_cloud_ind = None
    valid_point_cloud_ind = None

    test_point_cloud_ind = index_of_test_data

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'NPM3D':
        test_dataset = NPM3DDataset(config, chosen_data_folder, point_cloud_names, index_of_all_data, train_point_cloud_ind, valid_point_cloud_ind, test_point_cloud_ind, test_point_cloud_names, set=set)
        test_sampler = NPM3DSampler(test_dataset)
        collate_fn = NPM3DCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    if config.dataset_task == 'classification':
        net = KPCNN(config)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    if config.dataset_task == 'classification':
        tester.classification_test(net, test_loader, config)
    elif config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config)
    elif config.dataset_task == 'slam_segmentation':
        tester.slam_segmentation_test(net, test_loader, config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    
    need_to_start_metric_analyze = False
    print("Does you test data have field with label?(yes/no)")

    while(True):
        user_input = input()
        if user_input.lower() == 'yes':
            need_to_start_metric_analyze = True
            break
        elif user_input.lower() == 'no':
            print("well, that's mean we cannot find some metrics")
            need_to_start_metric_analyze = False
            break
        else:
            print('Type yes or no')
    
    if need_to_start_metric_analyze:
        print("What did you call this label?")
        label = input()
        label = f"{label}"
        print(test_point_cloud_names)
        print("That's your test files. Choose one and you will see confusion matrix and jaccard score")
        
        for i in range(len(list_of_data)):
            print(f"{i} {list_of_data[i]}")
        print("You can do this as long as you do not print -1") 
        while True:
            input_number = input()
            try:
                value_of_input_for_data = int(input_number)
                if value_of_input_for_data == -1:
                    break
                else:
                    if value_of_input_for_data < len(test_point_cloud_names):
                        item = f"{test_point_cloud_names[value_of_input_for_data]}.ply"

                        print(f"You choosed file: {item}")
                        full_path_to_original = f"{chosen_data_folder}/original_ply/{item}"
                        point_cloud_orig = PlyData.read(full_path_to_original)
                        data_original = point_cloud_orig['vertex'][label]
                        data_original.astype(np.int32)

                        full_path_to_result = f"{os.getcwd()}/test/{list_of_logs[value_of_input_for_log]}/predictions/{test_point_cloud_names[value_of_input_for_data]}.txt"
                        print(full_path_to_result)
                        data_from_test = np.loadtxt(full_path_to_result)
                        a = confusion_matrix(data_original, data_from_test)
                        print(a)
                        a = jaccard_score(data_original, data_from_test, average=None)
                        print(a)
                        print("Maybe smth else?")
                        
                    else:
                        print(f"you entered inccorent number")
            except ValueError:
                print("Please enter a number")
                pass
