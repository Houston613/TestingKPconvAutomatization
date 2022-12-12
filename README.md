Badge fpr Develop [![small Testing for KPConvAutomatization](https://github.com/Houston613/TestingKPconvAutomatization/actions/workflows/python_test.yml/badge.svg?branch=develop)](https://github.com/Houston613/TestingKPconvAutomatization/actions/workflows/python_test.yml)


Badge for main [![small Testing for KPConvAutomatization](https://github.com/Houston613/TestingKPconvAutomatization/actions/workflows/python_test.yml/badge.svg)](https://github.com/Houston613/TestingKPconvAutomatization/actions/workflows/python_test.yml)


# TestingKPconvAutomatization
This repository allows testing a pre-trained [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) on synthetic city data for point cloud segmentation.

## Testing process


### Selecting the test log

Initially, when starting up, we need to specify which training results we want to use. A list of available logs will be provided to choose from.

![some text](doc/Screenshot_1.png?raw=true)


### Selecting the data folder

Next, you need to create and place your data in the directory `data/name_of_your_dataset/original_ply/`.
If you forget the last folder, it will be created automatically and all files will be moved there

![some text](doc/Screenshot_2.png?raw=true)

### Selecting files
A list of the files available in this folder will be displayed. You need only select at least one file for testing. If you select all the files, the next step will be performed. If you need to select part of the files, you can complete the selection by entering `-1`.

![some text](doc/Screenshot_3.png?raw=true)

### Test phase 
Further testing will be launched

![some text](doc/Screenshot_5.png?raw=true)


### Metrics

If your data was originally marked up, you can do a comparison with how the data was marked up using the neural network. To do this, you can get a Confusion matrix and Jaccard index for each of the files used to evaluate the segmentation of each class

![some text](doc/Screenshot_4.png?raw=true)


## Docker

To start with you need to install [Nvidia-Container-Toolkit](https://github.com/NVIDIA/nvidia-docker) - this will allow you to use the video card in Docker

Build image:
```
sudo docker build -t name_like_you_want .
```

Run image:


```
sudo docker run --gpus all -ti --name test --rm --shm-size=20gb --mount type=bind,source="$(pwd)"/../testovik,target=/workspace/test  name_like_you_want 
```
`shm` used for saving staging data. Its size depends on the size of your data. If the image closes as a result of the test - increase the memory size

`--mount type=bind,source="$(pwd)"/../testovik,target=/workspace/test` used to saving you test results to specified in `source` folder. After closing the image, you will be able to access them and use the resulting point cloud annotations as you wish
if you have no need for the results, you can remove this option







