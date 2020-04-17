# Mandatory assignment 2

IN5400/IN9400 - Machine Learning for Image Analysis
University of Oslo
Spring 2020

## Deadline
May 3 23:59, 2020


## Submission
zip the files, excluding the data directories, and upload the zipfile to devilry.


## Content

Everything you need for this exercise is contained in this folder. A brief description of the
content follows.

## Important
Note that this exercise is individual work. You are required to follow regulations for mandatory assignments at IFI.
Your code will be checked against plagiarism.


### `Exercise_Train_an_image_captioning_network.ipynb`

Everything related to the assignment. This should be self-contained, and all information is found
in this notebook. You can start the notebook from the command line with

```
$ jupyter notebook Exercise_Train_an_image_captioning_network.ipynb
```

The data for this assignment can be found in
`data_preparation_download_onedrive.ipynb`

### Content of supplied code

The exercise contains this notebook and multiple folders with code:

```
 → tree
.
├── data
│
├── loss_images
│
├── sourceFiles
│   ├── cocoSource.py
│
├── unit_tests
│   ├── unit_tests.py
│   ├── cifar10_progress_default.png
│   ├── convolution_same.png
│   ├── convolution_same_x11.png
│   ├── convolution_same_x12.png
│   ├── convolution_same_x33.png
│   ├── mnist_progress_default.png
│   └── svhn_progress_default.png
├── utils
│   ├── dataLoader.py
│   ├── model.py
│   ├── plotter.py
│   ├── saverRestorer.py
│   ├── trainer.py
│   ├── validate.py
|
├── utils_data_preparation
│   ├── cocoDataset.py
│   ├── downloadCoco.py
│   ├── generateVocabulary.py
│   ├── produce_cnn_features.py
|
├── utils_data_preparation_download_onedrive
│   ├── download_from_onedrive.py
|
├── utils_images
|
├── data_preparation_download_onedrive.ipynb
│
├── Exercise_Train_an_image_captioning_network.ipynb
|
├── collectSubmission.sh
|
├── data_preparation.py
|
├── data_preparation_download_onedrive.py
|
├── Exercise_Train_an_image_captioning_network.py
│
└── README.md
```


#### `data_preparation_download_onedrive.ipynb`

Handles import of the following dataset

- COCO

You should not need to change anything here.


#### `sourceFiles/cocoSource.py`

Implements all the important logic of the classifier.

Everything you need to implement will be located in this file.

#### `unit_tests`

In this folder, predefined arrays are defined. To be used when checking your implementations.


## Resources

#### Theory:
- https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
- https://medium.com/dair-ai/building-rnns-is-fun-with-pytorch-and-google-colab-3903ea9a3a79

#### Code:
- https://github.com/ivartz/IN9400_exercises/tree/master/Mandatory2/oblig2_assignment_rev3
- http://ethen8181.github.io/machine-learning/deep_learning/rnn/1_pytorch_rnn.html
- https://github.com/mhmorta/IFT6135_RNN_GRU_Transformers/blob/master/models.py
