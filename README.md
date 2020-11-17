# ICBEB/CPSC Challenge 2020

Premature Beat Detection from Long-term ECGs Using Modified U-Net  

Shan Yang, Heng Xiang, Chunli Wang, Qingda Kong

Chengdu Spaceon Electronics Co., LTD.

## Abstract
Premature ventricular contraction (PVC) and supraventricular premature beat (SPB) are the most common arrhythmias, the detection of which plays an important role in ECG signal analysis.
Accurate detection is a challenging task from 24-hour dynamic single-lead ECG recordings. The rulebased PVC and SPB detection methods largely depend on hand-crafted manual features and parameters,
the fixed features and parameters of which require difficult offline tuning for adapting to new scenarios. In the 3rd China Physiological Signal Challenge 2020 (CPSC 2020), inspired by the popular application
of U-Net in medical image segmentation, the U-Net-like architecture based on 1-D convolutional neural network (CNN) is proposed. The ResNet and ResNeXt block are introduced as backbone of encoder and
decoder in the 1D U-Net model. In addition, the ECG records with frequency of 400 Hz are resampled to 200Hz, and to make the length of data fed into network is suitable, zero padding and data truncation
are introduced. To increase the diversity of dataset and improve the generalization performance, some common techniques of data augmentation used in this study consist of noise addition, y-axis shift, and
wavelet-based filter.
The proposed method has been validated against the 3rd china physiological signal challenge data set, obtaining a PVC score of 51335, SPB score of 72488 on the hidden subtest set. Experimental results
show that the proposed method acquires competitive performance.


## Contents

    'PythonEntry' is a folder contains the submit prediction code and model weight.
    'src_preliminary' is a folder contains the preliminary training and validation code.
    'src_final' is a folder contains the final training and validation code.
    'TrainingSet' is a folder contains the final training data.

## Use

You can run this code by installing the requirements and running

    python CPSC_YS1019.py   
    python CPSC_YS1020_predict.py --mtype resnet --model model_weights.h5

where `mtype` is a parameter of model type, `model` is a parameter of model weight, The [ICBEB/CPSC 2020 webpage](http://www.icbeb.org/CSPC2020) provides a training database with data files and a description of the contents and structure of these files.

## Submission

    python CPSC2020_score.py 

which obtains `score.txt`, example: 

    S_score: -3483.0
	V_score: -11086.0

## Details

See the [ICBEB/CPSC 2020 webpage](http://www.icbeb.org/CSPC2020) for more details, including instructions for the other files in this repository.
