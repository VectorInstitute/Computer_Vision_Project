# Road Obstacle Detection

## Overview

Detecting obstacles on the road/railway is a critical part of the driving task which has not been mastered by fully autonomous vehicles. Semantic segmentation plays an important role in addressing the challenges of identifying the locations of obstacles. In this phase of the project, we explore the application of semantic segmentation methods to the task of detecting road obstacles using the Lost and Found Dataset. The goal of the experiments is to determine which model architecture is the best for road obstacle detection - something that is of interest to both the practioner and researchers. 

## Dataset
The Lost and Found dataset was introduced to evaluate the performance of small road obstacle detection approaches. The Lost and Found Dataset includes 2k images recording from 13 different challenging street scenarios, featuring 37 different obstacles types.  Each object is labeled with a unique ID, allowing for a later refinement into subcategories. An overview of the Lost and Found dataset is available below, which is refined into three classes: driveable area, non drivable area and obstacles.

<p align="center">
<img width="750" alt="The Lost and Found Dataset" src="https://user-images.githubusercontent.com/34798787/163045378-e327a9c9-0738-4c4a-8a68-273ae659d3f7.png">  
    <br>
<div align="center"> 
   <b> Figure 1:</b> The Lost and Found Dataset.
</div> 
</p>

## Results

<p align="center">
<img width="750" alt="Validation Cross Entropy" src="https://user-images.githubusercontent.com/34798787/163045762-eee689fe-3115-4a49-8453-3e6ef5ab7deb.png">  
    <br>
<div align="center"> 
   <b> Figure 2:</b> The validation cross entropy loss for each model across epochs.
</div> 
</p>

<p align="center">
<img width="750" alt="Visual Result" src="https://user-images.githubusercontent.com/34798787/163046353-4929f6bb-126f-4ad5-b924-68a724cfa2f1.png">  
    <br>
<div align="center"> 
   <b> Figure 3:</b> Visual results comparing prediction made by each model for a test image. 
</div> 
</p>

## Running Code
To configure the environment to run the experiments navigate to the base of this directory and execute the following commands: 

```
conda create -n new_env
conda activate new_env 
pip install -r requirements.txt
```

To obtain results for a specific architecture simply pass the appropriate arguments to the **train.py** script: 
```
python train.py --epochs 10 --batch_size 4
```

The **train.py** script has the following arguments: 
- **epochs**        (int): The number of epochs to train the memory.
- **batch_size**    (int) The batch size for training, validation and testing.
- **learning_rate** (float): Learning rates of memory units.
- **height**        (int): Height of input image. 
- **width**         (int): Width of input image. 
- **train_perc**    (float): The proportion of samples used for train.
- **data_path**    (str): The root directory of the dataset.
- **ckpt_path**    (str): Path of checkpoint file. 
- **best_ckpt_path**  (str): Path of checkpoint file for best performing model on the validation set. 
- **sample_path**    (str): Path of file to save example images. 


