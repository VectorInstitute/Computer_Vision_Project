# Defect Detection

## Overview

Anomaly detection is an important task in computer vision that is concerned with identifying anomalous images given a training set of only normal images. In anomaly segmentation, the concept of anomaly detection is extended to the pixel level in order to identify anomalous regions of images. There are many applications to anomaly detection including biomedical image segmentation, video surveillance and defect detection. In particular, defect detection involves detecting abnormalities in manufacturing components and so is widely used in the industry to enhance quality assurance and efficiency in the production process \cite{bergmann2019mvtec}. However, having a person manually inspect each component is not feasible in most cases. To address this, systems have been proposed to automate the detection of defective components. These approaches generally take as input an image of a component and output a label or pixel-level mask that predicts whether the image or pixel is anomalous. Although initial approaches were generally ineffective, newer, deep learning based approaches have shown very strong performance in anomaly detection and segmentation \cite{sabokrou2018adversarially}. Thus, these new methods have the potential to dramatically increase quality assurance and efficiency. In order to compare anomaly detection methods, several datasets have been proposed as benchmarks such as MNIST \cite{deng2012mnist}, CIFAR \cite{krizhevsky2009learning}, and UCSD \cite{chan2008ucsd}, whereas there are much fewer benchmark datasets for the anomaly segmentation task. To address this, the MVTec Anomaly Detection Dataset \cite{bergmann2019mvtec} was recently introduced as a benchmark for anomaly segmentation. 
%MVTec is focused on industrial inspection; consisting of a training set of normal images of objects and textures as well as a test set with both normal and anomalous samples along with their corresponding labels. There are over 70 different types of defects across the anomalous images that are typical in the manufacturing process. The quality and practical nature of the MVTec dataset has made it a popular benchmark for recently proposed anomaly segmentation methods. The goal of this focus phase of the project is to apply state-of-the-art methods to accurately segment anomalies in the MVTec dataset. In doing so, we compared the performance of different anomaly segmentation methods in the industrial inspection setting. Additionally, we sought to optimize the performance of the methods  by altering the hyperparameters and architectures of the approaches. The approaches and corresponding results will be discussed at length in the following section.

## Dataset 
In order to explore the application of autoencoders to the task of anomaly segmentation in manufacturing, the MVTec anomaly detection dataset was used. It contains 5354 high-resolution images from 15 different object categories and includes 70 different types of defects across the anomalous images that are typical in the manufacturing process. For each object category, a training set of normal images of objects and textures as well as a test set with both normal and anomalous samples along with their corresponding labels.
  
<p align="center">
<img width="600" alt="mvtec dataset" src="https://user-images.githubusercontent.com/34798787/162048399-331745f0-1924-4323-af32-8174b5913ccf.png">  
    <br>
<div align="center"> 
   <b> Figure 1:</b>  An example of inlier images (left) and labels (right) for multiple object categories in the MVTec dataset.
</div> 
</p>

## Experimental Setup 
The MVTEC dataset object categories each include a train set of normal samples and a test set of both normal and anomalous samples. Models were optimized to be able to reconstruct samples from the inlier distribution during the training phase. Subsequently, at test time, both normal and anomalous images are input to the model and the pixelwise reconstruction error of samples is used to identify anomalous regions. Specifically, the models were evaluated on the testing data for each of the object categories and the average area under the ROC curve (AUC) is reported. A small validation set of normal images is used to determine which model step yields the most optimal set of parameters. Specifically, 10\% of images were randomly removed from the train set and used as the validation set. For testing, the entire test set was used and the average AUC across object categories is reported for each method. 
  
## Results 

<p align="center">
<img width="200" alt="vtec results" src="https://user-images.githubusercontent.com/34798787/162048982-38f64064-0893-440b-8ad9-9677d907d6ad.png">  
    <br>
<div align="center"> 
   <b> Figure 2:</b> Avergae AUC score on test set for each approach.
</div> 
</p>

<p align="center">
<img width="1000" alt="mvtec visual result" src="https://user-images.githubusercontent.com/34798787/162049359-00a997e7-69ef-42d9-852f-b0e9e98c242d.png">  
    <br> 
<div align="center"> 
    <b>Figure 3: </b> A visualization of the predictions generated by the network for an anomolous sample. 
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
python train.py --model vae --epochs 10 --batch_size 4
```

The **train.py** script has the following arguments: 
- **model**:        (str): Architecture variation for experiments.
- **epochs**        (int): The number of epochs to train the memory.
- **batch_size**    (int) The batch size for training, validation and testing.
- **learning_rate** (float): Learning rates of memory units.
- **size**          (int): Side length of input image. 
- **data_path**    (str): The root directory of the dataset.
- **ckpt_path**    (str): The directory to save model checkpoints.
