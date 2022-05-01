# Vector CV Project - Two Stream Traffic Accident Detection with YOWO

In this project, we use ***YOWO*** (***Y**ou **O**nly **W**atch **O**nce*), a unified CNN architecture designed for real-time spatiotemporal action localization, to detect traffic accidents in video. The codebase is built off of the [YOWO PyTorch Official Repository](https://github.com/wei-tim/YOWO).

The repository contains PyTorch code for accident detection on the DoTA dataset and spatiotemporal action localization AVA, UCF101-24 and JHMDB datasets!

Please see the [Computer Vision Project Report - coming soon!]() for more details on the project and method.

## Installation

### Datasets

* DOTA     : Download from [here](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
* AVA	   : Download from [here](https://github.com/cvdfoundation/ava-dataset)
* UCF101-24: Download from [here](https://drive.google.com/file/d/1o2l6nYhd-0DDXGP-IPReBP4y1ffVmGSE/view?usp=sharing)
* J-HMDB-21: Download from [here](http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets)

Use instructions [here](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md) for the preparation of AVA dataset.

Modify the paths in ucf24.data and jhmdb21.data under cfg directory accordingly.
Download the dataset annotations from [here](https://www.dropbox.com/sh/16jv2kwzom1pmlt/AABL3cFWDfG5MuH9PwnjSJf0a?dl=0).

### Download backbone pretrained weights

* Darknet-19 weights can be downloaded via:
```bash
wget http://pjreddie.com/media/files/yolo.weights
```

* ResNet pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).
* For resource efficient 3D CNN architectures (ShuffleNet, ShuffleNetv2, MobileNet, MobileNetv2), pretrained models can be downloaded from [here](https://github.com/okankop/Efficient-3DCNNs).

### Pretrained YOWO models

Pretrained models for UCF101-24 and J-HMDB-21 datasets can be downloaded from [here](https://www.dropbox.com/sh/16jv2kwzom1pmlt/AABL3cFWDfG5MuH9PwnjSJf0a?dl=0).
Pretrained models for AVA dataset can be downloaded from [here](https://drive.google.com/drive/folders/1g-jTfxCV9_uNFr61pjo4VxNfgDlbWLlb?usp=sharing).

All materials (annotations and pretrained models) are also available in Baiduyun Disk:
[here](https://pan.baidu.com/s/1yaOYqzcEx96z9gAkOhMnvQ) with password 95mm

## Running the code

* All training configurations are given in cfg/dota_train.yaml cfg/ava.yaml cfg/ucf24.yaml, and cfg/jhmdb.yaml files.
* DoTA training:
```bash
python main_dota.py --cfg cfg/dota_train.yaml
```
* AVA training:
```bash
python main.py --cfg cfg/ava.yaml
```
* UCF101-24 training:
```bash
python main.py --cfg cfg/ucf24.yaml
```
* J-HMDB-21 training:
```bash
python main.py --cfg cfg/jhmdb.yaml
```

## Validating the model

* For AVA dataset, after each epoch, validation is performed and frame-mAP score is provided.

* Note that calculating frame-mAP with DoTA is not currently implemented and precision and recall from the validation epoch are used as the main evaluation metrics.

* For DoTA, UCF101-24 and J-HMDB-21 datasets, after each validation, frame detections is recorded under 'dota_detections', 'jhmdb_detections' or 'ucf_detections'. From [here](https://www.dropbox.com/sh/16jv2kwzom1pmlt/AABL3cFWDfG5MuH9PwnjSJf0a?dl=0), 'groundtruths_jhmdb.zip' and 'groundtruths_jhmdb.zip' should be downloaded and extracted to "evaluation/Object-Detection-Metrics". Then, run the following command to calculate frame_mAP.

```bash
python evaluation/Object-Detection-Metrics/pascalvoc.py --gtfolder PATH-TO-GROUNDTRUTHS-FOLDER --detfolder PATH-TO-DETECTIONS-FOLDER

```

* For video_mAP, set the pretrained model in the correct yaml file and run:
```bash
python video_mAP.py --cfg cfg/ucf24.yaml
```

## Running on a test video

* You can run AVA pretrained model on any test video with the following code:
```bash
python test_video_ava.py --cfg cfg/ava.yaml
```