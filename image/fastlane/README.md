# Vector Fastlane

You need to have conda installed on you machine. Follow these intructions

```
conda create -n pytorch181 python=3.9
conda activate pytorch181
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install albumentations scikit-learn scikit-image matplotlib opencv-python yacs joblib natsort h5py tqdm
pip install gdown addict future pyyaml requests scipy yapf editdistance pyclipper pandas==1.4.0 shapely==2.0.1
```

You can download the datasets and pretrained weights from this [link](https://drive.google.com/drive/folders/1qqK1uQsgkj0MT7yOhx33mTRlISy27QCA?usp=share_link).
