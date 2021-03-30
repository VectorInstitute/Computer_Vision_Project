import os
import sys
os.chdir("mmaction2")
sys.path.append("mmaction2")
from mmaction import models
import torch
# Took config from here https://github.com/open-mmlab/mmaction2/blob/master/configs/_base_/models/slowfast_r50.py
config = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# Loading pretrained weights
checkpoint_path = "slowfast.pth"
slow_fast = models.build_model(config)
slow_fast.load_state_dict(torch.load(checkpoint_path)["state_dict"])
slow_fast.eval()

# Forward that works similar to torch.nn.Module
# Took the code from https://github.com/open-mmlab/mmaction2/blob/master/mmaction/models/recognizers/recognizer3d.py
#forward_train function
def hacked_forward(model, imgs):
    """Defines the computation performed at every call when training."""
    imgs = imgs.reshape((-1, ) + imgs.shape[2:])

    x = model.extract_feat(imgs)

    cls_score = model.cls_head(x)
    
    return cls_score