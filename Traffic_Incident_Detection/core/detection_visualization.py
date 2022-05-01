import os
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt


import argparse
import yaml
from argparse import Namespace
from dataset_factory.dota import DoTA

def get_detections_data(vid_id, epoch=200):
    detection_path = f'{detection_save_path}/detections_{epoch}/{vid_id}'
    f = open(detection_path, 'r')
    t = f.read()
    detections = pd.DataFrame([s.split(' ')
                               for s in t.split('\n') if s != ''],
                              columns=['label', 'conf', 'x1', 'y1', 'x2', 'y2'])
    return detections.sort_values('conf', ascending=False)


def create_detection_rectangles(detections, top_k=1):
    rects = []
    labels = []
    confs = []
    xs = []
    ys = []

    for i in range(0, top_k):
        if i >= len(detections):
            break

        detection = detections.iloc[i, :]
        x1 = int(detection['x1'])
        y1 = int(detection['y1'])
        x2 = int(detection['x2'])
        y2 = int(detection['y2'])
        conf = round(float(detection['conf']), 3)
        lbl = detection['label']

        confs.append(conf)
        labels.append(lbl)
        xs.append(x1)
        ys.append(y1)

        if i == 0:
            rects.append(patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                           linewidth=2, edgecolor='b', facecolor='none', linestyle='dashed'))
        else:
            rects.append(patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                           linewidth=1, edgecolor='r', facecolor='none', linestyle='dashed'))
    return rects, labels, confs, xs, ys


def visualize_detections(data_loader, epoch, n=100, top_k=4):
    for i in range(0, n):
        test_sample = data_loader[i]
        vid_id = test_sample[0]
        video = test_sample[1]
        gt = test_sample[2][0:5]
        label = gt[0]
        bbox = gt[1:5]
        last_frame = video[:, -1, :, :].permute(1, 2, 0)

        factor = 224
        cx, cy, iw, ih = bbox
        lx = int(factor * (cx - (iw / 2)))
        ly = int(factor * (cy - (ih / 2)))
        w = int(factor * iw)
        h = int(factor * ih)

        detections = get_detections_data(vid_id, epoch)
        pred_rects, labels, confs, xs, ys = create_detection_rectangles(detections, top_k)

        print("True", label, "    | Prediction", labels[0], "(", confs[0], ")", "     ", vid_id)
        fig, ax = plt.subplots()
        ax.imshow(last_frame)
        true_rect = patches.Rectangle((lx, ly), w, h, linewidth=2, edgecolor='g', facecolor='none', linestyle='dashed')
        ax.add_patch(true_rect)
        plt.text(lx, ly, f"GT-{label+1}", fontsize=12, color='g')

        for j, pred_rect in enumerate(pred_rects):
            if confs[j] > 0.25:
                ax.add_patch(pred_rect)

                if j == 0:
                    plt.text(xs[j], ys[j], f"Pred-{labels[j]}-{confs[j]}", fontsize=10, color='b')
                else:
                    plt.text(xs[j], ys[j], f"Pred-{labels[j]}-{confs[j]}", fontsize=8, color='r')

        if not os.path.exists("figures"):
            os.mkdir("figures")
        if not os.path.exists(f"figures/epoch_{epoch}"):
            os.mkdir(f"figures/epoch_{epoch}")
        plt.savefig(f"figures/epoch_{epoch}/{vid_id}")

config_file = 'cfg/dota_config.yaml'
with open(config_file, 'r') as f:
    dl_args = yaml.load(f)
dl_args = Namespace(**dl_args)

if not os.path.exists(dl_args.root):
    print('did not find data! -------------')
    sys.exit()

detection_save_path = 'dota_detections/run1'

d = DoTA(dl_args, phase='test', n_frames=16, combined_bbox=True)

visualize_detections(d, epoch=200, n=100, top_k=4)