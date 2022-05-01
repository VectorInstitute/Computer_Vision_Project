import os
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt


import argparse
import yaml
from argparse import Namespace
from dataset_factory.dota import DoTA

anomaly_dict = {'2': 'Start,Stop',
                '3': 'Moving,Waiting',
                '4': 'Lateral',
                '5': 'Oncoming',
                '6': 'Turning',
                '7': 'Pedestrian',
                '8': 'Obstacle',
                '9': 'OffRoad-L',
                '10': 'OffRoad-R',
                '11': 'Unknown',
}

object_dict = {'1': 'Person',
                '2': 'Rider',
                '3': 'Car',
                '4': 'Bus',
                '5': 'Truck',
                '6': 'Bike',
                '7': 'Motor',
}


run_name = 'matt_ce_05'

def get_detections_data(vid_id, epoch=200):
    detection_path = f'/home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/dota_detections/' \
                     f'{run_name}/detections_{epoch}/{vid_id}'
                     # f'yowo/matt_obj_anom_CE-Weight-10.0_run2/detections_{epoch}/{vid_id}'
    f = open(detection_path, 'r')
    t = f.read()
    detections = pd.DataFrame([s.split(' ')
                               for s in t.split('\n') if s != ''],
                              columns=['obj_label', 'conf', 'x1', 'y1', 'x2', 'y2', 'anomaly_label'])
    return detections.sort_values('conf', ascending=False)

# /home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/dota_detections/detections_96/lKMwX4nA64k_002645
# /home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/dota_detections/detections_100/lKMwX4nA64k_002645
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
        lbl = detection['obj_label']
        anom_lbl = detection['anomaly_label']

        confs.append(conf)
        labels.append(lbl)
        xs.append(x1)
        ys.append(y1)

        rects.append(patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                           linewidth=2, edgecolor='r', facecolor='none', linestyle='dashed'))
    return rects, labels, confs, xs, ys, anom_lbl


def visualize_detections(data_loader, epoch, n=100, top_k=4):
    images = []

    for i in range(0, n):
        test_sample = data_loader[i]
        vid_id = test_sample[0]
        video = test_sample[1]
        gt = test_sample[2][0:5]
        label = gt[0]
        anom_label = test_sample[3]
        bbox = gt[1:5]
        last_frame = video[:, -1, :, :].permute(1, 2, 0)

        factor = 224
        cx, cy, iw, ih = bbox
        lx = int(factor * (cx - (iw / 2)))
        ly = int(factor * (cy - (ih / 2)))
        w = int(factor * iw)
        h = int(factor * ih)

        detections = get_detections_data(vid_id, epoch)
        pred_rects, labels, confs, xs, ys, anom_pred = create_detection_rectangles(detections, top_k)

        print("True", label, "    | Prediction", labels[0], "(", confs[0], ")", "     ", vid_id)
        fig, ax = plt.subplots()
        ax.imshow(last_frame)
        true_rect = patches.Rectangle((lx, ly), w, h, linewidth=2, edgecolor='g', facecolor='none', linestyle='dashed')
        ax.add_patch(true_rect)

        obj_gt = object_dict[str(int(label+1))]
        anom_gt_text = anomaly_dict[str(int(anom_label))]
        anom_pred_text = anomaly_dict[str(int(anom_pred)-1)]

        plt.text(lx, ly, f"GT: {obj_gt}", fontsize=12, color='g')
        plt.text(0, 0, "AnomGT: " + anom_gt_text, fontsize=12, color='g')
        # plt.text(100, 0, "AnomPred: " + anom_pred_text, fontsize=12, color='r')

        # for j, pred_rect in enumerate(pred_rects):
        #     if confs[j] > 0.25:
        #         ax.add_patch(pred_rect)
        #         obj_pred = object_dict[str(int(labels[j]))]
        #         plt.text(xs[j], ys[j], f"Pred: {obj_pred}-{confs[j]}", fontsize=8, color='r')

        if not os.path.exists("/home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/example_outputs/GT/"):
            os.mkdir("/home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/example_outputs/GT/")
        # if not os.path.exists("/home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/example_outputs/" + run_name + "_figures"):
            # os.mkdir("/home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/example_outputs/" + run_name + "_figures")
        if not os.path.exists(f"/home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/example_outputs/" + run_name + "_figures/epoch_{}".format(epoch)):
            os.mkdir(f"/home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/example_outputs/" + run_name + "_figures/epoch_{}".format(epoch))
        plt.axis('off')
        # plt.savefig(f"/home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/example_outputs/" + run_name + "_figures/epoch_{}/{}".format(epoch, vid_id), bbox_inches='tight')
        plt.savefig(f"/home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/example_outputs/GT/" + "{}".format(vid_id), bbox_inches='tight')
        # plt.show(bbox_inches='tight')
        plt.close()
        print(i)


# with open("cfg/dota_config.yaml", 'r') as f:
with open("/home/matthew/Desktop/School/Research/yowo_dota/Computer_Vision_Project/yowo/cfg/dota_config.yaml", 'r') as f:
    dl_args = yaml.load(f)
dl_args = Namespace(**dl_args)

if not os.path.exists('/scratch/ssd002/datasets/cv_project/Detection-of-Traffic-Anomaly/dataset'):
    print('did not find vector data! -------------')
    dl_args.root = '/home/matthew/Desktop/Datasets/DoTA'
    dl_args.data_root = '/home/matthew/Desktop/Datasets/DoTA/DoTA_fol_train_data'
    dl_args.val_data_root = '/home/matthew/Desktop/Datasets/DoTA/DoTA_fol_val_data'
    dl_args.label_file = '/home/matthew/Desktop/Datasets/DoTA/metadata_val.json'
    dl_args.train_split = '/home/matthew/Desktop/Datasets/DoTA/train_split.txt'
    dl_args.val_split = '/home/matthew/Desktop/Datasets/DoTA/val_split.txt'
    dl_args.img_dir = '/home/matthew/Desktop/Datasets/DoTA/frames'

d = DoTA(dl_args, phase='test', n_frames=16, combined_bbox=False)

visualize_detections(d, epoch=300, n=1000, top_k=4)