import os
import numpy as np
import glob
import pickle as pkl
import json
import random

import torch
from torch.utils import data
import pdb
import time
from tqdm import tqdm

from torchvision.transforms import ToTensor, Resize, Scale

from PIL import Image
from dataset_factory.clip import *


def read_truths_args(lbl, min_box_scale):
    truths = lbl
    new_truths = []
    for i in range(truths.shape[0]):
        cx = (truths[i][1] + truths[i][3]) / (2 * 1280)  # (2 * 240)
        cy = (truths[i][2] + truths[i][4]) / (2 * 720)  # (2 * 240)
        imgw = (truths[i][3] - truths[i][1]) / 1280  # 240
        imgh = (truths[i][4] - truths[i][2]) / 720  # 240
        truths[i][0] = truths[i][0] - 1
        truths[i][1] = cx
        truths[i][2] = cy
        truths[i][3] = imgw
        truths[i][4] = imgh

        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


def fill_truth_detection(lbl, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes, 5))

    if lbl is None:
        return label
    bs = np.reshape(lbl, (-1, 5))

    for i in range(bs.shape[0]):
        cx = (bs[i][1] + bs[i][3]) / (2 * 1280)  # (2 * 240) #(2*1280) #(2 * 320)
        cy = (bs[i][2] + bs[i][4]) / (2 * 720)  # (2 * 240)# (2*720) #(2 * 240)
        imgw = (bs[i][3] - bs[i][1]) / 1280  # 240 #(1280)#320
        imgh = (bs[i][4] - bs[i][2]) / 720  # 240 #(720)#240
        bs[i][0] = bs[i][0] - 1
        bs[i][1] = cx
        bs[i][2] = cy
        bs[i][3] = imgw
        bs[i][4] = imgh

    cc = 0
    for i in range(bs.shape[0]):
        x1 = bs[i][1] - bs[i][3] / 2
        y1 = bs[i][2] - bs[i][4] / 2
        x2 = bs[i][1] + bs[i][3] / 2
        y2 = bs[i][2] + bs[i][4] / 2

        x1 = min(0.999, max(0, x1 * sx - dx))
        y1 = min(0.999, max(0, y1 * sy - dy))
        x2 = min(0.999, max(0, x2 * sx - dx))
        y2 = min(0.999, max(0, y2 * sy - dy))

        bs[i][1] = (x1 + x2) / 2
        bs[i][2] = (y1 + y2) / 2
        bs[i][3] = (x2 - x1)
        bs[i][4] = (y2 - y1)

        if flip:
            bs[i][1] = 0.999 - bs[i][1]

        if bs[i][3] < 0.001 or bs[i][4] < 0.001:
            continue
        label[cc] = bs[i]
        cc += 1
        if cc >= 50:
            break

    label = np.reshape(label, (-1))
    return label


def load_data_detection(img_paths, label=None, train=True, shape=None, jitter=0.2, hue=0.1, saturation=1.5,
                        exposure=1.5):
    clip = []

    for path in img_paths:
        clip.append(
            Resize((224, 224), interpolation=Image.NEAREST)(Image.open(path).convert('RGB')))  # .resize((224, 224))

    if train:  # Apply augmentation
        clip, flip, dx, dy, sx, sy = data_augmentation(clip, shape, jitter, hue, saturation, exposure)
        label = fill_truth_detection(label, clip[0].width, clip[0].height, flip, dx, dy, 1. / sx, 1. / sy)
        label = torch.from_numpy(label)
    else:  # No augmentation
        lbl = torch.zeros(50 * 5)
        try:
            tmp = torch.from_numpy(read_truths_args(label, 8.0 / clip[0].width).astype('float32'))
        except Exception:
            tmp = torch.zeros(1, 5)

        tmp = tmp.view(-1)
        tsz = tmp.numel()

        if tsz > 50 * 5:
            lbl = tmp[0:50 * 5]
        elif tsz > 0:
            lbl[0:tsz] = tmp
        # label = lbl
        label = fill_truth_detection(label, clip[0].width, clip[0].height, flip=False, dx=0, dy=0, sx=1.0, sy=1.0)

    return clip, label
    # if train:
    #    return clip, label
    # else:
    #    return im_split[0] + '_' +im_split[1] + '_' + im_split[2], clip, label


class DoTA(data.Dataset):
    def __init__(self, args, phase, n_frames=16, combined_bbox=False):
        self.args = args
        self.root = self.args.root
        self.data_root = self.args.data_root if phase == "train" else self.args.val_data_root
        self.sessions_dirs = glob.glob(os.path.join(self.data_root, '*'))
        self.image_size = (224, 224)  # (1280, 720)
        self.overlap = int(self.args.segment_len / 2)
        self.phase = phase
        self.data_frames = os.path.join(self.root, "frames")
        self.data_annotations = os.path.join(self.root, "annotations")
        self.avail_vids = [s.split("/")[-1] for s in
                           glob.glob(os.path.join(self.data_frames, "*"))]
        self.data_list = []
        self.transform = None
        self.target_transform = None
        self.combined_bbox = combined_bbox

        per_class_label_dict = {'0': [],
                                '1': [],
                                '2': [],
                                '3': [],
                                '4': [],
                                '5': [],
                                '6': [],
                                '7': [],
                                '8': [],
                                '9': [],
                                '10': [],
                                '11': [],
                                '12': [],
                                '13': [],
                                '14': [],
                                '15': [],
                                '16': [],}

        if phase == "test":
            random.seed(10)
        label_list = []
        for session_dir in tqdm(self.sessions_dirs):
            vid = session_dir.split('/')[-1].split('.')[0]
            if vid not in self.avail_vids:
                continue

            with open(os.path.join(self.data_annotations, f"{vid}.json")) as f:
                json_data = json.load(f)

            anomaly_start = max(json_data["anomaly_start"], n_frames)
            anomaly_end = json_data["anomaly_end"]
            if anomaly_end < n_frames:
                continue

            possible_end_frames = [
                d['frame_id'] for d in json_data['labels']
                if anomaly_start <= d['frame_id'] <= anomaly_end
                   and len(d['objects']) > 0
            ]

            n_anomaly_frames = len(possible_end_frames)
            if n_anomaly_frames == 0:
                continue

            anomaly_frame_info = [d for d in json_data['labels'] if d['frame_id'] in possible_end_frames]

            anomaly_box_list = []
            object_list = []

            if not combined_bbox:
                label = anomaly_frame_info[0]['accident_id']
                self.data_list.append((vid, possible_end_frames, label, anomaly_frame_info))
                for ob in anomaly_frame_info[0]['objects']:
                    per_class_label_dict[str(ob['category ID'])].append(ob['category'])
                # if label == 10:
                #     print(vid)
            else:
                for frame in anomaly_frame_info:
                    label = anomaly_frame_info[0]['accident_id']
                    label_list.append(frame['objects'][0]['category ID'])
                    if combined_bbox:
                        bbox = [100000000000, 10000000000, -1, -1]
                        for obj in frame['objects']:
                            bbox[0] = min(obj['bbox'][0], bbox[0])
                            bbox[1] = min(obj['bbox'][1], bbox[1])
                            bbox[2] = max(obj['bbox'][2], bbox[2])
                            bbox[3] = max(obj['bbox'][3], bbox[3])
                        anomaly_box_list.append(bbox)
                        self.data_list.append((vid, possible_end_frames, label, anomaly_box_list))
                    else:
                        object_list.append(frame['objects'])
                        if label == 0:
                            print(0) # NO LABEL IS 0!!! this never runs
                        # self.data_list.append((vid, possible_end_frames, label, object_list))
                        self.data_list.append((vid, possible_end_frames, label, anomaly_frame_info))
        print('Total videos: {}'.format(len(self.data_list)))
    def frame_code(self, frame_id):
        if frame_id < 10:
            return f"00000{frame_id}"
        elif frame_id < 100:
            return f"0000{frame_id}"
        elif frame_id < 1000:
            return f"000{frame_id}"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        vid, frames, anomaly_label, boxes = self.data_list[index]

        end_frame_idx = random.choice(list(range(len(frames))))
        end_frame = frames[end_frame_idx]
        bbox = boxes[end_frame_idx]['objects']
        frames = list(range(end_frame-16, end_frame))
        frames = [f"{self.data_frames}/{vid}/{self.frame_code(f)}.jpg" for f in frames]

        if self.combined_bbox:
            lbl = [anomaly_label + 1] + bbox # anomaly label and joined bounding box
            #print(lbl) # [6, 785.2962515114874, 288.174123337364, 1009.9153567110037, 410.0604594921403]
                        # [6, 0.8209806157354619, 0, 805.3819840364881, 643.6488027366021]
            if self.phase == "train":  # or self.phase == "test": # For Training
                jitter = 0.2
                hue = 0.1
                saturation = 1.5
                exposure = 1.5
                clip, label = load_data_detection(frames, lbl, True, self.image_size, jitter, hue, saturation, exposure)
            elif self.phase == "test":
                clip, label = load_data_detection(frames, lbl, False, self.image_size, jitter=0.2, hue=0.1, saturation=1.5,
                                              exposure=1.5)
        else:
            # all object labels and individual bounding boxes
            obj_lbl = []
            for box in bbox:
                obj_lbl.append(box['category ID'])
                obj_lbl = obj_lbl + box['bbox']
            # print(lbl) # [6, {'obj_track_id': 0, 'bbox': [378.0797636632201, 355.2141802067947, 1208.685376661743, 718.9364844903988],
                         # 'category': 'car', 'category ID': 3, 'trunc': True}]
            if self.phase == "train":  # or self.phase == "test": # For Training
                jitter = 0.2
                hue = 0.1
                saturation = 1.5
                exposure = 1.5
                clip, obj_lbl = load_data_detection(frames, obj_lbl, True, self.image_size, jitter, hue, saturation, exposure)
            elif self.phase == "test":
                clip, obj_lbl = load_data_detection(frames, obj_lbl, False, self.image_size, jitter=0.2, hue=0.1, saturation=1.5,
                                              exposure=1.5)

        if self.target_transform is not None:
            anomaly_label = self.target_transform(anomaly_label)

        clip = torch.stack([ToTensor()(img) for img in clip], dim=1)
        if self.combined_bbox:
            if self.phase == "train":
                return (clip, anomaly_label)
            else:
                frame_idx = vid
                return (frame_idx, clip, anomaly_label)
        else:
            if self.phase == "train":
                return (clip, obj_lbl, anomaly_label+1)
            else:
                frame_idx = vid
                return (frame_idx, clip, obj_lbl, anomaly_label+1)