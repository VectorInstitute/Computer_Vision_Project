import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
from argparse import Namespace
from dataset_factory.dota import DoTA
import seaborn as sns


num_anchors = 9
# image_size_wh = (224,224)
image_size_wh = (1280,720)

sns.set()  # for plot styling

with open("cfg/dota_config.yaml", 'r') as f:
    args = yaml.load(f)
args = Namespace(**args)

if not os.path.exists('/scratch/ssd002/dataset_factory/cv_project/Detection-of-Traffic-Anomaly/dataset'):
    args.root = '/home/matthew/Desktop/Datasets/DoTA'
    args.data_root = '/home/matthew/Desktop/Datasets/DoTA/DoTA_fol_train_data'
    args.val_data_root = '/home/matthew/Desktop/Datasets/DoTA/DoTA_fol_val_data'
    args.label_file = '/home/matthew/Desktop/Datasets/DoTA/metadata_val.json'
    args.train_split = '/home/matthew/Desktop/Datasets/DoTA/train_split.txt'
    args.val_split = '/home/matthew/Desktop/Datasets/DoTA/val_split.txt'
    args.img_dir = '/home/matthew/Desktop/Datasets/DoTA/frames'

d = DoTA(args, phase='train', n_frames=16, combined_bbox=True)


# RESIZE TO 224

lbls = [i[2] for i in d.data_list]

w, h = [], []

for lbl in lbls:
    w1 = lbl[2] - lbl[0]
    h1 = lbl[3] - lbl[1]

    # RESIZE
    if image_size_wh[0] != 1280 or image_size_wh[1] != 720:
        w2 = image_size_wh[0] * w1 / 1280
        h2 = image_size_wh[1] * h1 / 720
    else:
        w2 = w1
        h2 = h1
    w.append(w2)
    h.append(h2)

    # w.append(lbl[2] - lbl[0])
    # h.append(lbl[3] - lbl[1])

w = np.asarray(w)
h = np.asarray(h)

x = [w, h]
x = np.asarray(x)
x = x.transpose()
##########################################   K- Means
##########################################

from sklearn.cluster import KMeans

kmeans3 = KMeans(n_clusters=num_anchors)
kmeans3.fit(x)
y_kmeans3 = kmeans3.predict(x)

##########################################
centers3 = kmeans3.cluster_centers_

yolo_anchor_average = []
for ind in range(num_anchors):
    yolo_anchor_average.append(np.mean(x[y_kmeans3 == ind], axis=0))

yolo_anchor_average = np.array(yolo_anchor_average)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans3, s=2, cmap='viridis')
plt.scatter(yolo_anchor_average[:, 0], yolo_anchor_average[:, 1], c='red', s=50);
yoloV3anchors = yolo_anchor_average
yoloV3anchors[:, 0] = yolo_anchor_average[:, 0] / 1280 * 608
yoloV3anchors[:, 1] = yolo_anchor_average[:, 1] / 720 * 608
yoloV3anchors = np.rint(yoloV3anchors)
fig, ax = plt.subplots()
for ind in range(num_anchors):
    rectangle = plt.Rectangle((304 - yoloV3anchors[ind, 0] / 2, 304 - yoloV3anchors[ind, 1] / 2), yoloV3anchors[ind, 0],
                              yoloV3anchors[ind, 1], fc='b', edgecolor='b', fill=None)
    ax.add_patch(rectangle)
ax.set_aspect(1.0)
plt.axis([0, 608, 0, 608])
plt.show()
yoloV3anchors.sort(axis=0)
print("Your custom anchor boxes for the original image space are {}".format(yoloV3anchors))
x = np.array([[0.70458, 1.18803], [1.26654, 2.55121], [1.59382, 4.08321], [2.30548, 4.94180], [3.52332, 5.91979]])
print("YOWO boxes for 224 image space are {}".format(x*32))
print('>>>')
print('>>>')
print('>>>')
print("Your custom anchor boxes for 224 image space to a 7x7 feature map (i.e., 32x reduction) are {}".format(yoloV3anchors/32))
# YOWO anchor boxes
x = np.array([[0.70458, 1.18803], [1.26654, 2.55121], [1.59382, 4.08321], [2.30548, 4.94180], [3.52332, 5.91979]])
print("YOWO boxes for 224 image space to a 7x7 feature map (i.e., 32x reduction) are {}".format(x))
