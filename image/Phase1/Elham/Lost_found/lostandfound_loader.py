import os
import torch
import numpy as np
#import scipy.misc as m
from PIL import Image
from torch.utils import data
import matplotlib.pyplot as plt

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class lostandfoundLoader(data.Dataset):
    """lostandfoundLoader

        data download link: http://www.6d-vision.com/lostandfounddataset
    """

    colors = [ [  0,   0,   0],
        [128, 64, 128],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
        [0, 0, 142],
    ]

    label_colours = dict(zip(range(37), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
        "lostandfound": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        augmentations=None,
        img_norm=True,
        version="lostandfound",
        test_mode=False,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """

        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 37
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}
        if self.root:
            self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
            self.annotations_base = os.path.join(self.root, "gtCoarse", self.split)

            self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 31, 33, 34, 36, 37, 38, 39]
        self.valid_classes = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            32,
            35,
            40,
            41,
            42,
            43,
        ]
        self.class_names = [
            "unlabelled",
            "free",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "31",
            "34",
            "39",
            "40",
            "41",
            "42",
        ]

        self.ignore_index = 0
        self.class_map = dict(zip(self.valid_classes, range(1,37)))

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        #print(img_path)
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtCoarse_labelIds.png",
        )
        #print(lbl_path)
        img = Image.open(img_path).convert('RGB') #m.imread(img_path)
        newsize = (self.img_size[0], self.img_size[1])
        img = img.resize(newsize)
        img = np.array(img, dtype=np.uint8)

        lbl = Image.open(lbl_path) #m.imread(lbl_path)
        lbl = lbl.resize(newsize)
        lbl1 =np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(lbl1)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """

        # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        #lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


if __name__ == "__main__":


    local_path = "/media/rtuser1/easystore/OpenDataSet/lostandfound"
    dst = lostandfoundLoader(local_path, is_transform=True, augmentations=None)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        #import pdb

        #pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            arr_labels=labels.numpy()[j]
            axarr[j][1].imshow(dst.decode_segmap(arr_labels))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
