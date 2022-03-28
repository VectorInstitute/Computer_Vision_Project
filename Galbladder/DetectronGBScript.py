from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os
import json
import datetime

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--p', help='eval period', default=200)
parser.add_argument('--wd', help='weight decay', default=0.0001)
parser.add_argument('--w', help='number of workers for dataloader', default=8)
parser.add_argument('--ims', help='images per batch', default=4)
parser.add_argument('--wi', help='warmup iterations to increase lr', default=0)
parser.add_argument('--lr', help='base lr', default=0.000025)
parser.add_argument('--e', help='epochs', default=3)
parser.add_argument('--roi', help='roi heads per image', default=1024)
parser.add_argument('--n', help='number of classes', default=1)
parser.add_argument('--d', help='model output dir')

args = parser.parse_args()

if os.path.exists('parameters.txt'):
    mode='a'
else:
    mode='w'

with open('parameters.txt',mode) as f:
    f.write('\n'+str(args) + ' ' + str(datetime.datetime.now()))

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, jsonDir, scale=1, mask_suffix='',val=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.jsonDir=jsonDir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.val=val
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        print('initializing dataloader..')
        self.frameListUnfiltered = []
        imgsPerPathTemp=[]
        for r in imgs_dir:
            imgsPerPathTemp.append([f for f in [os.path.join(currentFolder, currentFile) for currentFolder, otherDirs, files in os.walk(os.path.expanduser(r)) for currentFile in files] if f[-8:]=='endo.png' or f[-4:]=='.jpg'])
        imgsPerPath=[]
        for path in imgsPerPathTemp:
            for img in path:
                self.frameListUnfiltered.append(img)
                
        print('matching images in json files to data')
        #make sure image in gallBladder annotation list
        self.frameList, self.coordsDict=self.matchImgFromJSON(self.jsonDir, self.frameListUnfiltered)
        #===        
        total=[os.path.basename(x) for x in self.frameList]
        unique=list(set(total))
        #==
        print('length total filepaths', len(total), 'length unique filenames', len(unique))
        logging.info(f'Creating dataset with {len(self.frameList)} examples')
    
    def __len__(self):
        #==return len(self.ids)
        return len(self.frameList)


    @classmethod
    def preprocess(cls, pil_img, scale, isMask=False):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)#==
        assert newW > 0 and newH > 0, 'Scale is too small'
        #pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
 
        if img_trans.max() > 1 and not isMask:#don't want to normalize the mask values
            img_trans = img_trans / 255

        return img_trans


    def __getitem__(self, i):
        #dont need mask for detectron only need coords
        #mask_file = self.frameList[i][:-4]+'_watershed_mask.png' #==glob(self.masks_dir + idx + self.mask_suffix + '.*')
#         mask_file = os.path.join(self.masks_dir, 
#                                  os.path.basename(os.path.dirname(self.frameList[i]) + '_' + os.path.basename(self.frameList[i]))) #==glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = self.frameList[i] #==glob(self.imgs_dir + idx + '.*')
        
        coords=np.array(self.coordsDict[img_file]).squeeze()
        
        coords=np.concatenate((coords,np.array([[coords[0,0]],[coords[1,0]]])),axis=1)#to close contour
        
#         mask = Image.open(mask_file)#==
        img = Image.open(img_file)#==
        #print('mask ', np.unique(np.array(mask)))
        #======took this out because we dont need mask anyways
#         assert img.size == mask.size, \
#             f'Image and mask {i} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
#         mask = self.preprocess(mask, self.scale,True)
#         mask = self.convertToSingle(mask)
#         maskOhe=self.make_ohe(mask)#for validation only to help dice coeffiecient for multi-channel#just added this as the other way might be confusing

        if self.val==True:
#             maskOhe=self.make_ohe(mask)#for validation only to help dice coeffiecient for multi-channel
            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
#                 'mask': torch.from_numpy(mask).type(torch.FloatTensor),
#                 'maskOhe': torch.from_numpy(maskOhe).type(torch.FloatTensor),
                'filename': img_file,
                'coords': coords
            }
        

        
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
#             'mask': torch.from_numpy(mask).type(torch.FloatTensor),
#             'maskOhe': torch.from_numpy(maskOhe).type(torch.FloatTensor),#just added this as the other way might be confusing
            'filename': img_file,
            'coords': coords
        }
#==
    def make_ohe(self,mskOrig):
        """
        converts image with labels into the one-hot encoded format
        (img[...,] --> img[..., N_CLASSES])
        mode : `catId` or `trainId`
        """ 
        classDict={50:0, 11:1, 21:2, 13:3, 12:4, 31:5, 23:6, 24:7, 25:8, 32:9, 22:10, 33:11, 5:12, 255:0}
        classes = classDict.values()#13 classes including bg     
        #ohe_labels = np.zeros(mskOrig.shape[:2] + (len(classes),))
        ohe_labels = np.zeros((len(classes)-1,mskOrig.shape[1],mskOrig.shape[2]))#len(classes) -1 because 2 of them represent bg in our case
        for c in classes:
            ys, xs = np.where(mskOrig[0,...] == c)
            ohe_labels[c, ys, xs] = 1
        return ohe_labels.astype(int)
    
    def convertToSingle(self,RGBMask): 
        classDict={50:0, 11:1, 21:2, 13:3, 12:4, 31:5, 23:6, 24:7, 25:8, 32:9, 22:10, 33:11, 5:12, 255:0}
        for c in classDict.keys():
            RGBMask[0,np.where(RGBMask[0,:,:]==c)[0],np.where(RGBMask[0,:,:]==c)[1]]=classDict[c]#already converted to CHW so [0,:,:] #also in our dataset, all channels have the same value in the mask for easy processing
        RGBMask=RGBMask[0].reshape((1, RGBMask.shape[1],RGBMask.shape[2] ))
        if np.max(RGBMask)>12:
            RGBMask[np.where(RGBMask>12)]=0#just set outliers as background
            
        return RGBMask
    #find the location of image  with same size and name
    def findImgSize(self, imgFile,fullList, sizeCheck):
        for fullPath in fullList:
            if os.path.basename(fullPath)==imgFile:
                sizeObtained=os.path.getsize(fullPath)
                if sizeObtained==sizeCheck:
#                     print(print('size matched for ', imgFile, 'size in json ', sizeCheck, 'size in path ', os.path.getsize(fullPath)))
                    return [fullPath, sizeObtained]#found file
#                 else:
#                     print('size not matched for ', imgFile, 'size in json ', sizeCheck, 'size in path ', os.path.getsize(fullPath))
        
        return False #didn't find file

    
    def getfilesReq(self, coordsPerFile, pathsReq):

        filesReq=[]#store filenames with atleast one annotation

        #loop through all files per json and add all the ones which have any coordinates
        for path in pathsReq:      

            if len(coordsPerFile[path][0])>0:#make sure atleast one annotation
                filesReq.append(path)

        return filesReq    
    def checkNewFormat(self,perJSON,file,coordsPerFile,pathsReq):
        
        fileRead=perJSON[file]
        
        #manually mapped to video name
        fileToFolderDict={'NOGO_annotated09.json':'Vid24',
                         'via_project_21Apr2021_17h46m30s nogo 100.json':'video_01_00080',#short file
                         'via_project_24Apr2021_11h02m19s NOGO138.json':'video_01_00080',#short file
                         'via_project_26Apr2021_08h44m21s nogo159.json':'video_01_00080',#short file
                         'via_project_19Apr2021_12h38m19s nogo.json':'video_01_00080',
                         'NOGO1179via_project_23May2021_12h39m22s.json':'video_35_00940',
                         'via_project_25Apr2021_12h43m56snogo147.json':'video_01_00080',#short
                         'via_project_31Jul2021_22h01m03s.json':'Vid03',#can't seem to find a few files in this one
                         'NOGO8625via_project_16Jul2021_16h39m14s.json':'Vid24',
                         'NOGO14000via_project_06Jul2021_12h40m23s.json':'Vid24'#can't seem to find a few files in this one
                         }

        lsJsonInd=list(fileRead['file'].keys())#to find filename

        for index in range(len(lsJsonInd)):                        
            fileListed=fileRead['file'][lsJsonInd[index]]['fname']#file listed in json
            metadataKeys=list(fileRead['metadata'].keys())#to find coordinates
            vertices=[]#start xy coordinate list
            if len(fileRead['metadata'][metadataKeys[index]]['xy'])>1:#make sure something is there in coords
                rawPoints=fileRead['metadata'][metadataKeys[index]]['xy'][1:]
                prev=0#starting index for coords

                for i in range(2,len(rawPoints),2):#starts from 2 so we start from prev=0 and i=2
                    vertices.append(rawPoints[prev:i])#make list of x,y coordinates
                    prev=i

                if fileListed.find('_no-go')!=-1:
                    fileName=fileListed[:fileListed.find('_no-go')]+'.jpg'#take out "no-go" part if exists and make it a regular filename
                else:
                    fileName=fileListed
                if 'endo' in fileName:#toggle between 2 folders found
                    pathReq='/scratch/ssd002/home/skhalid/Pytorch-UNet/archive/'+fileToFolderDict[file]+'/'+fileName
                else:                
                    pathReq='/scratch/ssd002/home/skhalid/Pytorch-UNet/CTC_Frames/'+fileToFolderDict[file]+'/'+fileName

                if os.path.exists(pathReq):
                    coordsPerFile[pathReq]=vertices#store list of x,y coordinates
                    pathsReq.append(pathReq)
                else:
                    print('DNE: ', pathReq)

        return coordsPerFile, pathsReq
        
        
    def getCoords(self, perJSON, frameList):
        coordsPerFile=dict()#store coordinate annotations (polygon vertices) with filename as key
        fileCount=0
        notFound=0
        pathsReq=[]
        dups=0
        for file in list(perJSON.keys()):     #loop through json file data
#             print('checking for coordinates in file name:', file)
            if '_via_img_metadata' not in perJSON[file].keys():
                print('skipping new json format: ', file)
                #coordsPerFile, pathsReq=self.checkNewFormat(perJSON,file,coordsPerFile,pathsReq)
                continue
            for key2 in perJSON[file]['_via_img_metadata'].keys():  
                sizeReq=perJSON[file]['_via_img_metadata'][key2]['size']#for comparison to find proper path in "findImgSize"
                fName=perJSON[file]['_via_img_metadata'][key2]['filename']#for comparison to find proper path in "findImgSize"
                pathReq=self.findImgSize(fName, frameList, sizeReq)#find the location of image  with same size and name
                
       
                if pathReq==False:
                    if notFound<2 and '_' not in fName:
                        print('file not found', fName, sizeReq)
                    if '_' in fName:
                        print('file not found', fName, sizeReq)
                        
                    notFound+=1
                    continue
                else:
                    fullPath=pathReq[0]
                    if fullPath in pathsReq:#maybe has more than 1 region
                        dups+=1                        
#                         print('dup', pathReq[0])
                    else:
                        pathsReq.append(fullPath)
                
                coordsPerFile[fullPath]=[] #initialize as list            
                xs=None
                ys=None
                if len(perJSON[file]['_via_img_metadata'][key2]['regions'])==0: #no coords found
                    coordsPerFile[fullPath].append([])#just to note there are no annotations                      
                    continue            
                #store x, y coords for polygon vertices
                xs=perJSON[file]['_via_img_metadata'][key2]['regions'][0]['shape_attributes']['all_points_x']
                ys=perJSON[file]['_via_img_metadata'][key2]['regions'][0]['shape_attributes']['all_points_y']
                coordsPerFile[fullPath].append([xs,ys]) #store x,y per file                            
                            
        print('files not found coords, multiple regions ', notFound, dups) 
        return coordsPerFile, pathsReq

    def matchImgFromJSON(self, dirReq, frameList):
        #====read json, get coords for each file, get files where atleast one set of coords specified====
        perJSON=dict()    
        for file in os.listdir(dirReq):
            with open(dirReq+file) as f:
                perJSON[file] = json.load(f) #get json list
        fileCount=0
        coordsPerFile, pathsReq=self.getCoords(perJSON, frameList)
        print('initial amount of files:', len(list(coordsPerFile.keys())), len(pathsReq), len(list(set(pathsReq))))
#         pathsReq=list(set(pathsReq))
        filesReq=self.getfilesReq(coordsPerFile, pathsReq)        
#         filesReq=pathsReq#maybe there should be coordinates without a segment
        
        print('after filesReq (filters for at least one annotation)', len(filesReq))
        #======================================================

        #===store filename and size, find exact path of file in dataset subfolders and put in a list===

        #===================================================================================
        return filesReq, coordsPerFile

#==    
        
class cholec8KDatasetGallBladderMasks(BasicDataset):#==
    def __init__(self, imgs_dir, masks_dir, jsonDir, scale=1):
        super().__init__(imgs_dir, masks_dir, jsonDir, scale, mask_suffix='_mask')

import sys
if '/scratch/ssd002/home/skhalid/Pytorch-UNet/' not in sys.path:
    sys.path.append('/scratch/ssd002/home/skhalid/Pytorch-UNet/')
    
from unet import UNet
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.datasetGB import BasicDataset
from dice_loss import dice_coeff
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
    
#================
# needs mask in the mask directory with the same name as frame
#================
device='cuda'#could use 'cpu' if needed
root_path = '/scratch/ssd002/home/skhalid' # remember to verify during actual training#==
dirReqImg=[root_path+'/Pytorch-UNet/archive/',
           root_path+'/Pytorch-UNet/CTC_Frames/']

dirReqMsk=root_path+'/Pytorch-UNet/gallBladder/masks/'
dirReqJson=root_path+'/Pytorch-UNet/GallbladderFiles/'#might not be needed for this

DS=cholec8KDatasetGallBladderMasks(dirReqImg, dirReqMsk, dirReqJson)



reqTrain=[]
reqTest=[]
foldersTest=['CTC_Frames', 'video24']
foldersTrain=['video01', 'video18', 'video20', 'video35', 'video55']
for f in DS:    
    folder=os.path.basename(os.path.dirname(os.path.dirname(f['filename'])))
    if folder in foldersTest:
        reqTest.append(f['filename'])
    elif folder in foldersTrain:
        reqTrain.append(f['filename'])
        
len(reqTrain), len(reqTest) 

from detectron2.structures import BoxMode#remember to change to  "from detectron2.structures import BoxMode" when doing detectron
from detectron2.data import MetadataCatalog, DatasetCatalog

dataset_dicts_train = []
dataset_dicts_val = []
c=0

for idx, loaded in enumerate(DS):    
    record = {}    
    height, width = loaded['image'][0].shape
    
    record["file_name"] = loaded['filename']
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width    
    objs = []
#==no for loop because seems to be only one zone for gallbladder==

    px = loaded["coords"][0]
    py = loaded["coords"][1]
    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
    poly = [p for x in poly for p in x]

    obj = {
        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [poly],
        "category_id": 0,
    }

    objs.append(obj)
    
    record["annotations"] = objs
    
    if loaded['filename'] in reqTrain:#splitting training and testing
        dataset_dicts_train.append(record)
    else:
        dataset_dicts_val.append(record)

import json

def readJson(filename):
    # Writing to sample.json
    with open(filename, "r") as inFile:
        fileRead = json.load(inFile)
    return fileRead
def XyxyPoly(polyCoords):
    xyxy=[xy for pair in polyCoords for xy in pair]
    return xyxy
    
# def augDataAppend():
    
aug=readJson("augCoords.json")
totalLength=len(DS)
for idx, file in enumerate(list(aug.keys())):
    recordAug={}
    poly=XyxyPoly(aug[file]['polygon'])
    recordAug['file_name']=file
    recordAug["image_id"] = idx+totalLength
    recordAug["height"] = aug[file]['shape'][0]
    recordAug["width"] = aug[file]['shape'][1]
    objs = []
    obj={
        "bbox": aug[file]['bbox'],
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [poly],
        "category_id": 0,        
    }
    objs.append(obj)
    
    recordAug['annotations']=objs
    
    dataset_dicts_train.append(recordAug)

for d in dataset_dicts_train:
    if d['annotations'][0]['segmentation'][0][-2:]!=d['annotations'][0]['segmentation'][0][:2]:
        print('appending ', d['file_name'])
        d['annotations'][0]['segmentation'][0].append(d['annotations'][0]['segmentation'][0][0])#adding first 2 coordinates to close contour
        d['annotations'][0]['segmentation'][0].append(d['annotations'][0]['segmentation'][0][1])    


DatasetCatalog.register("bladder_train", lambda d="train": dataset_dicts_train)
DatasetCatalog.register("bladder_val", lambda d="val": dataset_dicts_val)

MetadataCatalog.get("bladder_train").set(thing_classes=["no_go_zone"])
MetadataCatalog.get("bladder_val").set(thing_classes=["no_go_zone"])

MetadataCatalog.get("bladder_train").stuff_colors = [(0,128,128)]
MetadataCatalog.get("bladder_val").stuff_colors = [(0,128,128)]

MetadataCatalog.get("bladder_train").stuff_colors = [(0,128,128)]
MetadataCatalog.get("bladder_val").stuff_colors = [(0,128,128)]

MetadataCatalog.get("bladder_train").thing_colors = [(0,128,128)]
MetadataCatalog.get("bladder_val").thing_colors = [(0,128,128)]

MetadataCatalog.get("bladder_train").set(stuff_classes=["no_go_zone"])
MetadataCatalog.get("bladder_val").set(stuff_classes=["no_go_zone"])

bladder_metadata = MetadataCatalog.get("bladder")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks
        
        
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--p', help='eval period', default=200)
# parser.add_argument('--wd', help='weight decay', default=0.0001)
# parser.add_argument('--w', help='number of workers for dataloader', default=8)
# parser.add_argument('--ims', help='images per batch', default=4)
# parser.add_argument('--wi', help='warmup iterations to increase lr', default=0)
# parser.add_argument('--lr', help='base lr', default=0.000025)
# parser.add_argument('--e', help='epochs', default=3)
# parser.add_argument('--roi', help='roi heads per image', default=1024)
# parser.add_argument('--n', help='number of classes', default=1)
# parser.add_argument('--d', help='model output dir')

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("bladder_train",)
cfg.DATASETS.TEST = ("bladder_val",)
cfg.TEST.EVAL_PERIOD = int(args.p)
cfg.DATALOADER.NUM_WORKERS = int(args.w)
cfg.SOLVER.WEIGHT_DECAY = float(args.wd) #trying this out for generalization issues
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = int(args.ims) #number of images for 1 iteration to be complete
cfg.SOLVER.WARMUP_ITERS = int(args.wi)
cfg.SOLVER.BASE_LR = float(args.lr)  # was 0.000025 # pick a good LR
cfg.SOLVER.MAX_ITER = int(len(dataset_dicts_train)/cfg.SOLVER.IMS_PER_BATCH)*int(args.e) #(last number in the eqn is epochs)  #iterations for one epoch * num epochs
# cfg.SOLVER.STEPS = [300, 600]    # for decaying learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(args.roi)   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(args.n)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.SOLVER.CHECKPOINT_PERIOD = 5000
# cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
evaluator = COCOEvaluator(dataset_name="bladder_val",output_dir=os.getcwd()+"/jsonOutput")
val_loader = build_detection_test_loader(cfg, "bladder_val")
# cfg.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": True})#default is false

cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = -1 # was set to 255
cfg.OUTPUT_DIR='./'+args.d

evaluator = COCOEvaluator(dataset_name="bladder_val",output_dir=os.getcwd()+"/jsonOutput")
val_loader = build_detection_test_loader(cfg, "bladder_val")

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg) 

trainer.resume_or_load(resume=False)
trainer.train()
