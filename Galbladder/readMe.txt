
To work with this project, it is a prerequisite to install detectron2 and it's dependencies. The instrustions for this is in the website https://github.com/facebookresearch/detectron2

Training for classes for personal project in detectron2:

Detectron2 has a prespecified workflow for common machine learning datasets such as COCO, Pascal VOC, and cityscapes. There are also arrangements for the tasks that can be performed within these datasets such as object detection, and the different types of segmentation (see "detectron2/configs/" folder).  However, there are some additions required for the sake of using detectron2 in custom projects and external datasets. In our case, we are trying to use detectron2 to detect the No-Go-Zone in a laparoscopic surgery. 

To enable this, we first had to register the dataset under MetadataCatalog and DatasetCatalog. We need the dataset to be in a specific list-of-dictionaries format (keys=filename, imageId, height, width, annotations). Next, we simply had to call the DatasetCatalog and MetadataCatalog objects to register the training and evaluation parts of the dataset and the classes within it. A tutorial for this can be found in the official collaboratory page for detectron2 which is in "https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5" in the "Train on a custom dataset" section.

Also, to perform periodic evaluation during training, we followed the recommendations given in the build_evaluator method from the defaults.py module. We added a class called MyTrainer which inherits DefaultTrainer from the detectron2.engine location. Another addition we had to make was the LossEvalHook class which inherits the HookBase class from the detectron2.engine.hooks location. This enables us to register our own events on which the evaluation steps should automatically take place during training.

Changes made to default demo workflow:

Another change required was for creating an output video. In the VisualizationDemo class of predictor.py, we have to make sure the metadata it picks up is from our dataset, so we had to use the line 

self.metadata = MetadataCatalog.get("bladder_val") 

instead of the old line used to set the self.metadata variable in the __init__ part. To make the colour the same for all the frames, we had to add a line in video_visualizer.py which hard-codes the colour by making a list repeating the same RGB code. For example, we added "colors=[[0,0.502,0.502]]*10" after the line where it gets the colour.

For video output smoothing, we added some lines of code where it sums up the area of the segment predictions over an interval and outputs it to the video so the predictions look stable for that amount of frames. These changes were made in video_visualizer.py and predictor.py. These changes mainly included adding a buffer value to choose the interval over the averaging of prediction mask area, a way to retain the masks until the buffer criteria is met, and a signal to the draw_instance_predictions method in video_visualizer.py.

Instructions to run training and inference:

To run the training, there is an slrm file called runt4v1Detectron.slrm. Essentially that file runs the command:

python DetectronGBScript.py 
--wd <weight decay> 
--ims <images per batch>
--lr <learning rate> 
--e <epochs> 
--roi <regions of interest to check per image>
--d <output directory>

Example:

python DetectronGBScript.py 
--wd 0.0001 
--ims 8 
--lr 0.00001 
--e 30 
--roi 512 
--d 'detectron2/output/0.0001_8_0.00001_30/'

Also, the command in the jupyter notebook that we used from the root directory to run the inference on video and save the output is in the format:

%run detectron2/demo/demo.py 
--config-file <config file location> 
--video-input <input video location> 
--confidence-threshold <pct confidence needed for segment prediction to be included in output> 
--output <desired output location>  
--opts MODEL.WEIGHTS <trained model location> MODEL.ROI_HEADS.NUM_CLASSES <number of classes in your dataset>

Example:

%run detectron2/demo/demo.py 
--config-file detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml 
--video-input video64.mp4 
--confidence-threshold 0.7 
--output video-outputBigger2.mkv
--opts MODEL.WEIGHTS './output/model_final-Confident20210818.pth' MODEL.ROI_HEADS.NUM_CLASSES 1

Step By Step Tutorial:

For a step by step notebook tutorial, go to Detectron2StepByStep.ipynb in this folder and run the notebook cells

Set Up Instructions For Step-By-Step tutorial:

To run the step-by-step training, 
- Put the JSON annotation files in the "<root>/GallbladderFiles/" folder. 
- The images from the JSON file annotations need to be in the exact location as in the JSON file with respect to the root 	  of this project. For example, there are 2 folders where the images from the JSON file are specified, "archive", and "CTC_Frames". 
- The segmentation masks used for training also have to be stored in the "<root>/gallBladder/masks/" folder
- Further description of the steps are in the notebook itself.

