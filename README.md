## YOLOX：You Only Look Once Implementation of target detection model in Tensorflow2
---

## Content
1. [Performance](#performance)
2. [Achievement](#achievement)
3. [Environment](#environment)
4. [TricksSet](#tricksset)
5. [Download](#download)
6. [How2train](#how2train)
7. [How2predict](#how2predict)
8. [How2eval](#how2eval)
9. [Reference](#reference)

## Performance
| training dataset | weight file name | testing dataset | input image size | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| COCO-Train2017 | [yolox_tiny.h5](https://github.com/bubbliiiing/yolox-tf2/releases/download/v1.0/yolox_tiny.h5) | COCO-Val2017 | 640x640 | 34.7 | 53.6 
| COCO-Train2017 | [yolox_s.h5](https://github.com/bubbliiiing/yolox-tf2/releases/download/v1.0/yolox_s.h5) | COCO-Val2017 | 640x640 | 39.2 | 58.7
| COCO-Train2017 | [yolox_m.h5](https://github.com/bubbliiiing/yolox-tf2/releases/download/v1.0/yolox_m.h5) | COCO-Val2017 | 640x640 | 46.1 | 65.2
| COCO-Train2017 | [yolox_l.h5](https://github.com/bubbliiiing/yolox-tf2/releases/download/v1.0/yolox_l.h5) | COCO-Val2017 | 640x640 | 49.3 | 68.1
| COCO-Train2017 | [yolox_x.h5](https://github.com/bubbliiiing/yolox-tf2/releases/download/v1.0/yolox_x.h5) | COCO-Val2017 | 640x640 | 50.5 | 69.2

## Achievement
- [x] Backbone Feature Extraction Network: Focus network structure is used.
- [x] Classification and regression layer: Decoupled Head, in YoloX, Yolo Head is divided into two parts of classification and regression, and they are integrated together in the final prediction.
- [x] Tips for training: Mosaic data enhancement, CIOU (the original version is IOU and GIOU, the CIOU effect is similar, both are IOU series, or even updated), learning rate cosine annealing decay.
- [x] Anchor Free: don't use a priori box
- [x] SimOTA: Dynamically match positive samples for targets of different sizes.

## Environment
tensorflow-gpu==2.2.0  

## TricksSet
Under the train.py file:
1. The mosaic parameter can be used to control whether to implement Mosaic data enhancement.
2. Cosine_scheduler can be used to control whether to use learning rate cosine annealing decay.

## Download
The weights required for training can be downloaded from Baidu Netdisk.
Link: https://pan.baidu.com/s/1t6wgcSP85do1Y4lqVCVDMw
Extraction code: m25d

The download address of the VOC data set is as follows, which already includes the training set, test set, and validation set (same as the test set), no need to divide again:
Link: https://pan.baidu.com/s/1YuBbBKxm2FGgTU5OfaeC5A
Extraction code: uack

## How2train
### a. Training VOC07+12 dataset
1. Dataset Preparation
**This article uses VOC format for training. Before training, you need to download the VOC07+12 data set, decompress it and put it in the root directory**

2. Processing of datasets
Modify annotation_mode=2 in voc_annotation.py, and run voc_annotation.py to generate 2007_train.txt and 2007_val.txt in the root directory.

3. Start network training
The default parameters of train.py are used to train the VOC dataset, and the training can be started directly by running train.py.

4. Prediction of training results
Training result prediction requires two files, yolo.py and predict.py. We first need to go to yolo.py to modify model_path and classes_path, these two parameters must be modified.
**model_path points to the trained weights file in the logs folder.
classes_path points to the txt corresponding to the detection category. **
After completing the modification, you can run predict.py for detection. After running, enter the image path to detect. 

### b. Train your own dataset
1. Dataset Preparation
**This article uses the VOC format for training. Before training, you need to make a data set yourself.**
Before training, put the label file in the Annotation under the VOC2007 folder under the VOCdevkit folder.
Before training, put the image files in JPEGImages under the VOC2007 folder under the VOCdevkit folder.

2. Processing of datasets
After completing the placement of the dataset, we need to use voc_annotation.py to get 2007_train.txt and 2007_val.txt for training.
Modify the parameters in voc_annotation.py. For the first training, only classes_path can be modified, and classes_path is used to point to the txt corresponding to the detected category.
When training your own data set, you can create a cls_classes.txt by yourself, and write the categories you need to distinguish in it.
The content of the model_data/cls_classes.txt file is:
````python
cat
dog
...
````
Modify the classes_path in voc_annotation.py to correspond to cls_classes.txt, and run voc_annotation.py.

3. Start network training
**There are many parameters for training, which are all in train.py. You can read the comments carefully after downloading the library. The most important part is still the classes_path in train.py. **
**classes_path is used to point to the txt corresponding to the detection category, which is the same as the txt in voc_annotation.py! Training your own dataset must be modified! **
After modifying the classes_path, you can run train.py to start training. After training multiple epochs, the weights will be generated in the logs folder.

4. Prediction of training results
Training result prediction requires two files, yolo.py and predict.py. Modify model_path and classes_path in yolo.py.
**model_path points to the trained weights file in the logs folder.
classes_path points to the txt corresponding to the detection category. **
After completing the modification, you can run predict.py for detection. After running, enter the image path to detect.

## How2predict
### a. Use pre-trained weights
1. After downloading the library, unzip it, download yolo_weights.pth on Baidu network disk, put it in model_data, run predict.py, enter
````python
img/street.jpg
````
2. Set in predict.py to perform fps test and video video detection.
### b. Use the weights trained by yourself
1. Follow the training steps to train.
2. In the yolo.py file, modify the model_path and classes_path in the following sections to correspond to the trained files; **model_path corresponds to the weight file under the logs folder, and classes_path is the class corresponding to model_path**.
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Be sure to modify model_path and classes_path when using your own trained model for prediction!
    #   model_path points to the weights file under the logs folder, classes_path points to the txt under model_data
    #   If the shape does not match, pay attention to the modification of the model_path and classes_path parameters during training
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolox_s.h5',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   The size of the input image, which must be a multiple of 32.
    #---------------------------------------------------------------------#
    "input_shape"       : [640, 640],
    #---------------------------------------------------------------------#
    #   The version of YoloX used. s, m, l, x
    #---------------------------------------------------------------------#
    "phi"               : 's',
    #---------------------------------------------------------------------#
    #   Only prediction boxes with scores greater than confidence will be kept
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   nms_iou size used for non-maximum suppression
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   Maximum number of prediction boxes
    #---------------------------------------------------------------------#
    "max_boxes"         : 100,
    #---------------------------------------------------------------------#
    #   This variable is used to control whether to use letterbox_image to resize the input image without distortion,
    #   After many tests, it is found that the direct resize effect of closing letterbox_image is better
    #---------------------------------------------------------------------#
    "letterbox_image"   : True,
}
```
3. Run predict.py and enter
````python
img/street.jpg
````
4. Setting in predict.py can perform fps test and video video detection.

## How2eval
### a. Test set for evaluating VOC07+12
1. This paper uses the VOC format for evaluation. VOC07+12 has already divided the test set, there is no need to use voc_annotation.py to generate the txt under the ImageSets folder.
2. Modify model_path and classes_path in yolo.py. **model_path points to the trained weights file in the logs folder. classes_path points to the txt corresponding to the detection category. **
3. Run get_map.py to get the evaluation result, which will be saved in the map_out folder.

### b. Evaluate your own dataset
1. This paper uses the VOC format for evaluation.
2. If the voc_annotation.py file has been run before training, the code will automatically divide the data set into training set, validation set and test set. If you want to modify the proportion of the test set, you can modify the trainval_percent in the voc_annotation.py file. trainval_percent is used to specify the ratio of (training set + validation set) to test set, by default (training set + validation set): test set = 9:1. train_percent is used to specify the ratio of training set to validation set in (training set + validation set), by default training set:validation set = 9:1.
3. After dividing the test set with voc_annotation.py, go to the get_map.py file to modify the classes_path. The classes_path is used to point to the txt corresponding to the detection category. This txt is the same as the txt during training. Evaluating your own datasets must be modified.
4. Modify model_path and classes_path in yolo.py. **model_path points to the trained weights file in the logs folder. classes_path points to the txt corresponding to the detection category. **
5. Run get_map.py to get the evaluation result, which will be saved in the map_out folder.

## Reference
https://github.com/Megvii-BaseDetection/YOLOX
