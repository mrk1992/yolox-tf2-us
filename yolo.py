import colorsys
import os
import time

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from nets.yolo import yolo_body
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox


class YOLO(object):
    _defaults = {
        #------------------------------------------------- -------------------------#
        #   To use your own trained model for prediction, you must modify model_path and classes_path!
        #   model_path points to the weights file under the logs folder, classes_path points to the txt under model_data
        #  
        #   After training, there are multiple weight files in the logs folder, and you can select the validation set with lower loss.
        #   The lower loss of the validation set does not mean that the mAP is higher, it only means that the weight has better generalization performance on the validation set.
        #   If the shape does not match, pay attention to the modification of the model_path and classes_path parameters during training
        #------------------------------------------------- -------------------------#
        "model_path"        : 'model_data/yolox_s.h5',
        "classes_path"      : 'model_data/coco_classes.txt',
        #------------------------------------------------- --------------------#
        #   Enter the size of the image, which must be a multiple of 32.
        #------------------------------------------------- --------------------#
        "input_shape"       : [640, 640],
        #------------------------------------------------- --------------------#
        #   The version of YoloX used. tiny, s, m, l, x
        #------------------------------------------------- --------------------#
        "phi"               : 's',
        #------------------------------------------------- --------------------#
        #   Only prediction boxes with scores greater than confidence will be kept
        #------------------------------------------------- --------------------#
        "confidence"        : 0.5,
        #------------------------------------------------- --------------------#
        #   nms_iou size used for non-maximum suppression
        #------------------------------------------------- --------------------#
        "nms_iou"           : 0.3,
        #------------------------------------------------- --------------------#
        #   Maximum number of prediction boxes
        #------------------------------------------------- --------------------#
        "max_boxes"         : 100,
        #------------------------------------------------- --------------------#
        #   This variable is used to control whether to use letterbox_image to resize the input image without distortion,
        #   After many tests, it is found that the direct resize effect of closing letterbox_image is better
        #------------------------------------------------- --------------------#
        "letterbox_image"   : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   Get the number of kinds and a priori boxes
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        #---------------------------------------------------#
        #   Picture frame set different colors
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        
        self.generate()

    #---------------------------------------------------#
    #   Load model
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        self.yolo_model = yolo_body([None, None, 3], num_classes = self.num_classes, phi = self.phi)
        self.yolo_model.load_weights(self.model_path)
        print('{} model, and classes loaded.'.format(model_path))
        #---------------------------------------------------------#
        #   In the DecodeBox function, we will post-process the prediction results
        #   The content of post-processing includes decoding, non-maximum suppression, threshold filtering, etc.
        #---------------------------------------------------------#
        self.input_image_shape = Input([2,],batch_size=1)
        inputs  = [*self.yolo_model.output, self.input_image_shape]
        outputs = Lambda(
            DecodeBox, 
            output_shape = (1,), 
            name = 'yolo_eval',
            arguments = {
                'num_classes'       : self.num_classes, 
                'input_shape'       : self.input_shape, 
                'confidence'        : self.confidence, 
                'nms_iou'           : self.nms_iou, 
                'max_boxes'         : self.max_boxes, 
                'letterbox_image'   : self.letterbox_image
             }
        )(inputs)
        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes
    #---------------------------------------------------#
    #   Detect pictures
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   Convert the image to an RGB image here to prevent an error in the prediction of the grayscale image.
        #   The code only supports prediction of RGB images, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for identification
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add the batch_size dimension and normalize it
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        #---------------------------------------------------------#
        #   Feed the image into the network to make predictions!
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        #---------------------------------------------------------#
        #   Set font and border thickness
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        #---------------------------------------------------------#
        #   Image drawing
        #---------------------------------------------------------#
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box             = out_boxes[i]
            score           = out_scores[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        #   Convert the image to an RGB image here to prevent an error in the prediction of the grayscale image.
        #   The code only supports prediction of RGB images, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for identification
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add the batch_size dimension and normalize it
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        
        #---------------------------------------------------------#
        #   Feed the image into the network to make predictions!
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        #---------------------------------------------------------#
        #   Convert the image to an RGB image here to prevent the grayscale image from making errors during prediction.
        #   The code only supports prediction of RGB images, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for identification
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add the batch_size dimension and normalize it
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        #---------------------------------------------------------#
        #   Feed the image into the network to make predictions!
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        for i, c in enumerate(out_classes):
            predicted_class             = self.class_names[int(c)]
            try:
                score                   = str(out_scores[i].numpy())
            except:
                score                   = str(out_scores[i])
            top, left, bottom, right    = out_boxes[i]
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
