# Modified by Dong-Geon Kim
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from functools import partial

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from nets.yolo import get_train_model, yolo_body
from nets.yolo_training import get_yolo_loss
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint, WarmUpCosineDecayScheduler)
from utils.dataloader import YoloDatasets
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

'''
When training your own target detection model, you must pay attention to the following points:
1. Before training, carefully check whether your format meets the requirements. The library requires the data set format to be VOC format, and the content to be prepared includes input pictures and labels
   The input image is a .jpg image, no fixed size is required, and it will be automatically resized before being passed into training.
   Grayscale images will be automatically converted to RGB images for training, no need to modify them yourself.
   If the suffix of the input image is not jpg, you need to convert it into jpg in batches before starting training.

   The tag is in .xml format, and the file contains target information to be detected. The tag file corresponds to the input image file.

2. The trained weight file is saved in the logs folder, and each epoch will be saved once. If only a few steps are trained, it will not be saved. The concepts of epoch and step should be clarified.
   During the training process, the code does not set to save only the lowest loss, so after training with the default parameters, there will be 100 weights. If the space is not enough, you can delete it yourself.
   This is not to save as little as possible, nor to save as much as possible. Some people want to save them all, and some people want to save only a little bit. In order to meet most needs, it is still highly optional to save them.

3. The size of the loss value is used to judge whether to converge or not. More importantly, there is a trend of convergence, that is, the loss of the validation set continues to decrease. If the loss of the validation set basically does not change, the model basically converges.
   The specific size of the loss value does not make much sense. The big and small only depend on the calculation method of the loss, and it is not good to be close to 0. If you want to make the loss look better, you can directly divide 10000 into the corresponding loss function.
   The loss value during training will be saved in the loss_%Y_%m_%d_%H_%M_%S folder under the logs folder

4. Parameter tuning is a very important knowledge. No parameter is necessarily good. The existing parameters are the parameters that I have tested and can be trained normally, so I would recommend using the existing parameters.
   But the parameters themselves are not absolute. For example, as the batch increases, the learning rate can also be increased, and the effect will be better; too deep networks should not use too large a learning rate, etc.
   These are all based on experience, you can only rely on the students to inquire more information and try it yourself.
'''  

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == "__main__":
    #------------------------------------------------------#
    #   Whether to use eager mode training
    #------------------------------------------------------#
    eager = False
    #------------------------------------------------------#
    #   Be sure to modify classes_path before training so that it corresponds to your own dataset
    #------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Please refer to the README for the download of the weight file, which can be downloaded from the network disk. The pretrained weights of the model are common to different datasets because the features are common.
    #   The more important part of the pre-training weight of the model is the weight part of the backbone feature extraction network, which is used for feature extraction.
    #   Pre-training weights must be used in 99% of cases. If they are not used, the weights of the main part are too random, the feature extraction effect is not obvious, and the results of network training will not be good.
    #
    #   If there is an operation that interrupts the training during the training process, you can set the model_path to the weights file in the logs folder, and reload the weights that have been trained.
    #   At the same time, modify the parameters of the freeze phase or thaw phase below to ensure the continuity of the model epoch.
    #   
    #   When model_path = '', the weights of the entire model are not loaded.
    #
    #   The weights of the entire model are used here, so they are loaded in train.py.
    #   If you want the model to start training from 0, set model_path = '', the following Freeze_Train = Fasle, then start training from 0, and there is no process of freezing the backbone.
    #   Generally speaking, the training effect from 0 will be poor, because the weights are too random, and the feature extraction effect is not obvious.
    #
    #   The network generally does not start training from 0, at least the weights of the backbone part are used, and some papers mention that pre-training is not necessary, the main reason is that their data set is large and the parameter adjustment ability is excellent.
    #   If you must train the backbone part of the network, you can learn about the imagenet data set. First, train the classification model. The backbone part of the classification model is common to the model, and training is based on this.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolox_s.h5'
    #------------------------------------------------------#
    #   The size of the input shape must be a multiple of 32
    #------------------------------------------------------#
    input_shape     = [640, 640]
    #---------------------------------------------------------------------#
    #   The version of YoloX used. tiny, s, m, l, x
    #---------------------------------------------------------------------#
    phi             = 's'
    #------------------------------------------------------------------------------------------------------------#
    #   YoloX's tricks app
    #   mosaic mosaic data augmentation True or False
    #   YOLOX author emphasizes to turn off Mosaic at N epochs before the end of training. Because the training images generated by Mosaic are far from the true distribution of natural images.
    #   And Mosaic's large number of crop operations will bring a lot of inaccurate annotation boxes. This code will automatically use mosaic in the first 90% of epochs, and will not use it later.
    #   Cosine_scheduler Cosine annealing learning rate True or False
    #------------------------------------------------------------------------------------------------------------#
    mosaic              = False
    Cosine_scheduler    = False

    #------------------------------------------------------#
    #   The training is divided into two phases, the freezing phase and the thawing phase.
    #   Insufficient video memory has nothing to do with the size of the data set. If it indicates that the video memory is insufficient, please reduce the batch_size.
    #   Affected by the BatchNorm layer, the minimum batch_size is 2 and cannot be 1.
    #------------------------------------------------------#
    #------------------------------------------------------#
    #   Freeze phase training parameters
    #   At this time, the backbone of the model is frozen, and the feature extraction network does not change
    #   Occupy less memory, only fine-tune the network
    #------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    Freeze_lr           = 1e-3
    #------------------------------------------------------#
    #   Thawing phase training parameters
    #   At this time, the backbone of the model is not frozen, and the feature extraction network will change
    #   The occupied video memory is large, and all the parameters of the network will be changed
    #------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 1e-4
    #------------------------------------------------------#
    #   Whether to freeze training, the default is to freeze the main training first and then unfreeze the training.
    #------------------------------------------------------#
    Freeze_Train        = True
    #------------------------------------------------------#
    #   Used to set whether to use multi-threading to read data, 1 means to turn off multi-threading
    #   When enabled, it will speed up data reading, but it will take up more memory
    #   When multi-threading is enabled in keras, sometimes the speed is much slower
    #   Turn on multithreading when IO is the bottleneck, that is, the GPU operation speed is much faster than the speed of reading pictures.
    #   Valid when eager mode is False
    #------------------------------------------------------#
    num_workers         = 1
    #------------------------------------------------------#
    #   get image path and label
    #------------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    #------------------------------------------------------#
    #   get classes
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #------------------------------------------------------#
    #   Create a yolo model
    #------------------------------------------------------#
    model_body  = yolo_body([None, None, 3], num_classes = num_classes, phi = phi)
    if model_path != '':
        #------------------------------------------------------#
        #   Load pretrained weights
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)

    if not eager:
        model = get_train_model(model_body, input_shape, num_classes)
    else:
        yolo_loss = get_yolo_loss(input_shape, len(model_body.output), num_classes)
    #-------------------------------------------------------------------------------#
    #   set the training parameters
    #   logging indicates the storage address of tensorboard
    #   checkpoint is used to set the details of weight saving, period is used to modify how many epochs are saved once
    #   reduce_lr is used to set the way the learning rate decreases
    #   early_stopping is used to set early stop, and val_loss will automatically end the training without falling for many times, indicating that the model is basically converged
    #-------------------------------------------------------------------------------#
    logging         = TensorBoard(log_dir = 'logs/')
    checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                            monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
    if Cosine_scheduler:
        reduce_lr   = WarmUpCosineDecayScheduler(T_max = 5, eta_min = 1e-5, verbose = 1)
    else:
        reduce_lr   = ExponentDecayScheduler(decay_rate = 0.94, verbose = 1)
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
    loss_history    = LossHistory('logs/')

    #---------------------------#
    #   Read the txt corresponding to the dataset
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if Freeze_Train:
        freeze_layers = {'tiny': 125, 's': 125, 'm': 179, 'l': 234, 'x': 290}[phi]
        for i in range(freeze_layers): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
        
    #------------------------------------------------------#
    #   The backbone feature extraction network features are common, and freezing training can speed up training
    #   Also prevents weights from being corrupted at the beginning of training.
    #   Init_Epoch is the starting generation
    #   Freeze_Epoch is the epoch to freeze training
    #   UnFreeze_Epoch total training generation
    #   Prompt OOM or insufficient video memory, please reduce the Batch_size
    #------------------------------------------------------#
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('The dataset is too small for training, please expand the dataset.')

        train_dataloader    = YoloDatasets(train_lines, input_shape, batch_size, num_classes, end_epoch - start_epoch, mosaic = mosaic, train = True)
        val_dataloader      = YoloDatasets(val_lines, input_shape, batch_size, num_classes, end_epoch - start_epoch, mosaic = False, train = False)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        if eager:
            gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

            if Cosine_scheduler:
                lr_schedule = tf.keras.experimental.CosineDecayRestarts(
                    initial_learning_rate = lr, first_decay_steps = 5 * epoch_step, t_mul = 1.0, alpha = 1e-2)
            else:
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=0.94, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

            for epoch in range(start_epoch, end_epoch):
                fit_one_epoch(model_body, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            end_epoch, input_shape, num_classes)

        else:
            model.compile(optimizer=Adam(lr = lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
            )

    if Freeze_Train:
        for i in range(freeze_layers): model_body.layers[i].trainable = True

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('The dataset is too small for training, please expand the dataset.')

        train_dataloader    = YoloDatasets(train_lines, input_shape, batch_size, num_classes, end_epoch - start_epoch, mosaic = mosaic, train = True)
        val_dataloader      = YoloDatasets(val_lines, input_shape, batch_size, num_classes, end_epoch - start_epoch, mosaic = False, train = False)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        if eager:
            gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

            if Cosine_scheduler:
                lr_schedule = tf.keras.experimental.CosineDecayRestarts(
                    initial_learning_rate = lr, first_decay_steps = 5 * epoch_step, t_mul = 1.0, alpha = 1e-2)
            else:
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=0.94, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

            for epoch in range(start_epoch, end_epoch):
                fit_one_epoch(model_body, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            end_epoch, input_shape, num_classes)

        else:
            model.compile(optimizer=Adam(lr = lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
            )
