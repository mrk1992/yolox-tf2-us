import tensorflow as tf
from tensorflow.keras import backend as K


#---------------------------------------------------#
#   Adjust the box to match the real picture
#---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   The y-axis is placed in the front because it is convenient to multiply the prediction box and the width and height of the image
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   The offset obtained here is the offset of the effective area of the image relative to the upper left corner of the image
        #   new_shape refers to the width and height scaling
        #-----------------------------------------------------------------#
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

#---------------------------------------------------#
#   Picture prediction
#---------------------------------------------------#
def DecodeBox(outputs,
            num_classes,
            input_shape,
            max_boxes       = 100,
            confidence      = 0.5,
            nms_iou         = 0.3,
            letterbox_image = True):
            
    image_shape = K.reshape(outputs[-1],[-1])
    outputs     = outputs[:-1]
    #---------------------------#
    #   get batch_size
    #---------------------------#
    bs      = K.shape(outputs[0])[0]

    grids   = []
    strides = []
    #---------------------------#
    #   Obtain the height and width of three effective feature layers
    #---------------------------#
    hw      = [K.shape(x)[1:3] for x in outputs]
    #----------------------------------------------#
    #   batch_size, 80, 80, 4 + 1 + num_classes
    #   batch_size, 40, 40, 4 + 1 + num_classes
    #   batch_size, 20, 20, 4 + 1 + num_classes
    #   
    #   6400 + 1600 + 400
    #   outputs batch_size, 8400, 4 + 1 + num_classes
    #----------------------------------------------#
    outputs = tf.concat([tf.reshape(x, [bs, -1, 5 + num_classes]) for x in outputs], axis = 1)
    for i in range(len(hw)):
        #---------------------------#
        #   Generate grid points from feature layers
        #   Get the coordinates of each valid feature layer grid point
        #---------------------------#
        grid_x, grid_y  = tf.meshgrid(tf.range(hw[i][1]), tf.range(hw[i][0]))
        grid            = tf.reshape(tf.stack((grid_x, grid_y), 2), (1, -1, 2))
        shape           = tf.shape(grid)[:2]

        grids.append(tf.cast(grid, K.dtype(outputs)))
        strides.append(tf.ones((shape[0], shape[1], 1)) * input_shape[0] / tf.cast(hw[i][0], K.dtype(outputs)))
    #---------------------------#
    #   Stack grid points together
    #---------------------------#
    grids               = tf.concat(grids, axis=1)
    strides             = tf.concat(strides, axis=1)
    #-------------------------------------------#
    #   Decoding from grid points
    #   box_xy Obtain the result of normalization of the center of the prediction box
    #   box_wh Obtain the result of the normalized width and height of the prediction box
    #-------------------------------------------#
    box_xy = (outputs[..., :2] + grids) * strides / K.cast(input_shape[::-1], K.dtype(outputs))
    box_wh = tf.exp(outputs[..., 2:4]) * strides / K.cast(input_shape[::-1], K.dtype(outputs))

    #-------------------------------------------#
    #   box_confidence Whether the feature point has a corresponding object
    #   box_class_probs Goose Confidence of Feature Point Object Kinds
    #-------------------------------------------#
    box_confidence  = K.sigmoid(outputs[..., 4:5])
    box_class_probs = K.sigmoid(outputs[..., 5: ])
    #------------------------------------------------------------------------------------------------------------#
    #   Before the image is passed to the network prediction, letterbox_image will be performed to add gray bars around the image, so the generated box_xy, box_wh is relative to the image with gray bars
    #   We need to modify it to remove the gray bars. Adjust box_xy, and box_wh to y_min, y_max, xmin, xmax
    #   If you do not use letterbox_image, you also need to adjust the normalized box_xy, box_wh to the size of the original image
    #------------------------------------------------------------------------------------------------------------#
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    box_scores  = box_confidence * box_class_probs

    #-----------------------------------------------------------#
    #   Determine whether the score is greater than score_threshold
    #-----------------------------------------------------------#
    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):
        #-----------------------------------------------------------#
        #   Take out all boxes with box_scores >= score_threshold, and scores
        #-----------------------------------------------------------#
        class_boxes      = tf.boolean_mask(boxes, mask[..., c])
        class_box_scores = tf.boolean_mask(box_scores[..., c], mask[..., c])

        #-----------------------------------------------------------#
        #   non-maximal suppression
        #   Keep the box with the highest score in a certain area
        #-----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        #-----------------------------------------------------------#
        #   Get the result after non-maximal suppression
        #   The following three are: box position, score and type
        #-----------------------------------------------------------#
        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    def decode_for_vision(output):
        #---------------------------#
        #   Generate grid points from feature layers
        #---------------------------#
        # batch_size, 20, 20, 4 + 1 + num_classes
        bs, hw = np.shape(output)[0], np.shape(output)[1:3]
        # batch_size, 400, 4 + 1 + num_classes
        output          = np.reshape(output, [bs, hw[0] * hw[1], -1])

        #---------------------------#
        #   According to the height and width of the feature layer
        #   Build the grid
        #---------------------------#
        grid_x, grid_y  = np.meshgrid(np.arange(hw[1]), np.arange(hw[0]))
        #------------------------------------#
        #   Single image, xy-coordinates of four hundred grid points
        #   1, 400, 2
        #------------------------------------#
        grid            = np.reshape(np.stack((grid_x, grid_y), 2), (1, -1, 2))
        #------------------------#
        #   Decoding from grid points
        #   box_xy is the center of the predicted box
        #   box_wh is the width and height of the predicted box
        #------------------------#
        box_xy  = (output[..., :2] + grid)
        box_wh  = np.exp(output[..., 2:4])

        fig = plt.figure()
        ax  = fig.add_subplot(121)
        plt.ylim(-2,22)
        plt.xlim(-2,22)
        plt.scatter(grid_x,grid_y)
        plt.scatter(0,0,c='black')
        plt.scatter(1,0,c='black')
        plt.scatter(2,0,c='black')
        plt.gca().invert_yaxis()

        ax  = fig.add_subplot(122)
        plt.ylim(-2,22)
        plt.xlim(-2,22)
        plt.scatter(grid_x,grid_y)
        plt.scatter(0,0,c='black')
        plt.scatter(1,0,c='black')
        plt.scatter(2,0,c='black')

        plt.scatter(box_xy[0,0,0], box_xy[0,0,1],c='r')
        plt.scatter(box_xy[0,1,0], box_xy[0,1,1],c='r')
        plt.scatter(box_xy[0,2,0], box_xy[0,2,1],c='r')
        plt.gca().invert_yaxis()

        pre_left    = box_xy[...,0] - box_wh[...,0]/2 
        pre_top     = box_xy[...,1] - box_wh[...,1]/2 

        rect1   = plt.Rectangle([pre_left[0,0],pre_top[0,0]],box_wh[0,0,0],box_wh[0,0,1],color="r",fill=False)
        rect2   = plt.Rectangle([pre_left[0,1],pre_top[0,1]],box_wh[0,1,0],box_wh[0,1,1],color="r",fill=False)
        rect3   = plt.Rectangle([pre_left[0,2],pre_top[0,2]],box_wh[0,2,0],box_wh[0,2,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()

    #---------------------------------------------#
    #   batch_size, 20, 20, 4 + 1 + num_classes
    #---------------------------------------------#
    feat = np.concatenate([np.random.uniform(-1, 1, [4, 20, 20, 2]), np.random.uniform(1, 3, [4, 20, 20, 2]), np.random.uniform(1, 3, [4, 20, 20, 81])], -1)
    decode_for_vision(feat)
