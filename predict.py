#-----------------------------------------------------------------------#
#   predict.py integrates functions such as single image prediction, camera detection, FPS test and directory traversal detection
#   It is integrated into a py file, and the mode is modified by specifying the mode.
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from yolo import YOLO

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode is used to specify the mode of the test:
    #   'predict' indicates the prediction of a single picture. If you want to modify the prediction process, such as saving pictures, intercepting objects, etc., you can see the detailed comments below.
    #   'video' indicates video detection, you can call the camera or video for detection, see the notes below for details.
    #   'fps' indicates test fps, the image used is street.jpg in img, see the notes below for details.
    #   'dir_predict' indicates to traverse the folder to detect and save. By default, the img folder is traversed and the img_out folder is saved. For details, see the notes below.
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #----------------------------------------------------------------------------------------------------------#
    #   video_path is used to specify the path of the video, when video_path=0, it means to detect the camera
    #   If you want to detect the video, set it as video_path = "xxx.mp4", which means to read the xxx.mp4 file in the root directory.
    #   video_save_path indicates the path where the video is saved, when video_save_path="" it means not to save
    #   If you want to save the video, set it as video_save_path = "yyy.mp4", which means that it will be saved as a yyy.mp4 file in the root directory.
    #   video_fps for the fps of the saved video
    #   video_path, video_save_path and video_fps are only valid when mode='video'
    #   When saving the video, you need ctrl+c to exit or run to the last frame to complete the complete save step.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #-------------------------------------------------------------------------#
    #   test_interval is used to specify the number of image detections when measuring fps
    #   In theory, the larger the test_interval, the more accurate the fps.
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path specifies the folder path of the image used for detection
    #   dir_save_path specifies the save path of the detected image
    #   dir_origin_path and dir_save_path are only valid when mode='dir_predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        '''
        1. If you want to save the detected image, use r_image.save("img.jpg") to save it, and modify it directly in predict.py.
        2. If you want to get the coordinates of the prediction box, you can enter the yolo.detect_image function and read the four values of top, left, bottom, and right in the drawing part.
        3. If you want to use the prediction frame to intercept the target, you can enter the yolo.detect_image function, and use the obtained four values of top, left, bottom, and right in the drawing part
        Use the matrix method to intercept the original image.
        4. If you want to write extra words on the prediction map, such as the number of specific targets detected, you can enter the yolo.detect_image function and judge the predicted_class in the drawing part,
        For example, judging if predicted_class == 'car': can judge whether the current target is a car, and then record the number. Use draw.text to write.
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            # read a frame
            ref,frame=capture.read()
            # Format conversion, BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # test
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR meets the opencv display format
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm
        
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
