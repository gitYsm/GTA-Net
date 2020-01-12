# object_detector.py

import cv2
import os
import numpy as np
import tensorflow as tf
import tarfile
import six.moves.urllib as urllib
import time
import sys
import glob 
from os import listdir
import dkimgtracking_xyplusjsd  as dk
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

import sys

class ObjectDetector():
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    GRAPH_FILE_NAME = 'frozen_inference_graph_121086.pb'
    NUM_CLASSES = 90

    def download_model(self, model_name, pb_name):
        model_file = model_name + '.tar.gz'
        print("downloading model", model_name, "...")
        opener = urllib.request.URLopener()
        opener.retrieve(self.DOWNLOAD_BASE + model_file, model_file)
        print("download completed");
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if self.pb_name in file_name:
                tar_file.extract(file, os.getcwd())
                print(self.graph_file, "is extracted");

    #def __init__(self, model_name, label_file='data/mscoco_label_map.pbtxt'):
    def __init__(self, model_name, pb_name, label_file='data/cp_label_map.pbtxt'):
        # Initialize some variables
        print("ObjectDetector('%s','%s', '%s')" % (model_name, pb_name, label_file))
        self.process_this_frame = True

        # download model
        self.graph_file = model_name + '/' + pb_name
        if not os.path.isfile(self.graph_file):
            self.download_model(model_name, pb_name)

        # Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            graph = self.detection_graph

            ops = graph.get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                    ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = graph.get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, 480, 640)
                detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)

            self.tensor_dict = tensor_dict

        self.sess = tf.Session(graph=self.detection_graph)

        # Loading label map
        # Label maps map indices to category names,
        # so that when our convolution network predicts `5`,
        # we know that this corresponds to `airplane`.
        # Here we use internal utility functions,
        # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(label_file)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.output_dict = None

        self.last_inference_time = 0

    def run_inference(self, image_np):
        sess = self.sess
        graph = self.detection_graph
        with graph.as_default():
            image_tensor = graph.get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(self.tensor_dict,
                    feed_dict={image_tensor: np.expand_dims(image_np, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    def time_to_run_inference(self):
        unixtime = int(time.time())
        if self.last_inference_time != unixtime:
            self.last_inference_time = unixtime
            return True
        return False

    def detect_objects(self, frame):
        '''
        time1 = time.time()
        # Grab a single frame of video

        # Resize frame of video to 1/4 size for faster face recognition processing
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = frame

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        time2 = time.time()
        '''

        rgb_small_frame = frame[:, :, ::-1]

        '''
        # Only process every other frame of video to save time
        if self.time_to_run_inference():
            self.output_dict = self.run_inference(rgb_small_frame)
        '''

        self.output_dict = self.run_inference(rgb_small_frame)

        #time3 = time.time()

        vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                self.output_dict['detection_boxes'],
                self.output_dict['detection_classes'],
                self.output_dict['detection_scores'],
                self.category_index,
                instance_masks=self.output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=1)

        #time4 = time.time()

        #print("%0.3f, %0.3f, %0.3f sec" % (time2 - time1, time3 - time2, time4 - time3))

        return (frame, self.output_dict)

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

def Img_Extraction(poslist,frame,addWH):

    img = frame.copy()
    height,width = img.shape[:2]
    left_y = (poslist[0] * height)
    left_x = (poslist[1] * width) 
    right_y = (poslist[2] * height) 
    right_x = (poslist[3] * width)

    if addWH:
        if left_y - 5 >= 0:
            left_y -= 5
        else:
            left_y = 0

        if left_x -5 >= 0:
            left_x -= 5
        else:
            left_x = 0

        if right_y + 5 < height:
            right_y +=5
        else:
            right_y = height  

        if right_x + 5 < width:
            right_x += 5 
        else:
            right_x = width 

    result = frame[int(left_y) : int(right_y), int(left_x) : int(right_x)] 

    return result

if __name__ == '__main__':

    print("Main Start")
    #detector = ObjectDetector('ssd_mobilenet_v1_coco_2017_11_17')
    detector = ObjectDetector('modeltest', 'frozen_inference_graph.pb')
    detector2 = ObjectDetector('modeltest', 'frozen_inference_graph_121086.pb')
    #detector = ObjectDetector('mask_rcnn_inception_v2_coco_2018_01_28')
    #detector = ObjectDetector('pet', label_file='data/pet_label_map.pbtxt')

    # Using OpenCV to capture from device 0. If you have trouble capturing
    # from a webcam, comment the line below out and use a video file
    # instead.
    # files = listdir('./video')
    #train_txt = open("train.txt", 'w')

    cap = cv2.VideoCapture(sys.argv[1])
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    #writer = cv2.VideoWriter('output.mp4', fourcc, cap_fps, (int(cap_width), int(cap_height)))

    print("press `q` to quit")
    videoname = sys.argv[1][:-4] # sample_city
    detecbox_list = list()
    np_and_coordinate= list()
    filenum = list()

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        frameOrigin = frame.copy() 
        frame, output_dict = detector.detect_objects(frame)

        # show the frame
        cv2.imshow("Frame1111", frame)

        #ysm controlframe
        #for _ in range(0, 2):
        #        cap.read()

        for _ in range(0, 5):
            if(output_dict['detection_scores'][_]> 0.995):
                
                extracted_img = Img_Extraction(output_dict['detection_boxes'][_], frameOrigin, True)
                #pathNamejpg = "cropimg/jpg" + str(numbering) + ".jpg"
                #numbering = numbering + 1

                while True:
                    
                    frame2 = extracted_img.copy()
                    cv2.imshow("Frame1111", frame2)
                    frameOrigin2 = frame2.copy() 
                    frame2, output_dict2 = detector2.detect_objects(frame2)
                    foldername = ""
                    cpcnt = 0 
                    filename = ""
                    
                    xyxy = list()
                    xyxy.append(round(output_dict['detection_boxes'][_][0]* cap_height,2))
                    xyxy.append(round(output_dict['detection_boxes'][_][1]* cap_width,2))
                    for _ in range(0, 5): # top 5
                        if(output_dict2['detection_scores'][_]> 0.995):

                            extracted_img2 = Img_Extraction(output_dict2['detection_boxes'][_], frameOrigin2, False)
                            np_and_coordinate = list()
                            np_and_coordinate.append(extracted_img2)
                            np_and_coordinate.append(xyxy)
                                
                            if len(detecbox_list) == 0:                # init() ......................................
                                detecbox_list.append(np_and_coordinate)
                                foldername = dk.Create_Folder(videoname,1) # samplecity_1
                                foldername = "./" + foldername           # /samplecity_1

                                filenum.append(1)
                                filename = foldername + "/cp1_" + str(filenum[0]) + ".png"  #/samplecity_1/cp1_1.png
                                cv2.imwrite(filename, extracted_img2, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                            else:
                                cpcnt = dk.Check_Same(detecbox_list, np_and_coordinate) 
                                if cpcnt > len(detecbox_list): # if diffrent
                                    detecbox_list.append(np_and_coordinate)
                                    filenum.append(1)
                                else:                          # if same
                                    # detecbox_list[cpcnt- 1][0] = extracted_img2
                                    detecbox_list[cpcnt- 1][1] = xyxy 

                                foldername = dk.Create_Folder(videoname,cpcnt)
                                foldername = "./" + foldername          
                                filename = foldername + "/cp" + str(cpcnt) + "_" + str(filenum[cpcnt - 1])+ ".png"
                                filenum[cpcnt - 1] += 1 
                                cv2.imwrite(filename, extracted_img2, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                    #print('fin one box')
                    break
                    #print(pathName)
                    #cv2.imwrite(pathName, extracted_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    

            #break
            #writer.write(frame)
            #print("wrote", ii, 'frame')
            #ii += 1
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        if key == ord("s"):
            for _ in range(0, 60):
                cap.read()
            print("skipping 60 frames")

        # do a bit of cleanup
        #cap.release()
        #writer.release()
    cv2.destroyAllWindows()
    print('finish')
