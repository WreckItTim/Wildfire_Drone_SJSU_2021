#requirements
import pandas as pd
import geemap
import os
import ee
import numpy as np
import matplotlib.pyplot as plt
from imutils import contours
from skimage import measure
import argparse
import imutils
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

class Aerial:

        def __init__(self):
                print('Parent Aerial obj created...')
        def transform(self,read_from_path, write_to_path):
                print('Aerial transform() not set!')

class Fire(Aerial):
        def __init__(self):
                self.pixels = 200
            print('Fire Aerial obj created...')
            
        def transform(self,read_from_path, write_to_path):
                
                img = cv2.imread(read_from_path)
 
                lower_bound = np.array([5,50,100],np.uint8)
                upper_bound = np.array([15,255,255],np.uint8)
                 
                frame_smooth = cv2.GaussianBlur(img,(15,15),0)
                 
                mask = np.zeros_like(img)
                   
                mask[0:img.shape[0], 0:img.shape[1]] = [255,255,255]
                 
                img_roi = cv2.bitwise_and(frame_smooth, mask)
                 
                frame_hsv = cv2.cvtColor(img_roi,cv2.COLOR_BGR2HSV)
                 
                image_binary = cv2.inRange(frame_hsv, lower_bound, upper_bound)

                cv2.imwrite(write_to_path, img_binary)

def generate_detections(checkpoint,images):
                    
                    print("Creating Graph...")
                    detection_graph = tf.Graph()
                    with detection_graph.as_default():
                        od_graph_def = tf.GraphDef()
                        with tf.gfile.GFile(checkpoint, 'rb') as fid:
                            serialized_graph = fid.read()
                            od_graph_def.ParseFromString(serialized_graph)
                            tf.import_graph_def(od_graph_def, name='')

                    boxes = []
                    scores = []
                    classes = []
                    k = 0
                    with detection_graph.as_default():
                        with tf.Session(graph=detection_graph) as sess:
                            for image_np in tqdm(images):
                                image_np_expanded = np.expand_dims(image_np, axis=0)
                                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                                box = detection_graph.get_tensor_by_name('detection_boxes:0')
                                score = detection_graph.get_tensor_by_name('detection_scores:0')
                                clss = detection_graph.get_tensor_by_name('detection_classes:0')
                                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                                # Actual detection.
                                (box, score, clss, num_detections) = sess.run(
                                        [box, score, clss, num_detections],
                                        feed_dict={image_tensor: image_np_expanded})

                                boxes.append(box)
                                scores.append(score)
                                classes.append(clss)
                                
                    boxes =   np.squeeze(np.array(boxes))
                    scores = np.squeeze(np.array(scores))
                    classes = np.squeeze(np.array(classes))

                    return boxes,scores,classes

def chip_image(img, chip_size=(60,60)):

                    width,height,_ = img.shape
                    wn,hn = chip_size
                    images = np.zeros((int(width/wn) * int(height/hn),wn,hn,3))
                    k = 0
                    for i in tqdm(range(int(width/wn))):
                        for j in range(int(height/hn)):
                            
                            chip = img[wn*i:wn*(i+1),hn*j:hn*(j+1),:3]
                            images[k]=chip
                            
                            k = k + 1
                    
                    return images.astype(np.uint8)

def draw_bboxes(img,boxes,classes):

                    source = Image.fromarray(img)
                    draw = ImageDraw.Draw(source)
                    w2,h2 = (img.shape[0],img.shape[1])

                    idx = 0

                    for i in range(len(boxes)):
                        xmin,ymin,xmax,ymax = boxes[i]
                        c = classes[i]

                        draw.text((xmin+15,ymin+15), str(c))

                        for j in range(4):
                            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
                    return source



class Building(Aerial):
        def __init__(self):
                self.chip_size = 60
                self.path_to_model='multires_aug.pb'
                print('Aerial obj created...')
            
        def transform(self,read_from_path, write_to_path):
   
                #Parse and chip images
                arr = np.array(Image.open(read_from_path))
                chip_size = (self.chip_size,self.chip_size)
                images = chip_image(arr,chip_size)
                print(images.shape)

                #generate detections
                boxes, scores, classes = generate_detections(self.path_to_model,images)

                #Process boxes to be full-sized
                width,height,_ = arr.shape
                cwn,chn = (chip_size)
                wn,hn = (int(width/cwn),int(height/chn))

                num_preds = 250
                bfull = boxes[:wn*hn].reshape((wn,hn,num_preds,4))
                b2 = np.zeros(bfull.shape)
                b2[:,:,:,0] = bfull[:,:,:,1]
                b2[:,:,:,1] = bfull[:,:,:,0]
                b2[:,:,:,2] = bfull[:,:,:,3]
                b2[:,:,:,3] = bfull[:,:,:,2]

                bfull = b2
                bfull[:,:,:,0] *= cwn
                bfull[:,:,:,2] *= cwn
                bfull[:,:,:,1] *= chn
                bfull[:,:,:,3] *= chn
                for i in range(wn):
                    for j in range(hn):
                        bfull[i,j,:,0] += j*cwn
                        bfull[i,j,:,2] += j*cwn
                                    
                        bfull[i,j,:,1] += i*chn
                        bfull[i,j,:,3] += i*chn
                                    
                bfull = bfull.reshape((hn*wn,num_preds,4))

                #only display boxes with confidence > .2
                bs = bfull[scores > .2]
                cs = classes[scores>.2]
                s = (write_to_path).split("/")[::-1]
                draw_bboxes(arr,bs,cs).save(s[0].split(".")[0] + ".png")

                with open(write_to_path,'w') as f:
                    for i in range(bfull.shape[0]):
                        for j in range(bfull[i].shape[0]):
                            #box should be xmin ymin xmax ymax
                            box = bfull[i,j]
                            class_prediction = classes[i,j]
                            score_prediction = scores[i,j]
                            f.write('%d %d %d %d %d %f \n' % \
                                (box[0],box[1],box[2],box[3],int(class_prediction),score_prediction))


#uploaded "testfire.jpg" and "satellite.tif" to possibly use for Demo unless you have other image to test. Note that we have two inputs and outputs because they're two tasks until we get sim aerial view with buildings and fire and/or get the Map interface connected.
#args: path to aerial fire image, path to output fire detection, path to aerial building image, path to output object/building detection
#arg: path to model from main Git page
fire = Fire()
building = Building()
fire.transform("testfire.jpg", "firedetection.jpg")
building.transform("satellite.tif", "satellite_objectdetection.txt")
