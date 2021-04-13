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
from PIL import Image, ImageDraw,ImageFont

# tim edit - global functions and variables
import utils
import vision as v

class Fire(v.Vision):
    def __init__(self):
        self.pixels = 200
        #print('Fire Aerial obj created...')
            
    def transform(self, read_from_path, write_to_path):
        img = cv2.imread(read_from_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        labels = measure.label(thresh, connectivity=2, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")

        for label in np.unique(labels):
            if label == 0:
                continue

            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            if numPixels > self.pixels:
                mask = cv2.add(mask, labelMask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]
        
        # tim edit - writes white circles now only - indicate where fire is
        img[:,:,:] = 0
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(img, (int(cX), int(cY)), int(radius), (255, 0, 0), -1)
            #cv2.circle(img, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
            #cv2.putText(img, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
        # tim edit - match global image formatting
        img = Image.fromarray(img)
        img = utils.convertPIL(img)
        img.save(write_to_path)
        #cv2.imwrite(write_to_path, utils.convertPIL(img))

def generate_detections(checkpoint, images):
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



class Objects(v.Vision):
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
        #draw_bboxes(arr,bs,cs).save(s[0].split(".")[0] + ".png")
        draw_bboxes(arr,bs,cs).save(write_to_path)
        
        # tim edit - will need this for later, but off for now need help @olivia
        '''
        with open(write_to_path,'w') as f:
            for i in range(bfull.shape[0]):
                for j in range(bfull[i].shape[0]):
                    #box should be xmin ymin xmax ymax
                    box = bfull[i,j]
                    class_prediction = classes[i,j]
                    score_prediction = scores[i,j]
                    f.write('%d %d %d %d %d %f \n' % \
                        (box[0],box[1],box[2],box[3],int(class_prediction),score_prediction))
        '''


#uploaded "testfire.jpg" and "satellite.tif" to possibly use for Demo unless you have other image to test. Note that we have two inputs and outputs because they're two tasks until we get sim aerial view with buildings and fire and/or get the Map interface connected.
#args: path to aerial fire image, path to output fire detection, path to aerial building image, path to output object/building detection
#arg: path to model from main Git page
#fire = Fire()
#building = Building()
#fire.transform("testfire.jpg", "firedetection.jpg")
#building.transform("satellite.tif", "satellite_objectdetection.txt")
