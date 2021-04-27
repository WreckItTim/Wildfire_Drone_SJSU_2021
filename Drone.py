# @Tim

import numpy as np
import tempfile
import pprint
import cv2
import os
import setup_path # need this in same directory as python code
import sys
import pickle
import time
import threading
import socket
import random
try:
    import airsim
    print('airsim imported')
except ImportError:
    print('airsim not imported')
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile

def hello():
  print('hello there')

class Drone:
  def disconnect(self):
    pickle.dump(self.actions, open(os.path.join(self.logPath, 'actions.p'), 'wb'))
    print('Disconnected... List of Actions:')
    for timeStep in self.actions:
        print(timeStep, self.actions[timeStep])

  def __init__(self):
    self.actions = {}
    self.speed = 20
    self.distance = 20
    self.duration = 1
    self.pos = np.array([0, 0, 0]).astype(int)
    self.photosPath = ''
    self.logPath = ''
  def connect(self):
    print('connect() not defined from child')
  def getState(self):
    print('getState() not defined from child')
  def takeOff(self):
    print('takeOff() not defined from child')
  def land(self):
    print('land() not defined from child')
  def move(self, x, y, z):
    print('takePictures() not defined from child')
  def moveTo(self, x, y, z):
    print('takePictures() not defined from child')
  def takePictures(self, folderPath, index=0):
    print('takePictures() not defined from child')
  def liveStream(self, modules, write_folder):
    print('liveStream() not defined from child')
  def command(self):
    print('command() not defined from child')
  def snapAerial(self, path):
    print('snapAerial() not defined from child')
  def getPos(self):
    print('getPos() not defined from child')


    

def stream(modules, write_folder):
  camera = cv2.VideoCapture('udp://127.0.0.1:11111')
  # loop to read and display video with transformations for 3 passed in modules
  # WARNING: make sure to press q to quit, so properly shuts down
  counter = 0
  while Tello.streaming:
  
    #counter += 1
    ret, frame = camera.read()
    Tello.imgs[Tello.img_idx] = frame
    Tello.img_idx += 1
    #if counter % 100 == 0:
        #read_path = os.path.join(write_folder, 'Raw_' + str(counter/100) + '.png')
        #im = Image.fromarray(frame)
        #im.save(read_path)
        #time.sleep(5)
        #cv2.imwrite(read_path, frame)
    #imgs = [None] * 4
    #imgs[0] = cv2.resize(frame, (240, 180))
    #for idx, module in enumerate(modules):
      #write_path = os.path.join(write_folder, module + '_' + str(counter) + '.png')
      #modules[module].transform(read_path, write_path)
      #imgs[idx+1] = cv2.resize(cv2.imread(write_path), (240, 180))
    #'MonoDepth2'
    #numpy_vertical1 = np.vstack((imgs[0], imgs[2]))
    #numpy_vertical2 = np.vstack((imgs[1], imgs[3]))
    #numpy_img = np.hstack((numpy_vertical1, numpy_vertical2))
    #cv2.imshow('Computer Vision', imgs[0])#numpy_img)
    cv2.imshow('Computer Vision',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  camera.release()
  cv2.destroyAllWindows()

class Tello(Drone):
  imgs = [None] * 10000
  img_idx = 0
  camThread = None
  streaming = False
  def __init__(self):
    super().__init__()
    self.sock = None
    self.address = None
    self.recvThread = None
    self.receiving = False 

  # function that runs on another thread to constantly receive messages being sent from drone
  def recv(self):
    while self.receiving: 
        response, ip = self.sock.recvfrom(1518)
        print(response)

  # function to issue command to drone
  # for full list of commands see: https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
  def command(self, msg):
    self.sock.sendto(msg.encode(encoding="utf-8"), self.address)

  def connect(self):
    # open sockets to send/receive commands/stream to/from drone
    host = ''
    port = 9000
    locaddr = (host, port) 
    self.address = ('192.168.10.1', 8889)
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.bind(locaddr)
    self.receiving = True
    self.recvThread = threading.Thread(target=self.recv)
    self.recvThread.start()
    # establish link with drone
    s = self.command('command')
    time.sleep(10)
    liveStream(None, None)
    time.sleep(10)
    
  def liveStream(self, modules, write_folder):
    s = self.command('streamon')
    Tello.streaming = True
    Tello.camThread = threading.Thread(target=stream, args=[modules, write_folder])
    Tello.camThread.start()

  def disconnect(self):
    super().disconnect() 
    self.streaming = False
    Tello.camThread.join()
    self.receiving = False
    self.recvThread.join()
    # cleanup
    self.sock.close()

  def move(self, x, y, z):
    if x > 0:
      s = self.command('forward ' + str(x))
    if x < 0:
      s = self.command('back ' + str(abs(x)))
    if y > 0:
      s = self.command('right '+ str(y))
    if y < 0:
      s = self.command('left ' + str(abs(y)))
    if z < 0:
      s = self.command('down ' + str(z))
    if z > 0:
      s = self.command('up ' +  str(abs(z)))
    self.pos += np.array([x, y, z]).astype(int)

  def moveTo(self, x, y, z):
    self.command(f'go {x-self.pos[0]} {y-self.pos[1]} {z-self.pos[2]} {self.speed}')
    self.pos = np.array([x, y, z]).astype(int)

  def flip(self, direction=None):
    if direction is None:
      direction = random.choice(['l', 'r', 'f', 'b'])
    s = self.command('flip ' +  direction)

  def takeOff(self):
    s = self.command('takeoff')
    time.sleep(10)
    self.pos = (0, 0, 0)

  def land(self):
    s = self.command('land')
    
  def takePictures(self, folderPath, index=0):
    #s = self.command('streamon')
    #camera = cv2.VideoCapture('udp://127.0.0.1:11111')
    #ret, frame = camera.read()
    frame = Tello.imgs[Tello.img_idx - 1]
    cv2.imwrite(os.path.join(folderPath, 'Scene.png'), frame)
    #camera.release()
    #s = self.command('streamoff')

  def snapAerial(self, path):
    copyfile('satellite.png', os.path.join(path, 'SatelliteObjects.png'))
    copyfile('testfire.png', os.path.join(path, 'SatelliteFire.png'))




class Unreal(Drone):
  def __init__(self):
    super().__init__()
    self.client = None

  def connect(self):
    self.client = airsim.MultirotorClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    
  def disconnect(self):
    super().disconnect()
    self.client.armDisarm(False)
    self.client.reset()
    self.client.enableApiControl(False)
    self.client = None

  def getState(self):
    info = {}
    info['state'] = self.client.getMultirotorState()
    info['imu'] = self.client.getImuData()
    info['barometer'] = self.client.getBarometerData()
    info['magnetometer'] = self.client.getMagnetometerData()
    info['gps'] = self.client.getGpsData()
    return info

  def takeOff(self):
    self.client.takeoffAsync().join()

  def land(self):
    self.client.landAsync().join()
    
  def move(self, x, y, z):
    self.client.moveByVelocityAsync(x, y, -1*z, self.duration).join()
  
  def moveTo(self, x, y, z):
    self.client.moveToPositionAsync(x, y, -1*z, self.speed).join()
    
  def takePictures(self, folderPath, index=0):
    responses = self.client.simGetImages([
        self.client.simGetImages([airsim.ImageRequest(index, airsim.ImageType.Scene , False, False)])[0]
        #,self.client.simGetImages([airsim.ImageRequest(index, airsim.ImageType.DepthPlanner , False, False)])[0]
        #,self.client.simGetImages([airsim.ImageRequest(index, airsim.ImageType.DepthPerspective , False, False)])[0]
        ,self.client.simGetImages([airsim.ImageRequest(index, airsim.ImageType.DepthVis , False, False)])[0]
        #,self.client.simGetImages([airsim.ImageRequest(index, airsim.ImageType.DisparityNormalized , False, False)])[0]
        ,self.client.simGetImages([airsim.ImageRequest(index, airsim.ImageType.Segmentation , False, False)])[0]
        ,self.client.simGetImages([airsim.ImageRequest(index, airsim.ImageType.SurfaceNormals , False, False)])[0]
        #,self.client.simGetImages([airsim.ImageRequest(index, airsim.ImageType.Infrared , False, False)])[0]
        ])
    types = [
        'Scene'
        #,'DepthPlanner '
        #,'DepthPerspective '
        ,'DepthVis '
        #,'DisparityNormalized '
        ,'Segmentation '
        ,'SurfaceNormals '
        #,'Infrared '
    ]
    for idx, response in enumerate(responses):
        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        # write to png 
        airsim.write_png(os.path.join(folderPath, types[idx] + '.png'), img_rgb)

  def snapAerial(self, path, altitude=-500):
  
    self.client.takeoffAsync().join()
    self.client.hoverAsync().join() 
    time.sleep(2)
    pose = self.client.simGetVehiclePose()
    pose.position.z_val = altitude 
    self.client.simSetVehiclePose(pose, ignore_collison=True)
    self.client.hoverAsync().join() 
    time.sleep(2)
    self.takePictures(path, index=3)
    pose.position.z_val = 0
    self.client.simSetVehiclePose(pose, ignore_collison=True)
    self.client.hoverAsync().join() 
    time.sleep(2)
    self.client.landAsync().join()

  def getPos(self):
    v3r = self.getState()['state'].kinematics_estimated.position.to_numpy_array()
    v3r[2] *= -1
    return v3r