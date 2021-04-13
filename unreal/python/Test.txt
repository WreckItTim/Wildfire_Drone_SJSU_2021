%run D:\AirSim\PythonClient\multirotor\setup_path.py
import airsim
import numpy as np
import os
import tempfile
import pprint
import cv2
import pickle
import time

USER = 'Tim'
secondsSinceEpoch = time.time()
timeObj = time.localtime(secondsSinceEpoch)
timeStamp = '%d-%d-%d %d-%d-%d' % (timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
runPath = USER + ' ' + timeStamp
os.mkdir(runPath)
photosPath = os.path.join(runPath, 'Photos')
os.mkdir(photosPath)
logPath = os.path.join(runPath, 'Log')
os.mkdir(logPath)
currentPos = (0, 0, 0)
class Client:
    _client = None
    def get():
        return Client._client
    def _set(c):
        Client._client = c
class Action:
    num = 0
    _log = {}
    def log(s):
        Action._log[Action.num] = s
        Action.num += 1

def connect():
    Client._set(airsim.MultirotorClient())
    Client.get().confirmConnection()
    Client.get().enableApiControl(True)
    Client.get().armDisarm(True)
    #takePictures(str(Action.num) + '_' + 'connect')
    Action.log('connect()')
    
def disconnect():
    Client.get().armDisarm(False)
    Client.get().reset()
    Client.get().enableApiControl(False)
    Client._set(None)

def getState():
    info = {}
    info['state'] = Client.get().getMultirotorState()
    info['imu'] = Client.get().getImuData()
    info['barometer'] = Client.get().getBarometerData()
    info['magnetometer'] = Client.get().getMagnetometerData()
    info['gps'] = Client.get().getGpsData()
    takePictures(str(Action.num) + '_' + 'getState')
    Action.log('getState()')
    return info

def takeOff():
    Client.get().takeoffAsync().join()
    takePictures(str(Action.num) + '_' + 'takeOff')
    Action.log('takeOff()')
    
def moveTo(x=currentPos[0], y=currentPos[1], z=currentPos[2], speed=5):
    Client.get().moveToPositionAsync(x, y, z, speed).join()
    currentPos = (x, y, z)
    takePictures(str(Action.num) + '_' + 'moveTo')
    Action.log(f'moveTo(x={x}, y={y}, z={z}, speed={speed})')
    
def takePictures(folder):
    responses = Client.get().simGetImages([
        Client.get().simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene , False, False)])[0]
        ,Client.get().simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner , False, False)])[0]
        ,Client.get().simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective , False, False)])[0]
        ,Client.get().simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis , False, False)])[0]
        ,Client.get().simGetImages([airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized , False, False)])[0]
        ,Client.get().simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation , False, False)])[0]
        ,Client.get().simGetImages([airsim.ImageRequest("0", airsim.ImageType.SurfaceNormals , False, False)])[0]
        ,Client.get().simGetImages([airsim.ImageRequest("0", airsim.ImageType.Infrared , False, False)])[0]
        ]) 
    types = [
        'Scene'
        ,'DepthPlanner '
        ,'DepthPerspective '
        ,'DepthVis '
        ,'DisparityNormalized '
        ,'Segmentation '
        ,'SurfaceNormals '
        ,'Infrared '
    ]
    path = os.path.join(photosPath, folder)
    os.mkdir(path)
    for idx, response in enumerate(responses):
        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        # write to png 
        airsim.write_png(os.path.join(path, types[idx] + '.png'), img_rgb)


connect()
print('INITIAL STATE:\n', getState(), '\n', '\n')
takeOff()
moveTo(x=10)
moveTo(y=10)
moveTo(z=-20)
moveTo(x=0, y=0, z=0, speed=20)
print('FINAL STATE:\n', getState(), '\n', '\n')
disconnect()
pickle.dump(Action._log, open(os.path.join(logPath, 'actionLog.p'), 'wb'))
print('DONE')
Action._log