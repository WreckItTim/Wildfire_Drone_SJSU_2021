# @Tim

import time
import os
from PIL import Image
import Aerial, Drone, Depth, Segmentation, Decision

# prompt for user name
user = input("Enter user name: ").lower()

# prompt Obi-Wan to say hello
Drone.hello()

# set aerial fire module
aerialFire = Aerial.Fire()

# set aerial object module
aerialObjects = Aerial.Objects()

# set vision modules
visions = {
	'fire' : Aeriel.Fire(),
	'depth' : Depth.MonoDepth2(),
	'smoke' : Segmentation.UNET()
}

# set decision module
#decision = Decision.Input()
#decision = Decision.Path()
#decision = Decision.Deep()
decision = Decision.Rewards()

# set RL reward coefficients
coefficients = {
	'fire' : 1,
	'depth' : 1,
	'smoke' : 1,
	'path' : 1,
	'objective' : 1,
	'smoothness' : 1
}

# set drone module
drone_name = input('running from tello or unreal?').lower()
while drone_name != 'tello' and drone_name != 'unreal':
  drone_name = input('try again. running from tello or unreal?').lower()
if drone_name == 'tello':
  drone = Drone.Tello()
elif drone_name == 'unreal':
  drone = Drone.Unreal()
  drone.speed = 5

# create unique folder for this run - to log and store data
secondsSinceEpoch = time.time()
timeObj = time.localtime(secondsSinceEpoch)
timeStamp = '%d-%d-%d %d-%d-%d' % (timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
drone.runPath = drone_name  + '/runs/' + user + ' ' + timeStamp
os.mkdir(drone.runPath)
drone.photosPath = os.path.join(drone.runPath, 'photos')
os.mkdir(drone.photosPath)
drone.logPath = os.path.join(drone.runPath, 'log')
os.mkdir(drone.logPath)

# set path (later will be replaced by a module)
if drone_name == 'tello':
	path = np.array([
		[0, 0, 0]
		,[0, -50, 0]
		,[0, -250, 0]
		,[0, -400, 0]
	])
elif drone_name == 'unreal':
	path = np.array([
		[0, 0, 0]
		,[10, 0, 0]
		,[10, 0, 10]
		,[120, -40, 10]
		,[120, -80, 10]
		,[170, -70, -20]
	])

# connect to drone
drone.connect()
time.sleep(2)
#args['framesPath'] = os.path.join(drone.photosPath, str('frames'))
#os.mkdir(args['framesPath'])
#drone.liveStream({'MonoDepth2':depth, 'ColorFire':aerialFire, 'UNET':segmentation}, args['framesPath'])
#drone.liveStream(None, None)

# snap aerial photos
#args['aerialPath'] = os.path.join(drone.photosPath, '0')
#os.mkdir(args['aerialPath'])
#drone.snapAerial(args['aerialPath'])

# get paths to aerial photos
'''if drone_name == 'tello':
	args['aerialObjects_readPath'] = os.path.join(args['aerialPath'], 'SatelliteObjects.png')
	args['aerialFire_readPath'] = os.path.join(args['aerialPath'], 'SatelliteFire.png')
if drone_name == 'unreal':
	args['aerialObjects_readPath'] = os.path.join(args['aerialPath'], 'Scene.png')
	args['aerialFire_readPath'] = os.path.join(args['aerialPath'], 'Scene.png')
args['aerialObjects_writePath'] = os.path.join(args['aerialPath'], 'aerialObjects.png')
args['aerialFire_writePath'] = os.path.join(args['aerialPath'], 'aerialFire.png')

# transform aerial photos
aerialObjects.transform(args['aerialObjects_readPath'], args['aerialObjects_writePath'])
aerialFire.transform(args['aerialFire_readPath'], args['aerialFire_writePath'])
'''
# make decisions
sample_rate = 0 # make decision after this many seconds
drone.takeOff()
#drone.moveTo(0, 0, 100)
args = {}
args['timestep'] = 0
args['drone'] = drone
args['path'] = path
args['nextPoint'] = path[1]
args['lastPoint'] = path[0]
args['startPoint'] = path[0]
args['endPoint'] = path[-1]
args['pathstep'] = 1
args['nSteps'] = path.shape[0]
args['lastDirection'] = ''
args['visions'] = visions
args['coefficients'] = coefficients
stepthrough = input('Next Timestep?')
while(True):

  # move one timestep up
  args['timestep'] += 1
  args['timePath'] = os.path.join(drone.photosPath, str(timestep))
  os.mkdir(args['timePath'])

  # take photos for this timestep
  drone.takePictures(args['timePath'])
  
  # transform vision modules
  for vision in visions:
	  args[vision + '_readPath'] = os.path.join(args['timePath'], 'Scene.png')
	  args[vision + '_writePath'] = os.path.join(args['timePath'], vision + '.png')
	  visions[vision].transform(args[vision + '_readPath'], args[vision + '_writePath'])

  # make decision
  response = decision.decide(args)
  
  # wait for next time step
  time.sleep(sample_rate)

  # exit when reached end
  if response == 'goal':
    stepthrough = input('Finished! Exit?')
    if drone_name == 'tello':
      drone.flip()
    break

# clean up
drone.disconnect()
print('buayyyyyeeeee')