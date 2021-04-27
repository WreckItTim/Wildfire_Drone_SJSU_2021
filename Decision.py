# @Tim
import utils
import Drone
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from tensorflow.keras.preprocessing import image
from keras import layers, regularizers, optimizers, activations
from keras.layers import Flatten, Dense
from keras.models import Model



class Decision:
  dirMap = {0:'r', 1:'l', 2:'f', 3:'b', 4:'d', 5:'u'}
  def __init__(self):
    print('Parent Decision obj created...')
  def decide(self, args):
    print('Decision decide() not set!')

class Input(Decision):

  def __init__(self):
    print('Input Decision obj created...')
  def decide(self, args):
    timestep = args['timestep']
    drone = args['drone']
    command = input('enter command:').lower()
    if command == 'takeoff':
      drone.takeOff()
      drone.actions[timestep] = 'takeOff'
    if command == 'land':
      drone.land()
      drone.actions[timestep] = 'land'
    if command == 'stream':
      drone.liveStream()
      drone.actions[timestep] = 'stream'
    if command == 'command':
      command = input('enter drone command string:').lower()
      drone.command(command)
      drone.actions[timestep] = 'command ' + command
    if command in ['r', 'l', 'f', 'b', 'd', 'u']:
      if command == 'r':
        drone.move(0, drone.distance, 0)
      if command == 'l':
        drone.move(0, -1*drone.distance, 0)
      if command == 'f':
        drone.move(drone.distance, 0, 0)
      if command == 'b':
        drone.move(-1*drone.distance, 0, 0)
      if command == 'd':
        drone.move(0, 0, -1*drone.distance)
      if command == 'u':
        drone.move(0, 0, drone.distance)
      drone.actions[timestep] = 'move ' + command + ' ' + str(drone.distance)
    if command == 'flip':
      fdir = random.choice(['l', 'r', 'f', 'b'])
      drone.flip(fdir)
      drone.actions[timestep] = 'flip {fdir}'
    if command == 'connect':
      drone.connect()
      drone.actions[timestep] = 'connect'
    if command == 'disconnect':
      drone.disconnect()
      drone.actions[timestep] = 'disconnect'
    if command == 'snap':
      drone.actions[timestep] = 'snap'
    if command == 'goal':
      args['progress'] = 'goal'
    return args



class Path(Decision):

  def __init__(self):
    print('TakePath Decision obj created...')
  def decide(self, args):
    timestep = args['timestep']
    drone = args['drone']
    nextPoint = args['path'][timestep]
    drone.moveTo(float(nextPoint[0]), float(nextPoint[1]), float(nextPoint[2]))
    drone.actions[timestep] = 'moveTo {nextPoint[0]} {nextPoint[1]} {nextPoint[2]} {drone.speed}'
    args['progress'] = 'path'
    if timestep >= len(args['path']) - 1:
      args['progress'] =  'goal'
    return args
    


class Deep(Decision):

  def __init__(self, mdl_name='nasnet'):
    print(mdl_name, 'Deep Decision obj created...')
    from keras.applications.nasnet import NASNetLarge as loaded_model
    from keras.applications.nasnet import preprocess_input
    self.preproc = preprocess_input
    model = loaded_model(include_top=False, weights='imagenet')
    self.img_shape = (331, 331)
    # https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    output = Dense(6, activation='softmax')(flat1)
    # define new model
    self.model = Model(inputs=model.inputs, outputs=output)
    

  def decide(self, args):
    timestep = args['timestep']
    drone = args['drone']

    # inference
    img = image.load_img(args['depth_writePath'])
    img = img.resize(self.img_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = self.preproc(x)
    preds = self.model.predict(x)
    direction = self.dirMap[np.argmax(preds, axis=1).astype('uint8')[0]]
    if direction == 'r': drone.move(0, drone.distance, 0)
    if direction == 'l': drone.move(0, -1*drone.distance, 0)
    if direction == 'f': drone.move(drone.distance, 0, 0)
    if direction == 'b': drone.move(-1*drone.distance, 0, 0)
    if direction == 'd': drone.move(0, 0, -1*drone.distance)
    if direction == 'u': drone.move(0, 0, drone.distance)
    drone.actions[timestep] = 'move ' + direction + ' ' + str(drone.distance)
    print('move ' + direction + ' ' + str(drone.distance))
    
    args['progress'] = 'path'
    if timestep >= args['maxTimestep']:
      args['progress'] = 'goal'
    
    return args
 

class Rewards_v1(Decision):

  def __init__(self):
    print('Rewards Obj created')

  def decide(self, args):
    timestep = args['timestep']
    drone = args['drone']
    path = args['path']
    nextPoint = args['nextPoint']
    lastPoint = args['lastPoint']
    startPoint = args['startPoint']
    endPoint = args['endPoint']
    pathstep = args['pathstep']
    nSteps = args['nSteps']
    lastDirection = args['lastDirection']
    visions = args['visions']
    coeffecients = args['coefficients']

    ongoing_rewards = {
        'left':0
        ,'right':0
        ,'up':0
        ,'down':0
        ,'forward':0
    }
    pos_change = {
        'left':np.array([0, -1*drone.distance, 0])
        ,'right':np.array([0, drone.distance, 0])
        ,'up':np.array([0, 0, drone.distance])
        ,'down':np.array([0, 0, -1*drone.distance])
        ,'forward':np.array([drone.distance, 0, 0])
    }

    # get vision rewards
    for vision in visions:
        rewards_per_direction = visions[vision].reward(args[vision + '_writePath'], args[vision + '_rewardsPath'])
        for direction in rewards_per_direction:
            ongoing_rewards[direction] += coeffecients[vision] * rewards_per_direction[direction]
    
    # get path rewards
    currentPosition = drone.getPos().astype(int)
    distances = {}
    maxDistance = 0
    minDistance = 9999999999
    path_length = np.linalg.norm(nextPoint - lastPoint)
    for direction in ongoing_rewards:
        pos = currentPosition + pos_change[direction]
        distances[direction] = np.linalg.norm(nextPoint - pos) + np.linalg.norm(lastPoint - pos)
        maxDistance = max(distances[direction], maxDistance)
        minDistance = min(distances[direction], minDistance)
    for direction in ongoing_rewards:
        this_reward = coeffecients['path'] * (1 - (distances[direction] - minDistance) / (maxDistance - minDistance))
        #this_reward = -1 * coeffecients['path'] * (distances[direction] / path_length) ** 2
        ongoing_rewards[direction] += this_reward

    # get objective rewards
    distances = {}
    maxDistance = 0
    minDistance = 9999999999
    for direction in ongoing_rewards:
        pos = currentPosition + pos_change[direction]
        distances[direction] = np.linalg.norm(endPoint - pos) + np.linalg.norm(endPoint - pos)
        maxDistance = max(distances[direction], maxDistance)
        minDistance = min(distances[direction], minDistance)
    for direction in ongoing_rewards:
        ongoing_rewards[direction] += coeffecients['objective'] * (1 - (distances[direction] - minDistance) / (maxDistance - minDistance))

    # get smoothness rewards
    opposite = {
        'left':'right'
        ,'right':'left'
        ,'up':'down'
        ,'down':'up'
        ,'forward':'backward'
    }
    if lastDirection != '':
        for direction in ongoing_rewards:
            x = 0
            if lastDirection == direction:
                x = 1
            x = 1
            if opposite[lastDirection] == direction:
                x = 0
            ongoing_rewards[direction] += coeffecients['smooth'] * x
    
    # find optimal choice
    optimal_direction = 'forward'
    max_rewards = ongoing_rewards['forward']
    for direction in ongoing_rewards:
        if ongoing_rewards[direction] > max_rewards:
            max_rewards = ongoing_rewards[direction]
            optimal_direction = direction
    
    # make optimal choice
    if optimal_direction == 'right': drone.move(0, drone.speed, 0)
    if optimal_direction == 'left': drone.move(0, -1*drone.speed, 0)
    if optimal_direction == 'forward': drone.move(drone.speed, 0, 0)
    if optimal_direction == 'down': drone.move(0, 0, -1*drone.speed)
    if optimal_direction == 'up': drone.move(0, 0, drone.speed)
    args['lastDirection'] = optimal_direction

    # update path
    currentPosition = drone.getPos().astype(int)
    last_distance = np.linalg.norm(lastPoint - currentPosition)
    if pathstep < nSteps:
        next_distance = np.linalg.norm(nextPoint - currentPosition)
        if next_distance < drone.distance:
            args['lastPoint'] = nextPoint
            args['nextPoint'] = args['path'][pathstep + 1]
            args['pathstep'] += 1
    print('go', optimal_direction, 'path', currentPosition, 'from', args['lastPoint'], 'to', args['nextPoint'], 'at', args['pathstep'], 'outta', nSteps)
    '''
    currentPosition = drone.getPos().astype(int)
    last_distance = np.linalg.norm(lastPoint - currentPosition)
    if pathstep < nSteps:
        next_distance = np.linalg.norm(nextPoint[:2] - currentPosition[:2])
        next_next_distance = np.linalg.norm(args['path'][pathstep + 1] - currentPosition)
        if next_distance < 2*drone.distance or next_next_distance < last_distance:
            args['lastPoint'] = nextPoint
            args['nextPoint'] = args['path'][pathstep + 1]
            args['pathstep'] += 1
    print('go', optimal_direction, 'path', currentPosition, 'from', args['lastPoint'], 'to', args['nextPoint'], 'at', args['pathstep'], 'outta', nSteps)
    '''

    # check if at objective
    args['progress'] = 'path'
    objective_distance = np.linalg.norm(endPoint - currentPosition)
    if objective_distance < 2*drone.distance:
      args['progress'] = 'goal'
    
    return args


    

class Rewards_v2(Decision):

  def __init__(self):
    print('Rewards Obj created')

  def decide(self, args):
    timestep = args['timestep']
    drone = args['drone']
    path = args['path']
    nextPoint = args['nextPoint']
    lastPoint = args['lastPoint']
    startPoint = args['startPoint']
    endPoint = args['endPoint']
    pathstep = args['pathstep']
    nSteps = args['nSteps']
    lastDirection = args['lastDirection']
    visions = args['visions']
    coeffecients = args['coefficients']

    ongoing_rewards = {
        'left':0
        ,'right':0
        ,'up':0
        ,'down':0
        ,'forward':0
    }
    pos_change = {
        'left':np.array([0, -1*drone.distance, 0])
        ,'right':np.array([0, drone.distance, 0])
        ,'up':np.array([0, 0, drone.distance])
        ,'down':np.array([0, 0, -1*drone.distance])
        ,'forward':np.array([drone.distance, 0, 0])
    }

    # get vision rewards
    for vision in visions:
        rewards_per_direction = visions[vision].reward(args[vision + '_writePath'], args[vision + '_rewardsPath'])
        for direction in rewards_per_direction:
            ongoing_rewards[direction] += coeffecients[vision] * rewards_per_direction[direction]
    
    # get path rewards
    currentPosition = drone.getPos().astype(int)
    distances = {}
    maxDistance = 0
    minDistance = 9999999999
    path_length = np.linalg.norm(nextPoint - lastPoint)
    for direction in ongoing_rewards:
        pos = currentPosition + pos_change[direction]
        distances[direction] = np.linalg.norm(nextPoint - pos) + np.linalg.norm(lastPoint - pos)
        maxDistance = max(distances[direction], maxDistance)
        minDistance = min(distances[direction], minDistance)
    for direction in ongoing_rewards:
        this_reward = coeffecients['path'] * (1 - (distances[direction] - minDistance) / (maxDistance - minDistance))
        #this_reward = -1 * coeffecients['path'] * (distances[direction] / path_length) ** 2
        ongoing_rewards[direction] += this_reward

    # get objective rewards
    distances = {}
    maxDistance = 0
    minDistance = 9999999999
    for direction in ongoing_rewards:
        pos = currentPosition + pos_change[direction]
        distances[direction] = np.linalg.norm(endPoint - pos) + np.linalg.norm(endPoint - pos)
        maxDistance = max(distances[direction], maxDistance)
        minDistance = min(distances[direction], minDistance)
    for direction in ongoing_rewards:
        ongoing_rewards[direction] += coeffecients['objective'] * (1 - (distances[direction] - minDistance) / (maxDistance - minDistance))

    # get smoothness rewards
    opposite = {
        'left':'right'
        ,'right':'left'
        ,'up':'down'
        ,'down':'up'
        ,'forward':'backward'
    }
    if lastDirection != '':
        for direction in ongoing_rewards:
            x = 1
            if opposite[lastDirection] == direction:
                x = 0
            ongoing_rewards[direction] += coeffecients['smooth'] * x
    
    # find optimal choice
    optimal_direction = 'forward'
    max_rewards = ongoing_rewards['forward']
    for direction in ongoing_rewards:
        if ongoing_rewards[direction] > max_rewards:
            max_rewards = ongoing_rewards[direction]
            optimal_direction = direction
    
    # make optimal choice
    if optimal_direction == 'right': drone.move(0, drone.speed, 0)
    if optimal_direction == 'left': drone.move(0, -1*drone.speed, 0)
    if optimal_direction == 'forward': drone.move(drone.speed, 0, 0)
    if optimal_direction == 'down': drone.move(0, 0, -1*drone.speed)
    if optimal_direction == 'up': drone.move(0, 0, drone.speed)
    args['lastDirection'] = optimal_direction

    # update path
    currentPosition = drone.getPos().astype(int)
    last_distance = np.linalg.norm(lastPoint - currentPosition)
    if pathstep < nSteps:
        next_distance = np.linalg.norm(nextPoint - currentPosition)
        if next_distance < drone.distance:
            args['lastPoint'] = nextPoint
            args['nextPoint'] = args['path'][pathstep + 1]
            args['pathstep'] += 1
    print('go', optimal_direction, 'path', currentPosition, 'from', args['lastPoint'], 'to', args['nextPoint'], 'at', args['pathstep'], 'outta', nSteps)
    '''
    currentPosition = drone.getPos().astype(int)
    last_distance = np.linalg.norm(lastPoint - currentPosition)
    if pathstep < nSteps:
        next_distance = np.linalg.norm(nextPoint[:2] - currentPosition[:2])
        next_next_distance = np.linalg.norm(args['path'][pathstep + 1] - currentPosition)
        if next_distance < 2*drone.distance or next_next_distance < last_distance:
            args['lastPoint'] = nextPoint
            args['nextPoint'] = args['path'][pathstep + 1]
            args['pathstep'] += 1
    print('go', optimal_direction, 'path', currentPosition, 'from', args['lastPoint'], 'to', args['nextPoint'], 'at', args['pathstep'], 'outta', nSteps)
    '''

    # check if at objective
    args['progress'] = 'path'
    objective_distance = np.linalg.norm(endPoint - currentPosition)
    if objective_distance < 2*drone.distance:
      args['progress'] = 'goal'
    
    return args