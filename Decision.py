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
    drone.moveTo(nextPoint[0], nextPoint[1], nextPoint[2])
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
    
    args['progress'] =  = 'path'
    if timestep >= args['maxTimestep']:
      args['progress'] =  = 'goal'
    
    return args
    


class Rewards(Decision):

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
    coefficients = args['coefficients']

    ongoing_rewards = {
        'left':0
        ,'right':0
        ,'up':0
        ,'down':0
        ,'forward':0
    }
    pos_change = {
        'left':np.array([0, -1*drone.distance, 0])0
        ,'right':np.array([0, drone.distance, 0])0
        ,'up':np.array([0, 0, drone.distance])0
        ,'down':np.array([0, 0, -1*drone.distance])0
        ,'forward':np.array([drone.distance, 0, 0])0
    }

    # get vision rewards
    for vision in visions:
        rewards_per_direction = visions[vision].reward(args[vision + '_readPath'] )
        for direction in rewards_per_direction:
            ongoing_rewards[direction] += coeffecients[vision] * rewards_per_direction[direction]
    
    # get path rewards
    distances = {}
    maxDistance = 0
    for direction in ongoing_rewards:
        pos = drone.pos + pos_change[direction]
        distances[direction] = np.linalg.norm(nextPoint - pos) + np.linalg.norm(lastPoint - pos)
        maxDistance = max(distances[direction], maxDistance)
    for direction in ongoing_rewards:
        ongoing_rewards[direction] += coeffecients['path'] * distances[direction] / maxDir
    
    # get objective rewards
    distances = {}
    maxDistance = 0
    for direction in ongoing_rewards:
        pos = drone.pos + pos_change[direction]
        distances[direction] = np.linalg.norm(endPoint - pos) + np.linalg.norm(endPoint - pos)
        maxDistance = max(distances[direction], maxDistance)
    for direction in ongoing_rewards:
        ongoing_rewards[direction] += coeffecients['objective'] * distances[direction] / maxDir

    # get smoothness rewards
    for direction in ongoing_rewards:
        x = 0
        if 'forward' == direction or lastDirection == direction:
            x = 1
        ongoing_rewards[direction] += coeffecients['smoothness'] * x
    
    # find optimal choice
    optimal_direction = 'forward'
    max_rewards = ongoing_rewards['forward']
    for direction in ongoing_rewards:
        if ongoing_rewards[direction] > max_rewards:
            max_rewards = ongoing_rewards[direction]
            optimal_direction = direction
    
    # make optimal choice
    if optimal_direction == 'r': drone.move(0, drone.distance, 0)
    if optimal_direction == 'l': drone.move(0, -1*drone.distance, 0)
    if optimal_direction == 'f': drone.move(drone.distance, 0, 0)
    if optimal_direction == 'd': drone.move(0, 0, -1*drone.distance)
    if optimal_direction == 'u': drone.move(0, 0, drone.distance)
    if optimal_direction != 'forward':
        args['lastDirection'] = optimal_direction

    # update path
    if pathstep < nSteps:
        pos = drone.pos + pos_change[optimal_direction]
        last_distance = np.linalg.norm(lastPoint - pos)
        next_distance = np.linalg.norm(args['path'][pathstep + 1] - pos)
        if next_distance < last_distance:
            args['lastPoint'] = nextPoint
            args['nextPoint'] = args['path'][pathstep + 1]
            args['pathstep'] += 1

    # check if at objective
    args['progress'] = 'path'
    objective_distance = np.linalg.norm(endPoint - pos)
    if objective_distance <= args['objectiveEpsilon']:
      args['progress'] = 'goal'
    
    return args