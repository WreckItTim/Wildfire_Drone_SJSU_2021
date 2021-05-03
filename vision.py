# @Tim
import cv2
import utils
import numpy as np

class Vision:

    def __init__(self):
        print('Vision obj created!')

    def transform(self, read_from_path, write_to_path=None):
        print('Vision transform() not set!')

    def reward(self, read_from_path, write_to_path=None):
        img = cv2.imread(read_from_path, cv2.IMREAD_GRAYSCALE)/255
        height = img.shape[0]
        width = img.shape[1]

        height_cut = int(height/3)
        width_cut = int(width/3)
        positions = {
            'left':(0, 1)
            ,'right':(2, 1)
            ,'up':(1, 0)
            ,'down':(1, 2)
            ,'forward':(1, 1)
        }
        rewards = {
            'left':0
            ,'right':0
            ,'up':0
            ,'down':0
            ,'forward':0
        }

        for direction in positions:
            x = positions[direction][0]
            y = positions[direction][1]
            img_cut = img[y*height_cut:(y+1)*height_cut, x*width_cut:(x+1)*width_cut]
            cv2.imwrite(direction + '.png', img_cut*255)
            reward = 1 - np.mean(img_cut)
            rewards[direction] = reward

        img = img * 255
        if write_to_path is not None:
          for direction in positions:
            x = int(width_cut*positions[direction][0] + width_cut/2.5)
            y = int(height_cut*positions[direction][1] + height_cut/2)
            #bufferSize = 64
            #cv2.rectangle(img, (x-bufferSize, y-bufferSize), (x + bufferSize, y + bufferSize), (0,0,0), -1)
            cv2.putText(img, str(round(rewards[direction],2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 20, cv2.LINE_AA)
            cv2.putText(img, str(round(rewards[direction],2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5, cv2.LINE_AA)
          cv2.imwrite(write_to_path, img)

        return rewards
