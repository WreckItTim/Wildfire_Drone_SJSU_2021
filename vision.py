# @Tim
import cv2
import utils

class Vision:

    def __init__(self):
        print('Vision obj created!')

    def transform(self, read_from_path, write_to_path=None):
        print('Vision transform() not set!')

    def reward(self, read_from_path):
        img = cv2.imread(read_from_path, cv2.IMREAD_GRAYSCALE)/255
        height = img.shape[0]
        width = img.shape[1]

        height_cut = int(height/3)
        width_cut = int(width/3)
        directions = {
            'left':(0, 1)
            ,'right':(2, 1)
            ,'up':(1, 0)
            ,'down':(1, 2)
            ,'forward':(1, 1)
               }

        for direction in directions:
            x = directions[direction][0]
            y = directions[direction][1]
            img_cut = img[y*height_cut:(y+1)*height_cut, x*width_cut:(x+1)*width_cut]
            reward = 1 - np.mean(img_cut)
            directions[direction] = reward
        return directions
