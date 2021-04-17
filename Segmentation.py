# @Angelica
import torch
import torchvision
import albumentations as A   #pip install albumentations
from albumentations.pytorch import ToTensorV2
from PIL import Image
from UNET_MODEL import UNET as nn
import os
import numpy as np
import torch.optim as optim
from skimage.transform import resize
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import glob

# tim edit - global functions and variables
import utils

class Segmentation:

    def __init__(self):
        print('Parent Segmentation obj created...')

    def transform(self, read_from_path, write_to_path):
        print('Segmentation transform() not set!')


class UNET(Segmentation):
    def __init__(self):
        # tim edit - added test stuff to init()
        # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = nn(in_channels=3, out_channels=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        self.load_checkpoint(torch.load('bin_checkpoint.pth.tar', map_location='cpu'), self.model)

    def transform(self, read_from_path, write_to_path):
        # tim edit - writing to specific file path with full extension, no need to create directory
        #if not os.path.exists(write_to_path):
            #os.mkdir(write_to_path)

        height, width = 360, 640
        transforms = A.Compose([
            #A.Resize(height=height, width=width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])

        # load image
        image = Image.open(read_from_path).convert('RGB')
        image = np.array(image)
        transf = transforms(image=image)
        image = transf['image'].unsqueeze(0)

        # make model predict!!
        self.model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(self.model(image))
            preds = (preds > 0.5).float().squeeze()

        # tim edit - cast image to global format (it is messy, there is probably a better way)
        img = preds.numpy()
        img = resize(img, (height, width), preserve_range=True)
        img = Image.fromarray(np.uint8(img*255))
        # img = utils.convertPIL(img)
        img.save(write_to_path)
        # torchvision.utils.save_image(img, f'{write_to_path}/{read_from_path}')


    def load_checkpoint(self,checkpoint, model):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])


# # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = UNET()
# model.transform('unreal_fire_ex.png', 'saved_images/pred.png')