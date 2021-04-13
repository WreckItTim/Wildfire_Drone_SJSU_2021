<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:43d32597e172a1f1e9d1c7a05d820890a963196ef8142bf039e682579acfc4b0
size 2752
=======
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
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = nn(in_channels=3, out_channels=3).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        self.load_checkpoint(torch.load('my_checkpoint.pth.tar', map_location='cpu'), self.model)

    def transform(self, read_from_path, write_to_path):
        # tim edit - writing to specific file path with full extension, no need to create directory
        #if not os.path.exists(write_to_path):
            #os.mkdir(write_to_path)

        height, width = 640, 360
        transforms = A.Compose([
            A.Resize(height=height, width=width),
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
            preds = (preds > 0.5).float()
            
        # tim edit - cast image to global format (it is messy, there is probably a better way)
        img = preds.numpy()
        img = np.resize(img, (height, width))
        img = Image.fromarray(np.uint8(img*255))
        img = utils.convertPIL(img)
        img.save(write_to_path)
        #torchvision.utils.save_image(img, write_to_path)


    def load_checkpoint(self,checkpoint, model):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])




# TODO: test code!!
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = nn(in_channels=3, out_channels=1).to(DEVICE)
#unet = UNET(model, 'my_checkpoint2.pth.tar')
#unet.transform('tello/runs/tim 12-3-2021 15-25-16/photos/0_takePictures/Scene.png','saved_images')
>>>>>>> 3fc45c3a9152eb3c8b591b810d18d2b798a8ab7b
