# @Courtney

from __future__ import absolute_import, division, print_function
import os
import numpy as np
import PIL.Image as pil

import torch
from torchvision import transforms

from depth import networks
from layers import disp_to_depth

import matplotlib as mpl
import matplotlib.cm as cm

# tim edit, has global methods and variables to access
import utils
from vision import Vision

# TODO: this didn't work for me with Vision due to num of arguments CN 210413
class MonoDepth2():
    """
    Apply MonoDepth2 to image input as a file path
    """

    def __init__(self):
        #print('MonoDepth2 Depth obj created...')

        self.model_name = "mono_640x192"
        self.encoder_path = os.path.join("depth/models", self.model_name, "encoder.pth")
        self.depth_decoder_path = os.path.join("depth/models", self.model_name, "depth.pth")

        # LOADING PRETRAINED MODEL
        self.encoder = networks.ResnetEncoder(18, False)
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        self.loaded_dict_enc = torch.load(self.encoder_path, map_location='cpu')
        self.filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(self.filtered_dict_enc)

        self.loaded_dict = torch.load(self.depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(self.loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval()

    def transform(self, read_from_path, write_to_path=None):
        #print('MonoDepth2 transform()')

        # Load test image and preprocessing
        image_path = read_from_path

        if os.path.isfile(image_path):
          # Only testing on a single image
          paths = [image_path]
          output_directory = os.path.dirname(image_path)
        elif os.path.isdir(image_path):
          # Searching folder for images
          paths = glob.glob(os.path.join(image_path, '*.{}'.format(ext)))
          output_directory = image_path
        else:
          raise Exception("Can not find args.image_path: {}".format(image_path))

        print("-> Predicting on {:d} test images".format(len(paths)))

        with torch.no_grad():
            for idx, image_path in enumerate(paths):

                if image_path.endswith("_disp.jpg"):
                    # don't try to predict disparity for a disparity image!
                    continue

                # Load image and preprocess
                input_image = pil.open(image_path).convert('RGB')
                original_width, original_height = input_image.size

                feed_height = self.loaded_dict_enc['height']
                feed_width = self.loaded_dict_enc['width']

                input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                # PREDICTION
                with torch.no_grad():
                    features = self.encoder(input_image)
                    outputs = self.depth_decoder(features)

                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

                # Saving numpy file               tim edit - skip for now will turn back on later
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                #name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                #scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
                #np.save(name_dest_npy, scaled_disp.cpu().numpy())

                # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 50)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='binary_r')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)

                # tim edit - save to write path if specified
                if write_to_path is not None:
                    name_dest_im = write_to_path
                else:
                    name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
                im.save(name_dest_im)

                print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                    idx + 1, len(paths), name_dest_im))

        print('-> Done!')


depth = MonoDepth2()
depth.transform("unreal/runs/tim 24-3-2021 14-11-13/photos/1/Scene.png")
