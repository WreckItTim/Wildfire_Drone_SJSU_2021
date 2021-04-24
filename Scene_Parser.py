# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from scene_parser.mit_semseg.dataset import TestDataset
from scene_parser.mit_semseg.models import ModelBuilder, SegmentationModule
from scene_parser.mit_semseg.utils import colorEncode, find_recursive, setup_logger
from scene_parser.mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from scene_parser.mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from scene_parser.mit_semseg.config import cfg

# Wildfire Libs
from vision import Vision




class SceneParser(Vision):
    def __init__(self):
        cfg.merge_from_file("scene_parser/config/ade20k-resnet50dilated-ppm_deepsup.yaml")

        cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
        cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

        cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
        cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

        assert os.path.exists(cfg.MODEL.weights_encoder) and \
               os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

        # Network Builders
        net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder,
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder)
        net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder,
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder,
            use_softmax=True)

        crit = nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

        self.colors = loadmat('scene_parser/data/color150.mat')['colors']
        self.names = {}
        self.valid_indices = [2, 3, 5, 7, 1, 14, 17, 22, 26, 27, 30, 39, 44, 47, 49, 61, 62, 69, 91, 94, 127, 129]
        with open('scene_parser/data/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(";")[0]

    def visualize_result(self, data, pred, cfg, folder_name):
        (img, info) = data

        # print predictions in descending order
        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        print("Predictions in [{}]:".format(info))
        for idx in np.argsort(counts)[::-1]:
            if uniques[idx] + 1 in self.valid_indices:
                name = self.names[uniques[idx] + 1]
                ratio = counts[idx] / pixs * 100
                if ratio > 0.5:
                    print("  {}: {:.2f}%".format(name, ratio))

        # colorize prediction
        pred_color = colorEncode(pred, self.colors, self.valid_indices).astype(np.uint8)

        img_name = info.split('/')[-1]
        pred_color = Image.fromarray(pred_color, 'RGB')
        pred_color.save(os.path.join(folder_name, img_name.replace('.jpg', '.png')))

    def transform_step(self,loader,gpu):
        self.segmentation_module.eval()

        pbar = tqdm(total=len(loader))
        for batch_data in loader:
            # process data
            batch_data = batch_data[0]
            segSize = (batch_data['img_ori'].shape[0],
                       batch_data['img_ori'].shape[1])
            img_resized_list = batch_data['img_data']

            with torch.no_grad():
                scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
                # scores = async_copy_to(scores, gpu)

                for img in img_resized_list:
                    feed_dict = batch_data.copy()
                    feed_dict['img_data'] = img
                    del feed_dict['img_ori']
                    del feed_dict['info']
                    # feed_dict = async_copy_to(feed_dict, gpu)

                    # forward pass
                    pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
                    scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

                _, pred = torch.max(scores, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())

            # visualization
            if not os.path.exists(cfg.TEST.result):
                os.mkdir(cfg.TEST.result)
            self.visualize_result(
                (batch_data['img_ori'], batch_data['info']),
                pred,
                cfg,
                cfg.TEST.result
            )
            pbar.update(1)

    def transform(self, read_from_path, write_to_path=None):
        #leave this here so model can read folders and images.
        if os.path.isdir(read_from_path):
            imgs = find_recursive(read_from_path)
        else:
            imgs = [read_from_path]
        assert len(imgs), "imgs should be a path to image (.jpg) or directory."
        cfg.list_test = [{'fpath_img': x} for x in imgs]

        cfg.TEST.result = write_to_path
        if not os.path.isdir(cfg.TEST.result):
            os.makedirs(cfg.TEST.result)

        dataset_test = TestDataset(
            cfg.list_test,
            cfg.DATASET)
        loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=cfg.TEST.batch_size,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=5,
            drop_last=True)

        self.transform_step(loader_test,0)
        print('Inference done!')


# if __name__ == "__main__":
#     pp = SceneParser()
#     pp.transform('unreal_fire_ex.png', 'saved_images/')

