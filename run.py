import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
from torchvision import models

from frames_dataset import FramesDataset

from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector, HEEstimator
from modules.hopenet import Hopenet

import torch

from train import train
from select_source import save_source_paths

if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train",])
    parser.add_argument("--gen", default="original", choices=["original", "spade"])
    parser.add_argument("--save", default='none')
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--tb_dir", default='tensorboard', help="path to tensorboard into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0, 1, 2, 3, 4, 5, 6, 7", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        if opt.save == "none":
            log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
            log_dir += '_' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
        else:
            log_dir = os.path.join(opt.log_dir, opt.save)
            log_dir += '_' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
    print(f'log_dir: {log_dir}')

    if opt.save == "none":
        tb_dir = os.path.join(opt.tb_dir, os.path.basename(opt.config).split('.')[0]) 
        tb_dir += '_' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
    else:
        tb_dir = os.path.join(opt.tb_dir, opt.save)
        tb_dir += '_' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
    print(f'tb_dir: {tb_dir}')
    
    hopenet = None
    if config['train_params']['loss_weights']['headpose'] != 0:
        hopenet = Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        print('Loading hopenet')
        hopenet_state_dict = torch.load(config['train_params']['hopenet_snapshot'])
        hopenet.load_state_dict(hopenet_state_dict)
        if torch.cuda.is_available():
            hopenet = hopenet.cuda()
            hopenet.eval()

    source_paths = None
    if config['model_params']['common_params']['num_source'] > 1:
        source_paths = save_source_paths(hopenet, config['dataset_params']['root_dir'], is_train=(opt.mode == 'train'))


    if opt.gen == 'original':
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    elif opt.gen == 'spade':
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])

    if torch.cuda.is_available():
        print('cuda is available')
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(discriminator)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])

    if opt.verbose:
        print(kp_detector)

    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])

    if torch.cuda.is_available():
        he_estimator.to(opt.device_ids[0])

    dataset = FramesDataset(is_train=(opt.mode == 'train'), source_paths=source_paths, num_source=config['model_params']['common_params']['num_source'], **config['dataset_params'])
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, kp_detector, he_estimator, opt.checkpoint, log_dir, tb_dir, dataset, opt.device_ids, hopenet)
