from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater

from torchvision import transforms
from time import time
from modules.model import headpose_pred_to_degree
import numpy as np
from time import time

from torch.utils.tensorboard import SummaryWriter

# from select_source import select_source

def train(config, generator, discriminator, kp_detector, he_estimator, checkpoint, log_dir, tb_dir, dataset, device_ids, hopenet):
    tb_writer = SummaryWriter(log_dir=tb_dir)

    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_he_estimator = torch.optim.Adam(he_estimator.parameters(), lr=train_params['lr_he_estimator'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector, he_estimator,
                                      optimizer_generator, optimizer_discriminator, optimizer_kp_detector, optimizer_he_estimator)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_he_estimator = MultiStepLR(optimizer_he_estimator, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=16, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, he_estimator, generator, discriminator, train_params, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'], hopenet=hopenet, num_source=config['model_params']['common_params']['num_source'])
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq'], num_source=config['model_params']['common_params']['num_source']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            # start = time()
            batch_idx = 0
            for x in dataloader:
                # print(time() - start)
                # start = time()
                
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
                optimizer_he_estimator.step()
                optimizer_he_estimator.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

                for i in range(len(logger.names)):
                    tb_writer.add_scalar("iter_"+logger.names[i], np.array(logger.loss_list)[-1][i], epoch * len(dataloader) + batch_idx)
                batch_idx += 1


            avg_losses = np.array(logger.loss_list).mean(0)
            for i in range(len(logger.names)):
                tb_writer.add_scalar(logger.names[i], avg_losses[i], epoch)
            
            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            scheduler_he_estimator.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'he_estimator': he_estimator,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_he_estimator': optimizer_he_estimator}, inp=x, out=generated)
    print(f'finish to train (tb_dir: {tb_dir})')
    print(f'zero frames: {dataset.dataset.zero_frame}')

def get_head_pose(hopenet, source):
    transform_hopenet =  transforms.Compose([transforms.Resize(size=(224, 224)),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    source_224 = transform_hopenet(source)
    yaw, pitch, roll = hopenet(source_224)
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)
    return np.array([yaw.cpu().detach().numpy(), pitch.cpu().detach().numpy(), roll.cpu().detach().numpy()])

