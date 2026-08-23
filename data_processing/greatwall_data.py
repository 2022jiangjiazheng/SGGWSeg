# Project repository: https://github.com/2022jiangjiazheng

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import cv2
from osgeo import gdal
from torch.utils.data import Dataset
import torchvision
from data_processing.augmentations import MyWrap, Rotate, Bright, Noise
import torch
import numpy as np
import pickle


RASTER_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}


def patch_id(filename):
    """Match `sample__...` images with `sample_mask__...` labels across raster formats."""
    stem = os.path.splitext(filename)[0]
    return stem.replace("_mask__", "__")

class ToTensorZones(object):
    """Convert an image/mask pair to tensors expected by the model."""

    def __call__(self, image, zone):
        if image.dtype != np.uint8:
            raise TypeError(
                f"Expected an 8-bit image, but received {image.dtype}. "
                "Convert it to uint8 explicitly before training."
            )
        # Convert uint8 pixels from [0, 255] to float32 values in [0, 1].
        image = torch.from_numpy(image).float().div_(255.0)
        zone = torch.as_tensor(np.array(zone))
        zone[zone == 0] = 0
        zone[zone == 255] = 1

        return image, zone

class GreatwallDataset(Dataset):
    """Load paired image and binary-mask patches for SGGWSeg."""

    def __init__(self, mode, augmentation, data_dir, project_dir, bright, wrap, noise, rotate, hflip, vflip):
        '''
        :param mode: 'train', 'val' or 'test'
        :param augmentation: whether to use augmentation
        :param data_dir: the directory of the dataset
        :param project_dir: the directory of the code
        :param bright, wrap, noise, rotate, flip: the possibilty of bright, wrap, noise, rotate, flip
        '''
        self.images_path = os.path.join(data_dir, 'images', mode)
        self.zones_path = os.path.join(data_dir, 'zones', mode)

        image_files = {
            patch_id(name): name for name in os.listdir(self.images_path)
            if os.path.isfile(os.path.join(self.images_path, name))
            and os.path.splitext(name)[1].lower() in RASTER_EXTENSIONS
        }
        zone_files = {
            patch_id(name): name for name in os.listdir(self.zones_path)
            if os.path.isfile(os.path.join(self.zones_path, name))
            and os.path.splitext(name)[1].lower() in RASTER_EXTENSIONS
        }
        if image_files.keys() != zone_files.keys():
            missing_zones = sorted(image_files.keys() - zone_files.keys())
            missing_images = sorted(zone_files.keys() - image_files.keys())
            raise ValueError(
                f"Unpaired patches in {mode}: missing zones={missing_zones}, "
                f"missing images={missing_images}"
            )
        ordered_keys = sorted(image_files)
        self.imgs = [image_files[key] for key in ordered_keys]
        self.zones = [zone_files[key] for key in ordered_keys]


        # Shuffle once so limited runs sample patches from different Great Wall images.
        # However, always use same shuffle so that it is reproducible
        splits_dir='data_splits'

        if not os.path.exists(os.path.join(project_dir, 'data_processing', splits_dir)):
            os.makedirs(os.path.join(project_dir, 'data_processing', splits_dir))
        if not os.path.isfile(os.path.join(project_dir, 'data_processing', splits_dir, 'shuffle_' + mode + '.txt')):
            shuffle = np.random.permutation(len(self.imgs))
            with open(os.path.join(project_dir, 'data_processing', splits_dir, 'shuffle_' + mode + '.txt'), 'wb') as fp:
                pickle.dump(shuffle, fp)
        else:
            # use already existing shuffle
            with open(os.path.join(project_dir, 'data_processing', splits_dir, 'shuffle_' + mode + '.txt'), 'rb') as fp:
                shuffle = pickle.load(fp)
                # if lengths do not match, we need to create a new permutation
                if len(shuffle) != len(self.imgs):
                    shuffle = np.random.permutation(len(self.imgs))
                    with open(os.path.join(project_dir, 'data_processing', splits_dir, 'shuffle_' + mode + '.txt'), 'wb') as fp:
                        pickle.dump(shuffle, fp)

        self.imgs = np.array(self.imgs)
        self.zones = np.array(self.zones)

        tmp = self.imgs[shuffle]
        self.imgs = tmp
        tmp = self.zones[shuffle]
        self.zones = tmp

        self.imgs = list(self.imgs)
        self.zones = list(self.zones)


        # assert both lists have the same length
        assert len(self.imgs) == len(self.zones), "You don't have the same number of images and masks"

        self.mode = mode
        self.augmentation = augmentation
        self.data_dir = data_dir

        self.bright = bright
        self.wrap = wrap
        self.noise = noise
        self.rotate = rotate
        self.hflip = hflip
        self.vflip = vflip

    def custom_to_tensor(self, image, zone):
        to_tensor = ToTensorZones()
        image, zone = to_tensor(image=image, zone=zone)
        return image, zone

    def transform(self, image, zone):
        do_augmentation = self.augmentation
        # uint8 image pixels are converted to float32 and scaled to [0, 1].
        image, zone = self.custom_to_tensor(image=image, zone=zone)

        if self.mode == 'train' and do_augmentation:
            if np.random.random() >= (1 - self.hflip):
                image = torchvision.transforms.functional.hflip(image)
                zone = torchvision.transforms.functional.hflip(zone)

            if np.random.random() >= (1 - self.vflip):
                image = torchvision.transforms.functional.vflip(image)
                zone = torchvision.transforms.functional.vflip(zone)

            if np.random.random() >= (1 - self.rotate):
                rot = Rotate()
                image, zone = rot(image=image, zone=zone.unsqueeze(0))
                zone = zone.squeeze(0)
            if np.random.random() >= (1 - self.bright):
                bright = Bright()
                image, zone = bright(image=image, zone=zone)
            if np.random.random() >= (1 - self.wrap):
                wrap_transform = MyWrap()
                image, zone = wrap_transform(image=image, zone=zone)
            if np.random.random() >= (1 - self.noise):
                noise = Noise()
                image, zone = noise(image=image, zone=zone)

        image = torchvision.transforms.functional.normalize(
            image,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return image, zone

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # return: x, y_edge, y, x_name, y_edge_name, y_names
        img_name = self.imgs[idx]
        zone_name = self.zones[idx]

        # Assert that image and label name match
        assert patch_id(img_name) == patch_id(zone_name), \
            f"Image and label do not match: {img_name}, {zone_name}"
        dataset = gdal.Open(os.path.join(self.images_path, img_name).__str__())
        zone_dataset = gdal.Open(os.path.join(self.zones_path, zone_name).__str__())

        image = dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
        zone = zone_dataset.ReadAsArray(0, 0, zone_dataset.RasterXSize, zone_dataset.RasterYSize)
        zone = zone.astype(np.uint8)

        x, y = self.transform(image, zone)
        return x,  y, img_name,  zone_name

# Lightning data module for Great Wall training, validation, and testing.
class GreatwallDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, augmentation, data_dir, project_dir, bright, wrap, noise, rotate, hflip, vflip):
        """
        :param batch_size: batch size
        :param training_mode: 'training', 'debugging' or 'batch_overfit'
        :param augmentation: Whether or not augmentation shall be performed
        PyTorch Lightning data module for the Great Wall segmentation dataset.
        """
        super().__init__()
        self.batch_size = batch_size
        self.greatwall_test = None
        self.greatwall_train = None
        self.greatwall_val = None
        self.augmentation = augmentation

        self.data_dir = data_dir
        self.project_dir = project_dir

        self.bright = bright
        self.wrap = wrap
        self.noise = noise
        self.rotate = rotate
        self.hflip = hflip
        self.vflip = vflip

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            self.greatwall_test = GreatwallDataset(mode='test',
                                               augmentation=self.augmentation,
                                               data_dir=self.data_dir,
                                               project_dir=self.project_dir,
                                               bright=self.bright,
                                               wrap=self.wrap,
                                               noise=self.noise,
                                               rotate=self.rotate,
                                               hflip=self.hflip,
                                               vflip=self.vflip)
        if stage == 'fit' or stage is None:
            self.greatwall_train = GreatwallDataset(mode='train',
                                                augmentation=self.augmentation,
                                                data_dir=self.data_dir,
                                                project_dir=self.project_dir,
                                                bright=self.bright,
                                                wrap=self.wrap,
                                                noise=self.noise,
                                                rotate=self.rotate,
                                                hflip=self.hflip,
                                                vflip=self.vflip)
            self.greatwall_val = GreatwallDataset(mode='val',
                                              augmentation=self.augmentation,
                                              data_dir=self.data_dir,
                                              project_dir=self.project_dir,
                                              bright=self.bright,
                                              wrap=self.wrap,
                                              noise=self.noise,
                                              rotate=self.rotate,
                                              hflip=self.hflip,
                                              vflip=self.vflip)


    def train_dataloader(self):
        return DataLoader(
            self.greatwall_train,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.greatwall_val,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.greatwall_test,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def prepare_data(self, *args, **kwargs):
        pass
