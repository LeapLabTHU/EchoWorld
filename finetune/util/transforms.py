import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import random
import logging

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

class StandardMask:
    def __init__(self):
        self.mask = np.zeros((224, 224, 3), dtype=np.uint8)
        
    def __call__(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.ellipse(self.mask, center=(112, 8), axes=(208, 208), angle=0, startAngle=45,
                            endAngle=135, color=(1, 1, 1), thickness=-1)
        img = img * self.mask
        img = Image.fromarray(np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        return img


def build_transform(args, train=True):
    if train:
        logging.info(f'Aug Type: {args.aug_type}')
        logging.info(f'Color Jitter: {args.color_jitter}')
        color_aug = [
            transforms.RandomApply([transforms.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter)], p=0.6),
        ]
        if 'blur' in args.aug_type:
            color_aug.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=(random.choice([5, 7, 9, 11, 13])), sigma=random.uniform(0.1, 2))], p=0.6))
        if 'crop' in args.aug_type:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    # transforms.RandomCrop(224),
                    transforms.CenterCrop(224),
                    *color_aug,
                    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(random.choice([5, 7, 9, 11, 13])), sigma=random.uniform(0.1, 2))], p=0.6),
                    transforms.Grayscale(num_output_channels=3),
                    # StandardMask(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                ])
        else:
            return transforms.Compose(
                [
                    transforms.Resize((args.input_size, args.input_size)),
                    *color_aug,
                    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(random.choice([5, 7, 9, 11, 13])), sigma=random.uniform(0.1, 2))], p=0.6),
                    transforms.Grayscale(num_output_channels=3),
                    # StandardMask(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                ])
    else:
        if 'crop' in args.aug_type:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Grayscale(num_output_channels=3),
                    # StandardMask(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                ])
        else:
            return transforms.Compose(
                [
                    transforms.Resize((args.input_size, args.input_size)),
                    transforms.Grayscale(num_output_channels=3),
                    # StandardMask(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                ])
