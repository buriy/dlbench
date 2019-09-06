import argparse
import multiprocessing
import os
import os.path as osp
import random
from glob import glob
from time import time

import cv2
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

SEED = 486

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

if torch.cuda.is_available:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
else:
    print("ERROR: CUDA is not available. Exit")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from albumentations import (
    IAAPerspective,
    CLAHE,
    RandomRotate90,
    Transpose,
    ShiftScaleRotate,
    Blur,
    OpticalDistortion,
    GridDistortion,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    RandomBrightnessContrast,
    IAAPiecewiseAffine,
    IAASharpen,
    IAAEmboss,
    Flip,
    OneOf,
    Compose,
    Resize,
)


def strong_aug(p=0.5):
    return Compose(
        [
            RandomRotate90(),
            Flip(),
            Transpose(),
            IAAPerspective(),
            OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.2),
            OneOf([MotionBlur(p=0.2), MedianBlur(blur_limit=3, p=0.1), Blur(blur_limit=3, p=0.1)], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([OpticalDistortion(p=0.3), GridDistortion(p=0.1), IAAPiecewiseAffine(p=0.3)], p=0.2),
            OneOf([CLAHE(clip_limit=2), IAASharpen(), IAAEmboss(), RandomBrightnessContrast()], p=0.3),
            HueSaturationValue(p=0.3),
            Resize(256, 256, p=1, always_apply=True),
        ],
        p=p,
    )


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.filenames = sorted(glob(osp.join(folder_path, "*.jpg")))
        self.aug = strong_aug(p=1)
        print(f"total images: {len(self.filenames)}")

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = cv2.imread(filename)
        if self.aug:
            image = self.aug(image=image)["image"]
        return image

    def __len__(self):
        return len(self.filenames)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/sample/")
    parser.add_argument("--ncore", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--finish", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.finish > args.start, f"change start and finish"

    n_times = 10
    print("Running {} times with {} cores".format(n_times, args.ncore))

    times = []
    tmp = 0
    for i in range(10):
        dataset = FolderDataset(folder_path=args.path)
        vloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.ncore)
        progress_bar = tqdm(enumerate(vloader), total=len(vloader), desc=f"Predicting {i}")
        for n, data in progress_bar:
            images = data
            tmp += images.shape[0]
            if n == args.start:
                t0 = time()
            if n == args.finish:
                times.append(time() - t0)
                break
        progress_bar.close()
        print(f"Iteration #{i}: {times[-1]:0.1f}")

    print(f"mean {np.mean(times):0.1f}" + "\u00B1" + f"{np.std(times):0.1f}")
    print(f"speed {args.batch * (args.finish - args.start) / np.mean(times):0.1f} samples/sec")


if __name__ == "__main__":
    main()
