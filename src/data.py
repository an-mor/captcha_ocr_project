from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def collect_image_paths(data_dir):
    data_dir = Path(data_dir)
    paths = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")
    return paths


def build_vocabulary(image_paths):
    labels = [Path(path).stem for path in image_paths]
    captcha_length = len(labels[0])
    if not all(len(label) == captcha_length for label in labels):
        raise ValueError("All labels must have the same length")

    alphabet = sorted(set("".join(labels)))
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return labels, alphabet, char_to_idx, idx_to_char, captcha_length


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


class RandomLinesAndLocalNoise:
    def __init__(self, line_probability=0.35, noise_probability=0.4):
        self.line_probability = line_probability
        self.noise_probability = noise_probability

    def __call__(self, image):
        image = image.copy()

        if random.random() < self.line_probability:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for _ in range(random.randint(1, 2)):
                start = (random.randint(0, width - 1), random.randint(0, height - 1))
                end = (random.randint(0, width - 1), random.randint(0, height - 1))
                shade = random.randint(0, 90)
                draw.line((start, end), fill=(shade, shade, shade), width=1)

        if random.random() < self.noise_probability:
            array = np.asarray(image, dtype=np.int16).copy()
            height, width, _ = array.shape
            for _ in range(random.randint(1, 3)):
                patch_width = random.randint(6, 18)
                patch_height = random.randint(3, 8)
                left = random.randint(0, width - patch_width)
                top = random.randint(0, height - patch_height)
                noise = np.random.normal(
                    loc=0.0,
                    scale=random.uniform(5.0, 12.0),
                    size=(patch_height, patch_width, 1),
                )
                array[
                    top : top + patch_height,
                    left : left + patch_width,
                ] += noise.astype(np.int16)
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

        return image


def create_train_transform():
    return transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=4,
                translate=(0.03, 0.08),
                shear=(-2, 2),
                interpolation=InterpolationMode.BILINEAR,
                fill=(255, 255, 255),
            ),
            transforms.RandomPerspective(
                distortion_scale=0.08,
                p=0.3,
                interpolation=InterpolationMode.BILINEAR,
                fill=(255, 255, 255),
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            RandomLinesAndLocalNoise(),
            image_to_tensor,
        ]
    )


class CaptchaDataset(Dataset):
    def __init__(self, image_paths, char_to_idx, transform=image_to_tensor):
        self.image_paths = [Path(path) for path in image_paths]
        self.char_to_idx = dict(char_to_idx)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = image_path.stem

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        target = torch.tensor(
            [self.char_to_idx[char] for char in label],
            dtype=torch.long,
        )
        return image, target, label, str(image_path)


def split_dataset(
    dataset,
    train_dataset=None,
    train_fraction=0.7,
    validation_fraction=0.1,
    seed=42,
):
    train_dataset = dataset if train_dataset is None else train_dataset
    if len(train_dataset) != len(dataset):
        raise ValueError("Train and evaluation datasets must have the same length")

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test split")

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    train_end = int(train_fraction * len(indices))
    validation_end = train_end + int(validation_fraction * len(indices))
    return (
        Subset(train_dataset, indices[:train_end]),
        Subset(dataset, indices[train_end:validation_end]),
        Subset(dataset, indices[validation_end:]),
    )


def create_dataloaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    batch_size,
    seed,
    device,
):
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, validation_loader, test_loader
