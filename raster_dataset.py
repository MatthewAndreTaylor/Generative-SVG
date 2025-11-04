import os
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
from dataset import SketchDataset
import pyvips
from io import BytesIO
from PIL import Image


def svg_rasterize(svg_string: str) -> Image.Image:
    svg_bytes = svg_string.encode("utf-8")
    image = pyvips.Image.svgload_buffer(svg_bytes)
    image = image.flatten(background=0xFFFFFF)
    png_bytes = image.write_to_buffer(".png")
    img = Image.open(BytesIO(png_bytes)).convert("L")
    return img


class SketchImageDataset(Dataset):
    def __init__(self, sketch_dataset: SketchDataset):
        self.sketch_dataset = sketch_dataset

        self.sketch_images = []
        cache_file = os.path.join(
            sketch_dataset.parent_dataset.out_dir,
            f"sketch_image_cache.pt",
        )

        if os.path.exists(cache_file):
            image_data_cache = torch.load(cache_file)
        else:
            image_data_cache = defaultdict(list)

        for label_name in tqdm(
            sketch_dataset.parent_dataset.label_names, desc="Rasterizing sketches"
        ):
            if label_name in image_data_cache and image_data_cache[label_name]:
                images = image_data_cache[label_name]
                self.sketch_images.extend(images)
                continue

            images = []
            cached = sketch_dataset.parent_dataset.data_cache[label_name]
            for svg in cached:
                img = svg_rasterize(svg)
                images.append(img)

            self.sketch_images.extend(images)
            image_data_cache[label_name] = images

    def __getitem__(self, idx):
        return self.sketch_images[idx], self.sketch_dataset[idx]

    def __len__(self):
        return len(self.sketch_images)
