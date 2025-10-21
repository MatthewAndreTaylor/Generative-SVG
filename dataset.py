import os
import json
import urllib.request
import urllib.parse
import zipfile
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Callable, Optional

from svgpathtools import svgstr2paths
from prepare_data import quickdraw_to_svg, remove_rect


# Transforming all the items in the dataset
class BaseSketchDataset(Dataset):
    def __init__(
        self,
        data,
        base_transform: Optional[Callable] = None,
        cache_file: Optional[str] = None,
    ):
        if cache_file and os.path.exists(cache_file):
            self.data = torch.load(cache_file)
        else:
            self.data = data
            if base_transform:
                self.data = [
                    base_transform(d)
                    for d in tqdm(self.data, desc="Applying base transform")
                ]
            if cache_file:
                torch.save(self.data, cache_file)

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class TUBerlinDataset(BaseSketchDataset):
    # https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch
    primary_bucket_url = "https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip"
    backup_bucket_url = "https://drive.usercontent.google.com/download?id=1qQIasYwt-7MC8eqa0XRGYcBO_DBnY6um&export=download&authuser=0&confirm=t&uuid=333a7b8f-2227-4ea5-ba13-a1f893a27284&at=AKSUxGO0oPpXoLYi70ctYjKeN4GB:1759851532662"

    out_dir = "data/tub"

    def __init__(self, labels, download: bool = False, **kwargs):
        self.labels = labels

        if download:
            print("Downloading TUBerlin files")
            os.makedirs(self.out_dir, exist_ok=True)
            if not os.path.exists("sketches_svg.zip"):
                try:
                    urllib.request.urlretrieve(
                        self.primary_bucket_url, "sketches_svg.zip"
                    )
                except Exception as e:
                    print(f"Error downloading from primary bucket: {e}")
                    try:
                        urllib.request.urlretrieve(
                            self.backup_bucket_url, "sketches_svg.zip"
                        )
                    except Exception as e:
                        print(f"Error downloading from backup bucket: {e}")

            if not os.path.exists(f"{self.out_dir}/svg"):
                with zipfile.ZipFile("sketches_svg.zip", "r") as zip_ref:
                    zip_ref.extractall(self.out_dir)

        data = []
        labels_path = f"{self.out_dir}/svg"
        downloaded_labels = set(os.listdir(labels_path))

        self.label_map = {}
        self.labels = []

        for i, label in tqdm(enumerate(labels), desc="Loading TUBerlin files"):
            if label not in downloaded_labels:
                raise ValueError("Dataset missing or label has no samples.")

            self.label_map[i] = label

            data_path = f"{self.out_dir}/svg/{label}"
            for file in os.listdir(data_path):
                svg_path = f"{data_path}/{file}"

                with open(svg_path, "r") as f:
                    svg_data = f.read()
                    data.append(svg_data)
                    self.labels.append(i)

        super().__init__(data, **kwargs)


class QuickDrawDataset(BaseSketchDataset):
    # https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified
    bucket_url = "https://storage.googleapis.com/storage/v1/b/quickdraw_dataset/o?prefix=full/simplified/"
    out_dir = "data/quickdraw"

    def __init__(self, labels, download: bool = False, **kwargs):
        if download:
            files = self.get_buckets(labels)
            for name, url in tqdm(files, desc="Downloading QuickDraw files"):
                output_path = f"{self.out_dir}/{name}.ndjson"
                if not os.path.exists(output_path):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    urllib.request.urlretrieve(url, output_path)

        data = []
        self.label_map = {}
        self.labels = []

        for i, label in tqdm(enumerate(labels), desc="Loading QuickDraw files"):
            if not os.path.exists(f"{self.out_dir}/{label}.ndjson"):
                raise ValueError("Dataset missing or label has no samples.")

            self.label_map[i] = label

            with open(f"{self.out_dir}/{label}.ndjson") as f:
                for line in f:
                    d = json.loads(line)
                    if d["recognized"]:
                        data.append(quickdraw_to_svg(d["drawing"]))
                        self.labels.append(i)

        super().__init__(data, **kwargs)

    def get_buckets(self, labels):
        label_files = []
        with urllib.request.urlopen(self.bucket_url) as resp:
            data = json.load(resp)

        for item in data.get("items", []):
            name = item["name"]
            label = os.path.basename(name).replace(".ndjson", "")

            if label in labels and name.endswith(".ndjson"):
                enc = urllib.parse.quote(name, safe="/")
                label_files.append(
                    (label, f"https://storage.googleapis.com/{item['bucket']}/{enc}")
                )
        return label_files


class SketchyDataset(BaseSketchDataset):
    # Note: sketchy is a bit larger and requires 7z and extra pre-processing

    # https://sketchy.eye.gatech.edu
    # https://drive.google.com/file/d/1Qr8HhjRuGqgDONHigGszyHG_awCstivo/view

    bucket_url = "https://drive.usercontent.google.com/download?authuser=0&export=download&id=1Qr8HhjRuGqgDONHigGszyHG_awCstivo&confirm=t&uuid=8cbdb41b-3dd4-41d3-8882-19056058bf2b&at=AN8xHorrKtrtyloclKhSah7qRz9L%3A1758498381130"
    out_dir = "data/sketchy"

    def __init__(self, labels, download: bool = False, **kwargs):
        self.labels = labels

        if download:
            print("Downloading Sketchy files")
            os.makedirs(self.out_dir, exist_ok=True)
            if not os.path.exists("sketchy.7z"):
                urllib.request.urlretrieve(self.bucket_url, "sketchy.7z")

            if not os.path.exists(f"{self.out_dir}/sketches"):
                try:
                    import py7zr

                    with py7zr.SevenZipFile("sketchy.7z", mode="r") as z:
                        z.extractall(path=self.out_dir)

                except ImportError:
                    raise ImportError(
                        "Please install py7zr to extract Sketchy dataset: pip install py7zr"
                    )

        data = []
        labels_path = f"{self.out_dir}/sketches"
        downloaded_labels = set(os.listdir(labels_path))

        self.label_map = {}
        self.labels = []

        for i, label in tqdm(enumerate(labels), desc="Loading Sketchy files"):
            if label not in downloaded_labels:
                raise ValueError("Dataset missing or label has no samples.")

            self.label_map[i] = label

            invalid = ""
            with open(f"{self.out_dir}/sketches/{label}/invalid.txt", "r") as f:
                invalid = f.read()

            data_path = f"{self.out_dir}/sketches/{label}"
            for file in os.listdir(data_path):
                if file.endswith(".svg"):
                    svg_path = f"{data_path}/{file}"

                    stem = os.path.splitext(os.path.basename(file))[0]
                    if stem in invalid:
                        continue

                    with open(svg_path, "r") as f:
                        svg_data = f.read()
                        svg_data = remove_rect(svg_data)
                        data.append(svg_data)
                        self.labels.append(i)
                        # Validate SVG
                        try:
                            svgstr2paths(svg_data)
                        except Exception as e:
                            data.pop()
                            self.labels.pop()

        super().__init__(data, **kwargs)


def split_data(
    dataset, splits=(0.8, 0.1, 0.1), seed=42, cache_path=None, overwrite=False
):
    if cache_path and os.path.exists(cache_path) and not overwrite:
        with open(cache_path) as f:
            return json.load(f)

    rng = random.Random(seed)
    torch.manual_seed(seed)

    # keep = []
    # for i in range(len(dataset)):
    #    keep.append(i)

    a, b, c = splits
    s = a + b + c
    a, b, c = a / s, b / s, c / s

    n = len(dataset)
    all_indices = list(range(n))
    rng.shuffle(all_indices)

    n_train = int(round(n * a))
    n_val = int(round(n * b))
    n_test = n - n_train - n_val

    train = all_indices[:n_train]
    val = all_indices[n_train : n_train + n_val]
    test = all_indices[n_train + n_val : n_train + n_val + n_test]

    out = {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "meta": {"seed": seed, "splits": [a, b, c]},
    }

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(out, f, indent=2)

    return out
