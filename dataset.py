import os
import json
import urllib.request
import urllib.parse
import zipfile
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Callable, Optional

from prepare_data import quickdraw_to_svg


# transforming all the items in the dataset
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
    bucket_url = "https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip"
    out_dir = "data/tub"

    def __init__(self, labels, download: bool = False, **kwargs):
        self.labels = labels

        if download:
            print("Downloading TUBerlin files")
            os.makedirs(self.out_dir, exist_ok=True)
            if not os.path.exists("sketches_svg.zip"):
                urllib.request.urlretrieve(self.bucket_url, "sketches_svg.zip")

            if not os.path.exists(f"{self.out_dir}/svg"):
                with zipfile.ZipFile("sketches_svg.zip", "r") as zip_ref:
                    zip_ref.extractall(self.out_dir)

        data = []
        labels_path = f"{self.out_dir}/svg"
        downloaded_labels = set(os.listdir(labels_path))

        if not "cache_file" in kwargs:
            for label in tqdm(labels, desc="Loading TUBerlin files"):
                if label not in downloaded_labels:
                    raise ValueError("Dataset missing or label has no samples.")

                data_path = f"{self.out_dir}/svg/{label}"
                for file in os.listdir(data_path):
                    svg_path = f"{data_path}/{file}"

                    with open(svg_path, "r") as f:
                        svg_data = f.read()
                        data.append(svg_data)

        super().__init__(data, **kwargs)


class QuickDrawDataset(BaseSketchDataset):
    # https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified
    bucket_url = "https://storage.googleapis.com/storage/v1/b/quickdraw_dataset/o?prefix=full/simplified/"
    out_dir = "data/quickdraw"

    def __init__(self, labels, download: bool = False, **kwargs):
        self.labels = labels

        if download:
            files = self.get_buckets(labels)
            for name, url in tqdm(files, desc="Downloading QuickDraw files"):
                output_path = f"{self.out_dir}/{name}.ndjson"
                if not os.path.exists(output_path):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    urllib.request.urlretrieve(url, output_path)

        data = []

        if not "cache_file" in kwargs:
            for label in tqdm(labels, desc="Loading QuickDraw files"):
                if not os.path.exists(f"{self.out_dir}/{label}.ndjson"):
                    raise ValueError("Dataset missing or label has no samples.")

                with open(f"{self.out_dir}/{label}.ndjson") as f:
                    for line in f:
                        d = json.loads(line)
                        if d["recognized"]:
                            data.append(quickdraw_to_svg(d["drawing"]))

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
