import os
import json
import urllib.request
import urllib.parse
import zipfile
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict

from svgpathtools import svgstr2paths
from prepare_data import quickdraw_to_svg, remove_rect


class BaseSketchDataset(Dataset):
    """Common base for sketch datasets with caching and label management."""

    def __init__(self, label_names, out_dir, download: bool = False):
        self.label_names = label_names
        self.out_dir = out_dir
        self.cache_path = os.path.join(out_dir, "cache.pt")
        self.modified_cache = False
        os.makedirs(self.out_dir, exist_ok=True)
        if download:
            self.download_data()

        self.data_cache = self._load_cache()
        self.data, self.labels, self.label_map = self._load_data()
        if self.modified_cache:
            torch.save(self.data_cache, self.cache_path)

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            return torch.load(self.cache_path, weights_only=False)
        return defaultdict(list)

    # --- Abstract methods to implement in subclasses ---
    def download_data(self):
        """Download and extract dataset if needed."""
        raise NotImplementedError

    def get_label_files(self, label):
        """Return list of files or data entries for the label."""
        raise NotImplementedError

    def parse_file(self, label, file):
        """Read a single sample and return svg string."""
        raise NotImplementedError

    # ----------------------------

    def _load_data(self):
        data, labels = [], []
        label_map = {}

        for i, label in tqdm(
            enumerate(self.label_names), desc=f"Loading {self.__class__.__name__}"
        ):
            label_map[i] = label

            if label in self.data_cache and self.data_cache[label]:
                cached = self.data_cache[label]
                data.extend(cached)
                labels.extend([i] * len(cached))
                continue

            self.modified_cache = True
            samples = []
            for file in self.get_label_files(label):
                svg_data = self.parse_file(label, file)
                if isinstance(svg_data, list):
                    samples.extend(svg_data)
                elif svg_data is not None:
                    samples.append(svg_data)

            self.data_cache[label] = samples
            data.extend(samples)
            labels.extend([i] * len(samples))

        return data, labels, label_map

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class QuickDrawDataset(BaseSketchDataset):
    # https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified
    bucket_url = "https://storage.googleapis.com/storage/v1/b/quickdraw_dataset/o?prefix=full/simplified/"

    def __init__(self, label_names, download: bool = False):
        super().__init__(label_names, out_dir="data/quickdraw", download=download)

    def _get_buckets(self, label_names):
        with urllib.request.urlopen(self.bucket_url) as resp:
            data = json.load(resp)

        for item in data.get("items", []):
            name = item["name"]
            label = os.path.basename(name).replace(".ndjson", "")

            if label in label_names and name.endswith(".ndjson"):
                enc = urllib.parse.quote(name, safe="/")
                yield label, f"https://storage.googleapis.com/{item['bucket']}/{enc}"

    def download_data(self):
        buckets = list(self._get_buckets(self.label_names))
        for name, url in tqdm(
            buckets, desc=f"Downloading {self.__class__.__name__} files"
        ):
            output_path = f"{self.out_dir}/{name}.ndjson"
            if not os.path.exists(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                urllib.request.urlretrieve(url, output_path)

    def get_label_files(self, label):
        path = os.path.join(self.out_dir, f"{label}.ndjson")
        if os.path.exists(path):
            return [path]

        raise ValueError("Dataset missing or label has no samples.")

    def parse_file(self, label, file):
        with open(file) as f:
            return [
                quickdraw_to_svg(json.loads(line)["drawing"])
                for line in f
                if json.loads(line).get("recognized")
            ]


class TUBerlinDataset(BaseSketchDataset):
    # https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch
    primary_bucket_url = "https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip"
    backup_bucket_url = "https://drive.usercontent.google.com/download?id=1qQIasYwt-7MC8eqa0XRGYcBO_DBnY6um&export=download&authuser=0&confirm=t&uuid=333a7b8f-2227-4ea5-ba13-a1f893a27284&at=AKSUxGO0oPpXoLYi70ctYjKeN4GB:1759851532662"

    def __init__(self, label_names, download: bool = False):
        super().__init__(label_names, out_dir="data/tub", download=download)

    def download_data(self):
        if not os.path.exists("sketches_svg.zip"):
            try:
                print("Downloading TUBerlin files from primary source...")
                urllib.request.urlretrieve(self.primary_bucket_url, "sketches_svg.zip")
            except:
                print("Primary download failed, trying backup source...")
                urllib.request.urlretrieve(self.backup_bucket_url, "sketches_svg.zip")

        if not os.path.exists(f"{self.out_dir}/svg"):
            with zipfile.ZipFile("sketches_svg.zip", "r") as zip_ref:
                zip_ref.extractall(self.out_dir)

    def get_label_files(self, label):
        path = os.path.join(self.out_dir, "svg", label)
        files = [os.path.join(path, f) for f in os.listdir(path)]
        if not files:
            raise ValueError("Dataset missing or label has no samples.")
        return files

    def parse_file(self, label, file):
        with open(file, "r") as f:
            return f.read()


class SketchyDataset(BaseSketchDataset):
    # Note: sketchy is a bit larger and requires 7z and extra pre-processing

    # https://sketchy.eye.gatech.edu
    # https://drive.google.com/file/d/1Qr8HhjRuGqgDONHigGszyHG_awCstivo/view
    bucket_url = "https://drive.usercontent.google.com/download?authuser=0&export=download&id=1Qr8HhjRuGqgDONHigGszyHG_awCstivo&confirm=t&uuid=8cbdb41b-3dd4-41d3-8882-19056058bf2b&at=AN8xHorrKtrtyloclKhSah7qRz9L%3A1758498381130"

    def __init__(self, label_names, download: bool = False):
        self.invalid_labels = {}
        out_dir = "data/sketchy"

        for label_name in label_names:
            with open(f"{out_dir}/sketches/{label_name}/invalid.txt", "r") as f:
                self.invalid_labels[label_name] = f.read()

        super().__init__(label_names, out_dir=out_dir, download=download)

    def download_data(self):
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

    def get_label_files(self, label):
        path = os.path.join(self.out_dir, "sketches", label)
        if not os.path.exists(path):
            raise ValueError("Dataset missing or label has no samples.")
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".svg")]

    def parse_file(self, label, file):
        stem = os.path.splitext(os.path.basename(file))[0]
        if stem in self.invalid_labels.get(label, ""):
            return None

        with open(file, "r") as f:
            svg_data = remove_rect(f.read())
        try:
            svgstr2paths(svg_data)
        except Exception:
            return None
        return svg_data


class SketchDataset(Dataset):
    def __init__(
        self,
        dataset: BaseSketchDataset,
        tokenizer,
        max_len=200,
    ):
        self.seqs = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.vocab["PAD"]
        cache_modified = False

        self.parent_dataset = dataset

        # Try to load some labeled data from cache
        cache_file = os.path.join(
            dataset.out_dir,
            f"cache_maxlen-{max_len}_tokenizer-{tokenizer.__class__.__name__.lower()}-vocabsize-{len(tokenizer.vocab)}.pt",
        )

        if os.path.exists(cache_file):
            tokenized_data_cache = torch.load(cache_file, weights_only=False)
        else:
            tokenized_data_cache = defaultdict(list)

        for label_name in tqdm(dataset.label_names, desc="Tokenizing dataset"):
            if label_name in tokenized_data_cache and tokenized_data_cache[label_name]:
                tokenized = tokenized_data_cache[label_name]
                self.seqs.extend(tokenized)
                continue

            cache_modified = True
            tokenized = []
            cached = dataset.data_cache[label_name]
            for svg in cached:
                tokens = tokenizer.encode(svg)[:max_len]
                tokens = tokens + [self.pad_id] * (max_len - len(tokens))
                tokenized.append(tokens)

                # Another option is to skip samples that are too long

            tokenized_data_cache[label_name] = tokenized
            self.seqs.extend(tokenized)

        if cache_modified:
            torch.save(tokenized_data_cache, cache_file)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        input_ids = torch.tensor(seq[:-1])
        target_ids = torch.tensor(seq[1:])
        return input_ids, target_ids, self.parent_dataset.labels[idx]

    def __len__(self):
        return len(self.seqs)
