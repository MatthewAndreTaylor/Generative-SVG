from torch.utils.data import Dataset
from dataset import QuickDrawDataset
from prepare_data import stroke_to_bezier_single, convert_and_quantize_svg
from tqdm import tqdm
import pickle

class QuickDrawFineTuningDataset(Dataset):
    def __init__(self, labels, tokenizer, max_length=1024, cache_file="sketch_tokenized_dataset_test_gpt2.pkl"):
        self.samples = QuickDrawDataset(
            labels=labels,
            download=True,
        )
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        self.data = []
        
        try:
            with open(cache_file, "rb") as f:
                self.data = pickle.load(f)
            print(f"Loaded tokenized data from {cache_file}")
        except FileNotFoundError:
            for item in tqdm(self.samples, desc="Processing samples"):
                item = stroke_to_bezier_single(item, num_samples=20, maxError=10.0)
                item = convert_and_quantize_svg(item, 64)
                
                # parse the string inside d="..." from <path d="..."/>
                item = item.split('d="')[1].split('"')[0]
                self.data.append(item)

            with open(cache_file, "wb") as f:
                pickle.dump(self.data, f)
            print(f"Saved tokenized data to {cache_file}")
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_num = self.samples.labels[idx]
        label = self.samples.label_map[label_num]
        
        text = f"{label} sketch {item}"
        
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Trainer expects these fields
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # standard causal LM training
        }
        
        
        

# Transform dataset that processes on-the-fly without caching
class QuickDrawFineTuningDatasetTransform(Dataset):
    def __init__(self, labels, tokenizer, max_length=1024, cache_file="sketch_tokenized_dataset_test_gpt2.pkl"):
        self.samples = QuickDrawDataset(
            labels=labels,
            download=True,
        )
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        item = stroke_to_bezier_single(item, num_samples=20, maxError=10.0)
        item = convert_and_quantize_svg(item, 64)
        
        # parse the string inside d="..." from <path d="..."/>
        item = item.split('d="')[1].split('"')[0]
        
        label_num = self.samples.labels[idx]
        label = self.samples.label_map[label_num]
        
        text = f"{label} sketch {item}"
        
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Trainer expects these fields
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # standard causal LM training
        }