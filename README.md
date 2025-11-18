# Conditional Sketch Generation & Completion

[![Github](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/MatthewAndreTaylor/Generative-SVG)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MatthewAndreTaylor/Generative-SVG/HEAD)
[![Website](https://img.shields.io/badge/Flask-matttaylordev.pythonanywhere.com-informational?style=flat-square&color=E23237&logo=flask&logoColor=white)](https://matttaylordev.pythonanywhere.com)



This is the codebase for [Conditional Sketch Generation & Completion](paper_href)

This README walks through how to train and sample from the sketch generation and completion model.


## Installation

You must install Python >= 3.11

Clone this repository and navigate to it in your terminal. Then install the project requirements.

```bash
pip install -r requirements.txt
```

Pytorch may have system specific requirements you can find a wheel that works for your system at the following site:

https://pytorch.org/get-started/locally/


## Dataset

The module `dataset.py` for downloading, organizing, preprocessing and using the datasets. An example of how to use the datasets is in the notebook [dataset_visualization.ipynb](https://github.com/MatthewAndreTaylor/Generative-SVG/blob/main/dataset_visualization.ipynb)

We primarily use a stratified version of the [Quick, Draw! Dataset](https://quickdraw.withgoogle.com/data) and train with a 85% train, 10% validation, 5% test split.

## Training

To train your model, you should first decide some hyperparameters. Hyperparameters are split up into four groups: 

- 1. **Dataset**
- 2. **Tokenizer encoding**
- 3. **Model architecture**
- 4. **Training configuration**

Here is an example of setting up training using either `Jupyter Notebook` or `Python + Toml Configuration`

#### Jupyter Notebook

Create a new notebook in the project directory or get started from an existing one in the [experiments](https://github.com/MatthewAndreTaylor/Generative-SVG/tree/main/experiments) directory. Our existing training experiment notebooks are named `sketch_experiments_*.ipynb`

Below is an example training configuration from [example.ipynb]()


```py
from dataset import QuickDrawDataset
from sketch_tokenizers import DeltaPenPositionTokenizer
from models import SketchTransformerConditional
from runner import SketchTrainer

label_names = ["bird", "crab", "guitar"]
dataset = QuickDrawDataset(label_names=label_names)
tokenizer = DeltaPenPositionTokenizer(bins=32)

model = SketchTransformerConditional(
    vocab_size=len(tokenizer.vocab),
    d_model=512,
    nhead=8,
    num_layers=8,
    max_len=200,
    num_classes=len(label_names),
)

training_config = {
    "batch_size": 128,
    "num_epochs": 15,
    "learning_rate": 1e-4,
    "log_dir": "logs/sketch_transformer_experiment_2",
    "splits": [0.85, 0.1, 0.05], 
    # "checkpoint_path": "logs/path/to/existing/checkpoint/model_checkpoint.pt"
}

trainer = SketchTrainer(model, dataset, tokenizer, training_config)
```

Then in the next cell run

```py
trainer.train_mixed(training_config["num_epochs"])
```

#### Python + Toml Configuration

<details>
<summary>If you prefer to train the model in the terminal</summary>

<br>

First create or modify an existing training toml configuration file. We provide some examples in the [configs](https://github.com/MatthewAndreTaylor/Generative-SVG/tree/main/configs) directory. Below is an example training configuration:


```toml
[dataset]
class = "QuickDrawDataset"
label_names = ["bird", "crab", "guitar"]

[tokenizer]
class = "DeltaPenPositionTokenizer"
bins = 32

[model]
class = "SketchTransformerConditional"
d_model = 512
nhead = 8
num_layers = 8
max_len = 200

[training]
batch_size = 128
num_epochs = 15
learning_rate = 1e-4
log_dir = "logs/sketch_transformer_example"
use_padding_mask = false
splits = [0.85, 0.1, 0.05] # 85% train, 10% validation, 5% test

# Resume from a specific model checkpoint
# checkpoint_path = "logs/path/to/existing/checkpoint/model_checkpoint.pt"
```


Then run `python main.py --config path/to/config.toml`

</details>

<br>

Training logs and checkpoints are saved to the `log_dir`. You can follow the notebook [experiments/sample_outputs.ipynb](https://github.com/MatthewAndreTaylor/Generative-SVG/blob/main/experiments/sample_outputs.ipynb) to load and sample from a saved model. To visualize the metrics collected while training run `tensorboard --logdir log_dir` and navigate to http://localhost:6006/


## Demo

We have created a demo webapp to test out the system and put on display some examples.

Take a look at https://matttaylordev.pythonanywhere.com