# Generative-SVG

## Team Members

<code>![Profile MatthewAndreTaylor](https://images.weserv.nl/?url=avatars.githubusercontent.com/u/100451342?v=4&h=30&w=30&fit=cover&mask=circle&maxage=7d) <a href="https://github.com/MatthewAndreTaylor">Matthew Taylor</a></code> , 
<code>![Profile tassonse](https://images.weserv.nl/?url=avatars.githubusercontent.com/u/116180211?v=4&h=30&w=30&fit=cover&mask=circle&maxage=7d) <a href="https://github.com/tassonse">Sebastian Tasson</a></code>

## Distribution of Responsibilities

- Matthew Taylor: Data acquisition, svg preprocessing, model architecture and training
- Sebastian Tasson: Data cleaning, indexing, validation, testing, modeling drawing quality.

Both of us will collaborate on the end product including documentation and a working proof of concept application.

## Machine-Vision Problem

We are addressing the problem of automatically generating SVG sketch images conditioned on object labels.
By extracting important features from labelled sketches, we aim to be able to generate new sketches.
This involves learning the underlying structure and semantics of the sketches. Then, training a model to produce outputs that are both visually accurate to the requested target label and semantically meaningful.

## Dataset Name and Location

- **Name** How Do Humans Sketch Objects? Dataset
- **Location:** [https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip](https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip)
- **License:** The sketch dataset is licensed under a Creative Commons Attribution 4.0 International License. http://creativecommons.org/licenses/by/4.0/


- **Name:** Quick, Draw! Dataset
- **Location:** [https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified)
- **License:** This data made available by Google, Inc. under the Creative Commons Attribution 4.0 International license.
https://creativecommons.org/licenses/by/4.0/


- **Name:** The Sketchy Database
- **Location:** [https://drive.google.com/file/d/1Qr8HhjRuGqgDONHigGszyHG_awCstivo/view](https://drive.google.com/file/d/1Qr8HhjRuGqgDONHigGszyHG_awCstivo/view)
- **License:** The MIT License (MIT) Copyright (c) 2016 janesjanes

## Accuracy Measurement

We plan to measure the accuracy of our system using:
- User study for qualitative assessment
- FiD?
- ...

## Paper References

```bibtex
@inproceedings{Sketchformer2020,
 title = {Sketchformer: Transformer-based Representation for Sketched Structure},
 author = {Leo Sampaio Ferraz Ribeiro and Tu Bui and John Collomosse and Moacir Ponti},
 booktitle = {Proc. CVPR},
 year = {2020},
} 
```

```bibtex
@article{DBLP:journals/corr/VaswaniSPUJGKP17,
  author       = {Ashish Vaswani and
                  Noam Shazeer and
                  Niki Parmar and
                  Jakob Uszkoreit and
                  Llion Jones and
                  Aidan N. Gomez and
                  Lukasz Kaiser and
                  Illia Polosukhin},
  title        = {Attention Is All You Need},
  journal      = {CoRR},
  volume       = {abs/1706.03762},
  year         = {2017},
  url          = {http://arxiv.org/abs/1706.03762},
  eprinttype    = {arXiv},
  eprint       = {1706.03762},
  timestamp    = {Sat, 23 Jan 2021 01:20:40 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/VaswaniSPUJGKP17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


```bibtex
@article{DBLP:journals/corr/abs-2001-02600,
  author       = {Peng Xu},
  title        = {Deep Learning for Free-Hand Sketch: {A} Survey},
  journal      = {CoRR},
  volume       = {abs/2001.02600},
  year         = {2020},
  url          = {http://arxiv.org/abs/2001.02600},
  eprinttype    = {arXiv},
  eprint       = {2001.02600},
  timestamp    = {Mon, 13 Jan 2020 12:40:17 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2001-02600.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@article{eitz2012hdhso,
author={Eitz, Mathias and Hays, James and Alexa, Marc},
title={How Do Humans Sketch Objects?},
journal={ACM Trans. Graph. (Proc. SIGGRAPH)},
year={2012},
volume={31},
number={4},
pages = {44:1--44:10}
}
```

```bibtex
@article{
 sketchy2016,
 author = {Patsorn Sangkloy and Nathan Burnell and Cusuh Ham and James Hays},
 title = {The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies},
 journal = {ACM Transactions on Graphics (proceedings of SIGGRAPH)},
 year = {2016},
}
```

```bibtex
@article{DBLP:journals/corr/abs-2007-02190,
  author       = {Ayan Das and
                  Yongxin Yang and
                  Timothy M. Hospedales and
                  Tao Xiang and
                  Yi{-}Zhe Song},
  title        = {B{\'{e}}zierSketch: {A} generative model for scalable vector
                  sketches},
  journal      = {CoRR},
  volume       = {abs/2007.02190},
  year         = {2020},
  url          = {https://arxiv.org/abs/2007.02190},
  eprinttype    = {arXiv},
  eprint       = {2007.02190},
  timestamp    = {Wed, 10 Jan 2024 18:05:26 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2007-02190.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


## Motivation for Problem and Dataset Selection

By focusing on SVG generation, we aim to overcome the limitations of raster-based approaches, such as poor scalability and unwanted background artifacts. We hope to provide a solution that is well-suited for applications in web design, game development, and digital art. Generating images with the traditional pixel image representations is very computationally expensive. A generative image model will output an $n \times n$ image that uses multiple bytes per pixel. This motivates our usage of a scalable vector image representation. We believe this representation will scale well with recent machine learning architectures. We also see improvements we could make to how previous strategies represent the sketch data. Such as quantizing coordinates ahead of time (See diagram below).

<img src="https://github.com/user-attachments/assets/a0ff91dc-6e19-4c23-bbf2-9fa8a2a9bfd4" alt="Diagram illustration" width="400"/>


This problem presents some interesting technical challenges, which include:

- Representing a sketch in a way that it can be understood by machine learning models.
- Ensuring the quality of the training data.
- Handling the ambiguity in human sketches, which can differ greatly in style and completeness.
- Creating models that generalize across different object categories and drawing techniques.
- Dealing with class imbalances and sparsity in the chosen datasets.
- Evaluating the semantic accuracy of generated SVG sketches, not just their visual similarity.

The reason that we chose the datasets above is because they are diverse, relevant, and accessible.
We think they will work well for generative SVG modeling. Each dataset contains a large number of hand-drawn sketches in vector formats, providing a rich source of training data. They cover a wide range of object categories and drawing styles, which we think will improve model robustness and generalization. Additionally, these datasets are well documented, used in other academic research, and available under permissive licenses. This makes them useful for both experimentation and reproducible results.


## Dataset Access

- Matthew Taylor has created a module `dataset.py` for downloading, organizing, preprocessing and using the datasets with `pytorch`. An example of how to use the datasets is in the notebook [dataset_visualization.ipynb](https://github.com/MatthewAndreTaylor/Generative-SVG/blob/main/dataset_visualization.ipynb)

- Sebastian Tasson validated that the dataset was accessible in the example notebook.

## Prerequisites

Python >= 3.9

```bash
pip install -r requirements.txt

```




