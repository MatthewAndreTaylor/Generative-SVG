# Generative-SVG

## Team Members

- <code>![Profile MatthewAndreTaylor](https://images.weserv.nl/?url=avatars.githubusercontent.com/u/100451342?v=4&h=30&w=30&fit=cover&mask=circle&maxage=7d) <a href="https://github.com/MatthewAndreTaylor">Matthew Taylor</a></code>

- <code>![Profile tassonse](https://images.weserv.nl/?url=avatars.githubusercontent.com/u/116180211?v=4&h=30&w=30&fit=cover&mask=circle&maxage=7d) <a href="https://github.com/tassonse">Sebastian Tasson</a></code>

## Distribution of Responsibilities

- Matthew Taylor: Data acquisition, svg preprocessing, model architecture and training
- Sebastian Tasson: Data cleaning, indexing, validation, testing, modeling drawing quality.

Both of us will collaborate on the end product including documentation and a working proof of concept application.

## Machine-Vision Problem

We are addressing the problem of automatic generation of SVG images from hand-drawn sketches. The goal is to extract important features in sketches and condition the model on labels, enabling the system to generate scalable vector representations that capture the essence of human drawing. This involves learning the underlying structure and semantics of sketches, so that the model can produce SVG outputs that are both visually accurate and semantically meaningful. Our approach leverages recent advances in machine learning to bridge the gap between human creativity and automated image synthesis, ultimately enabling more efficient and flexible workflows for designers and artists.

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


## Motivation for Problem and Dataset Selection

By focusing on SVG generation, we aim to overcome the limitations of raster-based approaches, such as poor scalability and loss of detail, and provide a solution that is well-suited for applications in web design, game development, and digital art.

Generating images with the traditional pixel image representations is very computationally expensive. Large input sizes n x n image multiple bytes per pixel. This motivates our usage of a scalable vector image representation. We believe this representation will scale well with recent machine learning architectures.

Traditional using models that generate raster images are less useful for web design, game development and other tasks. These raster images have many drawbacks...

Examples: 

Logos, branding, icons, ui design: require sharp interpretable images that work across different devices and and products. No additional model is required to scale up/down these images.

Game design: Vector images are well suited for animation.

Chatgpt and existing models are not fucused on drawing sketch images (example: [ChatGPT trying to generate a dog SVG](https://chatgpt.com/share/68cc2e09-39f4-8002-9bc6-bffad012a5e7) )


We use two of the datasets which were used for this task in the past, however we attempt to use a different tensor representation.

We see improvements we could make to how previous stratergies represent the sketch data. Such as quantizing point coordinates ahead of time. 


## Dataset Access

- Matthew Taylor has created a module `dataset.py` for downloading, organizing, preprocessing and using the datasets with `pytorch`. An example of how to use this dataset is in `dataset_visualization.ipynb`

- Sebastian Tasson validated that the dataset was accessible in the example notebook.

## Prerequisites

Python >= 3.9

```bash
pip install -r requirements.txt

```

