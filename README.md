# Generative-SVG

A Python toolkit for collecting, processing, and preparing SVG files for machine learning and generative applications.
This project provides utilities to pull SVG assets in the public domin, minify them, and quantize their paths.

## Features

- 🔍 **Automated SVG Collection**: Scrape SVG files from SVGRepo collections
- 🗜️ **SVG Minification**: Reduce file sizes using web-based minification
- 📐 **Path Quantization**: Convert SVG paths to discrete integers
- 🧹 **SVG Cleaning**: Remove unnecessary attributes and optimize SVG structure


### Quantization Process

The quantization process:
1. Parses the SVG viewBox to determine coordinate bounds
2. Maps all path coordinates to a discrete grid (default: 96x96)
3. Converts different path segment types (lines, curves, arcs) to quantized equivalents
4. Cleans and optimizes the resulting SVG


### Project Goals

- [x] Provide a novel way to build a collection of assets which are easy for an LLM to consume. See [text completion dataset](https://docs.pytorch.org/torchtune/0.3/generated/torchtune.datasets.text_completion_dataset.html) spec.
- [ ] [Fine tune](https://github.com/pytorch/torchtune/tree/main?tab=readme-ov-file#supervised-finetuning-sft) existing LLM's using the prepared svg dataset.


### Attribution

All assets used exist in the public domain. See https://www.svgrepo.com/page/licensing/ for more information.
