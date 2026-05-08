# Vision Language Model Fine-Tuning with QLoRA for Document-to-Markdown Generation

This repository contains a notebook-based implementation of a Vision Language Model pipeline that fine-tunes Qwen2-VL-2B-Instruct with QLoRA to convert document images into structured Markdown.

The project is designed around the assignment requirements for multimodal learning, parameter-efficient fine-tuning, document understanding, validation testing, and deployment through a lightweight Gradio interface.

## Project Overview

The model receives a document image and generates Markdown that preserves structural elements such as headings, lists, tables, equations, and code blocks. The workflow includes dataset inspection, ChatML-style formatting, 4-bit quantized training, inference on validation and unseen samples, visualization of results, and a small web app for interactive prediction.

## Repository Contents

- main.ipynb: complete notebook for data preparation, training, evaluation, and demo
- README.md: project documentation and usage guide

## Problem Statement

The goal is to adapt a pretrained vision-language model so that it can read document pages and produce Markdown output suitable for downstream documentation or text extraction tasks. This is a practical document understanding problem that combines image comprehension with structured text generation.

## Model and Method

- Base model: Qwen2-VL-2B-Instruct
- Fine-tuning method: QLoRA
- Quantization: 4-bit NF4
- Adapter approach: LoRA on q_proj and v_proj
- Frameworks: PyTorch, Hugging Face Transformers, PEFT, bitsandbytes

QLoRA keeps the pretrained model frozen and trains only small adapter weights, which makes the setup practical for Kaggle GPUs while still allowing the model to learn the document-to-Markdown mapping.

## Dataset

The notebook uses the Nougat training dataset example from Kaggle. Each sample is treated as a paired example:

- Input: document image
- Target: Markdown transcription

During preprocessing, the notebook filters out:

- missing markdown files
- unreadable images
- empty targets

The final subset is then split into training and validation data using an 80/20 ratio.

## Notebook Workflow

The notebook follows this end-to-end pipeline:

1. Install the required libraries
2. Import dependencies and configure seeds, GPU settings, and hyperparameters
3. Load and inspect the Nougat dataset
4. Visualize image and markdown pairs
5. Convert samples into a ChatML-style format
6. Build training and validation splits
7. Load Qwen2-VL-2B-Instruct in 4-bit mode
8. Add LoRA adapters and prepare the model for k-bit training
9. Train the model with gradient accumulation
10. Plot training and validation losses
11. Generate Markdown on validation samples
12. Evaluate predictions with ROUGE
13. Test on 3 training images and 3 unseen images
14. Compare zero-shot and fine-tuned outputs and other all 3-bonus tasks
15. Launch a Gradio app for image upload and Markdown generation

## Training Configuration

The notebook is tuned for Kaggle T4 x2 style constraints. The main settings are:

| Setting | Value |
|---|---|
| Batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | 2e-4 |
| Epochs | 1 in the current notebook, adjustable to 2 or 3 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Image size | 448 |
| Max sequence length | 768 |
| Max new tokens | 384 |

These values are selected to reduce memory pressure while still producing meaningful adapter training on the Nougat dataset.

## Evaluation and Outputs

The notebook produces several useful artifacts:

- dataset_exploration.png for dataset inspection
- loss_curves.png for training and validation loss tracking
- train_result_1.png, train_result_2.png, train_result_3.png for training sample comparisons
- unseen_result_1.png, unseen_result_2.png, unseen_result_3.png for unseen sample comparisons
- zeroshot_vs_finetuned.png for the bonus comparison
- ROUGE scores for validation evaluation
- Gradio demo output for interactive inference

## Example Use Cases Covered

The notebook explicitly demonstrates:

- dataset exploration
- markdown target inspection
- visual comparison of image and text pairs
- markdown generation from validation images
- markdown generation for 3 training examples
- markdown generation for 3 unseen examples
- zero-shot versus fine-tuned comparison
- prompt-style comparison experiments

## How to Run

1. Open the notebook in Kaggle or another GPU-enabled environment.
2. Ensure the Nougat dataset is available at the path defined in the notebook.
3. Run the notebook from the top in order.
4. Let the training cell complete and save the LoRA adapter.
5. Review the generated figures and validation outputs.
6. Launch the Gradio interface to test document image uploads.

## Deployment

The notebook includes a Gradio interface that accepts a document image and returns generated Markdown. This makes it easy to present the model as a small demo without building a full web application stack.

## Deliverables Covered

This repository supports the assignment deliverables:

- complete PyTorch implementation
- fine-tuning with QLoRA
- training and validation metrics
- generated Markdown outputs
- comparison between ground truth and predictions
- screenshots and visualizations
- app deployment through Gradio

## Reproducibility Notes

- The notebook uses a fixed random seed for reproducibility.
- Dataset subsetting is bounded by max_samples to keep runtime manageable.
- The code uses 4-bit quantization and gradient accumulation to fit Kaggle GPU constraints.
- The notebook saves outputs into the configured working directory.