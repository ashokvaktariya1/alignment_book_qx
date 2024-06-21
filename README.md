# Updated Alignment-Handbook

## How to navigate this project 🧭

This project is simple by design and mostly consists of:

* [`scripts`](./scripts/) to train and evaluate models. Four steps are included: continued pretraining, supervised-finetuning (SFT) for chat, preference alignment with DPO, and supervised-finetuning with preference alignment with ORPO. Each script supports distributed training of the full model weights with DeepSpeed ZeRO-3, or LoRA/QLoRA for parameter-efficient fine-tuning.
* [`recipes`](./recipes/) to reproduce models like Zephyr 7B. Each recipe takes the form of a YAML file which contains all the parameters associated with a single training run. A `gpt2-nl` recipe is also given to illustrate how this handbook can be used for language or domain adaptation, e.g. by continuing to pretrain on a different language, and then SFT and DPO tuning the result. 

We are also working on a series of guides to explain how methods like direct preference optimization (DPO) work, along with lessons learned from gathering human preferences in practice. To get started, we recommend the following:

1. Follow the [installation instructions](#installation-instructions) to set up your environment etc.
2. Replicate Zephyr-7b-β by following the [recipe instructions](./recipes/zephyr-7b-beta/README.md).

If you would like to train chat models on your own datasets, we recommend following the dataset formatting instructions [here](./scripts/README.md#fine-tuning-on-your-datasets).


## Contents

The initial release of the handbook will focus on the following techniques:

* **Continued pretraining:** adapt language models to a new language or domain, or simply improve it by continue pretraning (causal language modeling) on a new dataset.
* **Supervised fine-tuning:** teach language models to follow instructions and tips on how to collect and curate your own training dataset.
* **Reward modeling:** teach language models to distinguish model responses according to human or AI preferences.
* **Rejection sampling:** a simple, but powerful technique to boost the performance of your SFT model.
* **Direct preference optimisation (DPO):** a powerful and promising alternative to PPO.
* **Odds Ratio Preference Optimisation (ORPO)**: a technique to fine-tune language models with human preferences, combining SFT and DPO in a single stage.

## Installation instructions

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n handbook python=3.10 && conda activate handbook
```

Next, install PyTorch `v2.1.2` - the precise version is important for reproducibility! Since this is hardware-dependent, we
direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

You can then install the remaining package dependencies as follows:

```shell
git clone https://github.com/ashokvaktariya1/alignment_book_qx.git
cd ./alignment_book_qx/
python -m pip install .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn --no-build-isolation
```

> **Note**
> If your machine has less than 96GB of RAM and many CPU cores, reduce the `MAX_JOBS` arguments, e.g. `MAX_JOBS=4 pip install flash-attn --no-build-isolation`

Next, log into your Hugging Face account as follows:

```shell
huggingface-cli login
```

Finally, install Git LFS so that you can push models to the Hugging Face Hub:

```shell
sudo apt-get install git-lfs
```

You can now check out the `scripts` and `recipes` directories for instructions on how to train some models 🪁!

## Project structure

```
├── README.md                   <- The top-level README for developers using this project
├── chapters                    <- Educational content to render on hf.co/learn
├── recipes                     <- Recipe configs, accelerate configs, slurm scripts
├── scripts                     <- Scripts to train and evaluate chat models
├── setup.cfg                   <- Installation config (mostly used for configuring code quality & tests)
├── setup.py                    <- Makes project pip installable (pip install -e .) so `alignment` can be imported
├── src                         <- Source code for use in this project
└── tests                       <- Unit tests
```

