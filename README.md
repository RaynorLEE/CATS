# CATS: Context-aware Inductive Knowledge Graph Completion with Latent Type Constraints and Subgraph Reasoning

This repository provides the official implementation of the paper *"Context-aware Inductive Knowledge Graph Completion with Latent Type Constraints and Subgraph Reasoning"* (To appear in AAAI2025).

![CATS](CATS.png)

## Experiment Environment Setup
Create a python environment and install the required packages. We suggest you use Python 3.10 with PyTorch 2.2.2.
For detailed Python package versions, you may refer to the suggested settings listed in `requirements.txt` from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/requirements.txt). (VLLM is not required) 

```bash
pip install -r requirements.txt
```
   
Additionally, install `sentence_transformers`:

```bash
pip install sentence_transformers
```

## Dataset

1. Download the full dataset and LLM instructions from the following link:

- [Dataset &amp; Instructions](https://drive.google.com/drive/folders/17C3BsllCWy_TK3B5WwCjxPQo2heuLJPz?usp=drive_link)

2. Copy the two subfolders "datasets" and "instructions" into the project directory.
Alternatively, you can construct the LLM instruction prompts by executing `python build_instruction.py`.

## LLM Setup

You may download LLM checkpoints from the following links:
Our experimental results can be reproduced with the Qwen2-7B-Instruct LLM. 

- [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

Please update the default value of `LLM_PATH` in script `data_manager.py` with your local model path.

## Intruction-tuning

Please refer to the official document of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/) to conduct LLM supervised fine-tuning with the provided prompts. You may need to specify the prompt path and other training settings in a configuration file. Detailed hyper-parameters are provided in our paper. 

## Inference

The following command evaluates the model performance. You may alter the parameters below to test the model in different (transductive, inductive, and few-shot) scenarios.

```bash
python3 prediction.py --dataset FB15k-237-subset --setting inductive --training_size full --model_name {model_path_after_sft} --prompt_type CATS --subgraph_type together --path_type degree
```

## Citation

If you find this code useful, please consider citing the following paper.
```
@misc{li2024contextawareinductiveknowledgegraph,
      title={Context-aware Inductive Knowledge Graph Completion with Latent Type Constraints and Subgraph Reasoning}, 
      author={Muzhi Li and Cehao Yang and Chengjin Xu and Zixing Song and Xuhui Jiang and Jian Guo and Ho-fung Leung and Irwin King},
      year={2024},
      eprint={2410.16803},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.16803}, 
}
```
