# CATS: Context-aware Inductive Knowledge Graph Completion with Latent Type Constraints and Subgraph Reasoning

This repository provides the official implementation of the paper *"Context-aware Inductive Knowledge Graph Completion with Latent Type Constraints and Subgraph Reasoning"* (To appear in AAAI2025).

![CATS](CATS.png)

## Environment Setup

Please follow these steps to execute our code:

1. **Install dependencies:**
   Create a python environment and install the required packages. You can refer to the `requirements.txt` from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/requirements.txt).

   ```bash
   pip install -r requirements.txt
   ```
   
   Additionally, install `sentence_transformers`:

   ```bash
   pip install sentence_transformers
   ```

## Dataset

Download the full dataset and pre-built instructions from the following link:

- [Dataset &amp; Instructions](https://drive.google.com/drive/folders/17C3BsllCWy_TK3B5WwCjxPQo2heuLJPz?usp=drive_link)

Alternatively, you can manually build the instruction set by executing `python build_instruction.py`.

## LLM Setup

Download LLMs from the following links:

- [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

After downloading, update the `LLM_PATH` in the `data_manager.py` script to point to your local model path.

## Inference

To run inference, execute the following command:

```bash
python prediction.py
```

This will start the inference process using the provided model and dataset in zero-shot setting.

## Intruction-tuning

Please refer to the official document from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/.) with the provided instruction set. Hyper-parameters are provided in our paper.
