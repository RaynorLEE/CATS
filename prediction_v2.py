import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_manager import DataManager

def cal_Y_prob(model, tokenizer, generation_config, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    generated_output = model.generate(
        input_ids=inputs.input_ids,
        return_dict_in_generate=True,
        output_scores=True,
        **generation_config
    )
    
    scores = generated_output.scores[0]
    probs = scores.softmax(dim=-1)
    
    Y_id = tokenizer.encode("Y", add_special_tokens=False)[0]
    N_id = tokenizer.encode("N", add_special_tokens=False)[0]
    
    Y_prob = probs[0, Y_id].item()
    
    return Y_prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset", help="Name of the dataset")
    parser.add_argument("--setting", type=str, choices=["inductive", "transductive"], default="inductive", help="Inductive or Transductive setting")
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")
    args = parser.parse_args()

    data_manager = DataManager(dataset=args.dataset, setting=args.setting, train_size=args.train_size)
    test_batches = data_manager.get_test_batches()

    model = AutoModelForCausalLM.from_pretrained(data_manager.model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(data_manager.model_path)
    generation_config = dict(
        temperature=0,
        top_k=0,
        top_p=0,
        do_sample=False,
        max_new_tokens=1,
    )

    hits_result_ontology = []
    hits_result_path = []

    for idx, batch in enumerate(tqdm(test_batches, desc="Processing batches")):
        ontology_prob_in_batch = []

        for batch_idx, test_triple in enumerate(batch):
            ontology_prompt = data_manager.build_ontology_prompt(test_triple)
            Y_prob = cal_Y_prob(model, tokenizer, generation_config, ontology_prompt)
            ontology_prob_in_batch.append((Y_prob, batch_idx))

        sorted_ontology_indices = sorted(range(len(ontology_prob_in_batch)), key=lambda i: ontology_prob_in_batch[i][0], reverse=True)
        hits_position_ontology = sorted_ontology_indices.index(0) + 1 if 0 in sorted_ontology_indices else 0
        hits_result_ontology.append(hits_position_ontology)
        
        # Select top 10 based on Y_prob
        top_10_indices = sorted_ontology_indices[:10]
        filtered_batch = [batch[i] for _, i in [ontology_prob_in_batch[j] for j in top_10_indices]]

        path_prob_in_batch = []
        for test_triple in filtered_batch:
            path_prompt = data_manager.build_path_prompt(test_triple)
            Y_prob = cal_Y_prob(model, tokenizer, generation_config, path_prompt)
            path_prob_in_batch.append(Y_prob)

        sorted_path_indices = sorted(range(len(path_prob_in_batch)), key=lambda i: path_prob_in_batch[i], reverse=True)
        hits_position_path = sorted_path_indices.index(0) + 1 if 0 in sorted_path_indices else 0
        hits_result_path.append(hits_position_path)
    
    print("Proportion of top 10 in ontology:", sum(1 for hits in hits_result_ontology if hits <= 10) / len(hits_result_ontology))
    print("Proportion of top 5 in ontology:", sum(1 for hits in hits_result_ontology if hits <= 5) / len(hits_result_ontology))
    
    print("Ontology Hits results:", hits_result_ontology)
    print("Ontology Hit@1:", sum(1 for hits in hits_result_ontology if hits == 1) / len(hits_result_ontology))
    print("Ontology MRR:", sum(1 / hits for hits in hits_result_ontology if hits != 0) / len(hits_result_ontology))

    print("Path Hits results:", hits_result_path)
    print("Path Hit@1:", sum(1 for hits in hits_result_path if hits == 1) / len(hits_result_path))
    print("Path MRR:", sum(1 / hits for hits in hits_result_path if hits != 0) / len(hits_result_path))

if __name__ == "__main__":
    main()
