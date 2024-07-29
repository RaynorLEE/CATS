import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_manager import DataManager

def cal_Y_prob(model, tokenizer, generation_config, prompt_list):
    messages_batch = [
        [{"role": "user", "content": prompt}]
        for prompt in prompt_list
    ]
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

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
    
    Y_probs = [probs[i, Y_id].item() for i in range(probs.shape[0])]
    
    return Y_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset", help="Name of the dataset")
    parser.add_argument("--setting", type=str, choices=["inductive", "transductive"], default="inductive", help="Inductive or Transductive setting")
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")
    args = parser.parse_args()

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{args.dataset}_{args.setting}_{args.train_size}.txt")

    data_manager = DataManager(dataset=args.dataset, setting=args.setting, train_size=args.train_size)
    test_batches = data_manager.get_test_batches()[:10]

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
    hits_result_all = []
    hits_result_ontology_filtered_path = []
    llm_batch_size = 8

    with open(log_file, 'w') as log:
        log.write(f"Using model: {data_manager.model_path}\n")
        for idx, batch in enumerate(tqdm(test_batches, desc="Processing test batches")):
            ontology_prompts = [data_manager.build_ontology_prompt(test_triple) for test_triple in batch]
            path_prompts = [data_manager.build_path_prompt(test_triple) for test_triple in batch]

            ontology_probs = []
            for i in range(0, len(ontology_prompts), llm_batch_size):
                batch_prompts = ontology_prompts[i:i + llm_batch_size]
                ontology_probs.extend(cal_Y_prob(model, tokenizer, generation_config, batch_prompts))
            
            for i, (prompt, prob) in enumerate(zip(ontology_prompts, ontology_probs)):
                log.write(f"Sample {idx * llm_batch_size + i} Ontology Prompt: {prompt}")
                log.write(f"Sample {idx * llm_batch_size + i} Ontology Probability: {prob}\n")
                log.write("*"*50 + "\n")
                
            ontology_prob_in_batch = list(zip(ontology_probs, range(len(ontology_probs))))
            sorted_ontology_indices = sorted(range(len(ontology_prob_in_batch)), key=lambda i: ontology_prob_in_batch[i][0], reverse=True)
            log.write(f"Sorted ontology indices: {sorted_ontology_indices}\n")
            hits_position_ontology = sorted_ontology_indices.index(0) + 1 if 0 in sorted_ontology_indices else 0
            hits_result_ontology.append(hits_position_ontology)

            top_10_indices = sorted_ontology_indices[:10]
            ontology_filtered_list = [ontology_prob_in_batch[i][1] for i in top_10_indices]

            path_probs = []
            for i in range(0, len(path_prompts), llm_batch_size):
                batch_prompts = path_prompts[i:i + llm_batch_size]
                path_probs.extend(cal_Y_prob(model, tokenizer, generation_config, batch_prompts))

            for i, (prompt, prob) in enumerate(zip(path_prompts, path_probs)):
                log.write(f"Sample {idx * llm_batch_size + i} Path Prompt: {prompt}")
                log.write(f"Sample {idx * llm_batch_size + i} Path Probability: {prob}\n")
                log.write("*"*50 + "\n")

            path_prob_in_batch = list(zip(path_probs, range(len(path_probs))))
            sorted_path_indices = sorted(range(len(path_prob_in_batch)), key=lambda i: path_prob_in_batch[i][0], reverse=True)
            log.write(f"Sorted path indices: {sorted_path_indices}\n")
            hits_position_path = sorted_path_indices.index(0) + 1 if 0 in sorted_path_indices else 0
            hits_result_path.append(hits_position_path)

            # Ensemble ontology reasoning and path reasoning
            combined_ranks = [sorted_ontology_indices.index(i) + sorted_path_indices.index(i) for i in range(len(sorted_ontology_indices))]
            sorted_combined_indices = sorted(range(len(combined_ranks)), key=lambda i: combined_ranks[i])
            hits_position_all = sorted_combined_indices.index(0) + 1 if 0 in sorted_combined_indices else 0
            hits_result_all.append(hits_position_all)
            
            # Filter path results based on ontology_filtered_list
            filtered_path_prob_in_batch = [path_prob_in_batch[i] for i in ontology_filtered_list]
            sorted_filtered_path_indices = sorted(range(len(filtered_path_prob_in_batch)), key=lambda i: filtered_path_prob_in_batch[i][0], reverse=True)
            hits_position_ontology_filtered_path = sorted_filtered_path_indices.index(0) + 1 if 0 in sorted_filtered_path_indices else 0
            hits_result_ontology_filtered_path.append(hits_position_ontology_filtered_path)
        
        def log_results(label, results):
            log.write(f"{label} Hits results: {results}\n")
            log.write(f"{label} Hit@1: {sum(1 for hits in results if hits == 1) / len(results)}\n")
            log.write(f"{label} MRR: {sum(1 / hits for hits in results if hits != 0) / len(results)}\n")

        log.write("Final Results:\n")
        log.write("Propotion of ontology reasoning top 5: {}\n".format(sum(1 for hits in hits_result_ontology if hits <= 5) / len(hits_result_ontology)))
        log.write("Propotion of ontology reasoning top 10: {}\n".format(sum(1 for hits in hits_result_ontology if hits <= 10) / len(hits_result_ontology)))
        
        log_results("Ontology", hits_result_ontology)
        log_results("Path", hits_result_path)
        log_results("Ensemble", hits_result_all)
        log_results("Ontology Filtered Path", hits_result_ontology_filtered_path)

if __name__ == "__main__":
    main()
