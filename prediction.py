import os
import argparse
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_manager import DataManager
from datetime import datetime

def cal_Y_prob(model:AutoModelForCausalLM, tokenizer:AutoTokenizer, generation_config, prompt_list):
    messages_batch = [
        [{"role": "user", "content": prompt}]
        for prompt in prompt_list
    ]
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    generated_output = model.generate(
        input_ids=inputs.input_ids,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
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
    parser.add_argument("--model_name", type=str, choices=["Qwen2-7B-Instruct", "Meta-Llama-3-8B-Instruct", "Qwen2-1.5B-Instruct"], default="Qwen2-7B-Instruct")
    parser.add_argument("--llm_type", type=str, choices=["sft", "base"], default="sft")
    parser.add_argument("--prompt_type", type=str, choices=["CATS", "none", "CATS-all"], default="CATS")
    parser.add_argument("--subgraph_type", type=str, choices=["neighbor-only", "path-only", "combine"], default="combine")
    parser.add_argument("--path_type", type=str, choices=["degree", "no-degree"], default="degree")

    args = parser.parse_args()

    log_dir = f"logs_{args.model_name}_{args.llm_type}_{args.prompt_type}_{args.subgraph_type}_{args.path_type}"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_file = os.path.join(log_dir, f"log_{args.dataset}_{args.setting}_{args.train_size}_{timestamp}.txt")

    data_manager = DataManager(dataset=args.dataset, setting=args.setting, train_size=args.train_size, model_name=args.model_name, llm_type=args.llm_type)
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

    llm_batch_size = 1
    sample_counter = 0

    def log_results(label, results):
        log.write(f"{label} Hits results: {results}\n")
        hit_at_1 = round(sum(1 for hits in results if hits == 1) / len(results), 3)
        mrr = round(sum(1 / hits for hits in results if hits != 0) / len(results), 3)
        log.write(f"{label} Hit@1: {hit_at_1}\n")
        log.write(f"{label} MRR: {mrr}\n")

    with open(log_file, 'w') as log:
        if args.prompt_type == "none":
            hits_result_none = []
            log.write(f"Using model: {data_manager.model_path}\n")
            
            for idx, batch in enumerate(tqdm(test_batches, desc="Processing test batches")):
                none_prompts = [data_manager.build_none_prompt(test_triple) for test_triple in batch]
                none_probs = []
                for i in range(0, len(none_prompts), llm_batch_size):
                    batch_prompts = none_prompts[i:i + llm_batch_size]
                    none_probs.extend(cal_Y_prob(model, tokenizer, generation_config, batch_prompts))
                for i, (prompt, prob) in enumerate(zip(none_prompts, none_probs)):
                    log.write(f"Sample {sample_counter} none Prompt: {prompt}\n")
                    log.write(f"Sample {sample_counter} none 'Y' token Probability: {prob}\n")
                    log.write("*"*50 + "\n")
                    sample_counter += 1
                none_prob_in_batch = list(zip(none_probs, range(len(none_probs))))
                sorted_none_indices = sorted(range(len(none_prob_in_batch)), key=lambda i: none_prob_in_batch[i][0], reverse=True)
                log.write(f"Sorted none indices: {sorted_none_indices}\n")
                hits_position_base = sorted_none_indices.index(0) + 1 if 0 in sorted_none_indices else 0
                hits_result_none.append(hits_position_base)
                log.write("*"*50 + "\n")
                log.flush()
                
                if (idx + 1) % 100 == 0:
                    log.write(f"\nMetrics after processing {idx + 1} batches:\n")
                    log_results("None", hits_result_none)
                    log.write("\n" + "="*50 + "\n")
                    log.flush()
                    
            log.write("Final Results:\n")
            log_results("None", hits_result_none)
            log.flush()
        
        elif args.prompt_type == "CATS-all":
            hits_result_all = []
            log.write(f"Using model: {data_manager.model_path}\n")
            
            for idx, batch in enumerate(tqdm(test_batches, desc="Processing test batches")):
                all_prompts = [data_manager.build_all_prompt(test_triple) for test_triple in batch]
                all_probs = []
                for i in range(0, len(all_prompts), llm_batch_size):
                    batch_prompts = all_prompts[i:i + llm_batch_size]
                    all_probs.extend(cal_Y_prob(model, tokenizer, generation_config, batch_prompts))
                for i, (prompt, prob) in enumerate(zip(all_prompts, all_probs)):
                    log.write(f"Sample {sample_counter} all Prompt: {prompt}\n")
                    log.write(f"Sample {sample_counter} all 'Y' token Probability: {prob}\n")
                    log.write("*"*50 + "\n")
                    sample_counter += 1
                all_prob_in_batch = list(zip(all_probs, range(len(all_probs))))
                sorted_all_indices = sorted(range(len(all_prob_in_batch)), key=lambda i: all_prob_in_batch[i][0], reverse=True)
                log.write(f"Sorted all indices: {sorted_all_indices}\n")
                hits_position_all = sorted_all_indices.index(0) + 1 if 0 in sorted_all_indices else 0
                hits_result_all.append(hits_position_all)
                log.write("*"*50 + "\n")
                log.flush()
                
                if (idx + 1) % 100 == 0:
                    log.write(f"\nMetrics after processing {idx + 1} batches:\n")
                    log_results("All", hits_result_all)
                    log.write("\n" + "="*50 + "\n")
                    log.flush()
                    
            log.write("Final Results:\n")
            log_results("All", hits_result_all)
            log.flush()
            
        elif args.prompt_type == "CATS":
            hits_result_type = []
            hits_result_subgraph = []
            hits_result_average_ensemble = []
            TAR_infer_times = []
            SR_infer_times = []
            # hits_result_weighted_ensemble = []
            # hits_result_type_filtered_subgraph = []
            log.write(f"Using model: {data_manager.model_path}\n")
            
            for idx, batch in enumerate(tqdm(test_batches, desc="Processing test batches")):
                type_prompts = [data_manager.build_type_prompt(test_triple) for test_triple in batch]
                if args.subgraph_type == "combine":
                    subgraph_prompts = [data_manager.build_subgraph_prompt(test_triple) for test_triple in batch]
                elif args.subgraph_type == "neighbor-only":
                    subgraph_prompts = [data_manager.build_neighbor_prompt(test_triple) for test_triple in batch]
                elif args.subgraph_type == "path-only":
                    if args.path_type == "degree":
                        subgraph_prompts = [data_manager.build_close_path_prompt(test_triple) for test_triple in batch]
                    elif args.path_type == "no-degree":
                        subgraph_prompts = [data_manager.build_close_path_no_degree_prompt(test_triple) for test_triple in batch]
                type_probs = []
                batch_infer_times = 0
                for i in range(0, len(type_prompts), llm_batch_size):
                    batch_prompts = type_prompts[i:i + llm_batch_size]
                    start_time = time.time()
                    type_probs.extend(cal_Y_prob(model, tokenizer, generation_config, batch_prompts))
                    end_time = time.time()
                    time_interval = end_time - start_time
                    batch_infer_times += time_interval
                    # log.write(f"Time for type reasoning inference: {time_interval}\n")
                TAR_infer_times.append(batch_infer_times)
                for i, (prompt, prob) in enumerate(zip(type_prompts, type_probs)):
                    log.write(f"Sample {sample_counter} type Prompt: {prompt}")
                    log.write(f"Sample {sample_counter} type 'Y' token Probability: {prob}\n")
                    log.write("*"*50 + "\n")
                    sample_counter += 1
                    
                type_prob_in_batch = list(zip(type_probs, range(len(type_probs))))
                sorted_type_indices = sorted(range(len(type_prob_in_batch)), key=lambda i: type_prob_in_batch[i][0], reverse=True)
                log.write(f"Sorted type indices: {sorted_type_indices}\n")
                hits_position_type = sorted_type_indices.index(0) + 1 if 0 in sorted_type_indices else 0
                hits_result_type.append(hits_position_type)

                top_10_type_indices = sorted_type_indices[:10]
                type_filtered_set = set(top_10_type_indices)
                
                subgraph_probs = []
                batch_infer_times = 0
                for i in range(0, len(subgraph_prompts), llm_batch_size):
                    batch_prompts = subgraph_prompts[i:i + llm_batch_size]
                    start_time = time.time()
                    subgraph_probs.extend(cal_Y_prob(model, tokenizer, generation_config, batch_prompts))
                    end_time = time.time()
                    time_interval = end_time - start_time
                    batch_infer_times += time_interval
                    # log.write(f"Time for subgraph reasoning inference: {time_interval}\n")
                SR_infer_times.append(batch_infer_times)

                for i, (prompt, prob) in enumerate(zip(subgraph_prompts, subgraph_probs)):
                    log.write(f"Sample {sample_counter} Subgraph Prompt: {prompt}\n")
                    log.write(f"Sample {sample_counter} Subgraph 'Y' token Probability: {prob}\n")
                    log.write("*"*50 + "\n")
                    sample_counter += 1

                subgraph_prob_in_batch = list(zip(subgraph_probs, range(len(subgraph_probs))))
                sorted_subgraph_indices = sorted(range(len(subgraph_prob_in_batch)), key=lambda i: subgraph_prob_in_batch[i][0], reverse=True)
                log.write(f"Sorted Subgraph indices: {sorted_subgraph_indices}\n")
                hits_position_subgraph = sorted_subgraph_indices.index(0) + 1 if 0 in sorted_subgraph_indices else 0
                hits_result_subgraph.append(hits_position_subgraph)

                # Ensemble type reasoning and subgraph reasoning
                combined_ranks = [sorted_type_indices.index(i) + sorted_subgraph_indices.index(i) for i in range(len(sorted_type_indices))]
                sorted_combined_indices = sorted(range(len(combined_ranks)), key=lambda i: combined_ranks[i])
                hits_position_average_ensemble = sorted_combined_indices.index(0) + 1 if 0 in sorted_combined_indices else 0
                hits_result_average_ensemble.append(hits_position_average_ensemble)
                
                # # Weighted Ensemble
                # weighted_scores = [(1 / (sorted_type_indices.index(i) + 1) + 1 / (sorted_subgraph_indices.index(i) + 1)) for i in range(len(sorted_type_indices))]
                # sorted_weighted_indices = sorted(range(len(weighted_scores)), key=lambda i: weighted_scores[i], reverse=True)
                # hits_position_weighted_ensemble = sorted_weighted_indices.index(0) + 1 if 0 in sorted_weighted_indices else 0
                # hits_result_weighted_ensemble.append(hits_position_weighted_ensemble)
                
                # # Filter subgraph results based on type_filtered_list
                # sorted_filtered_subgraph_indices = [index for index in sorted_subgraph_indices if index in type_filtered_set]
                # log.write(f"Sorted filtered subgraph indices: {sorted_filtered_subgraph_indices}\n")
                # hits_position_type_filtered_subgraph = sorted_filtered_subgraph_indices.index(0) + 1 if 0 in sorted_filtered_subgraph_indices else 0
                # hits_result_type_filtered_subgraph.append(hits_position_type_filtered_subgraph)
                
                log.write("*"*50 + "\n")
                log.flush()
                
                if (idx + 1) % 100 == 0:
                    log.write(f"\nMetrics after processing {idx + 1} batches:\n")
                    log_results("Type", hits_result_type)
                    log_results("Subgraph", hits_result_subgraph)
                    log_results("Average Ensemble", hits_result_average_ensemble)
                    # log_results("Weighted Ensemble", hits_result_weighted_ensemble)
                    # log_results("Type Filtered Subgraph", hits_result_type_filtered_subgraph)
                    log.write("\n" + "="*50 + "\n")
                    log.flush()

            log.write("Final Results:\n")
            log.write("Propotion of type reasoning top 5: {}\n".format(sum(1 for hits in hits_result_type if hits <= 5) / len(hits_result_type)))
            log.write("Propotion of type reasoning top 10: {}\n".format(sum(1 for hits in hits_result_type if hits <= 10) / len(hits_result_type)))
            
            log_results("Type", hits_result_type)
            log_results("Subgraph", hits_result_subgraph)
            log_results("Average Ensemble", hits_result_average_ensemble)
            # log_results("Weighted Ensemble", hits_result_weighted_ensemble)
            # log_results("Type Filtered Subgraph", hits_result_type_filtered_subgraph)
            
            # Time cost
            log.write("Average time for type reasoning inference: {}\n".format(sum(TAR_infer_times) / len(TAR_infer_times)))
            log.write("Average time for subgraph reasoning inference: {}\n".format(sum(SR_infer_times) / len(SR_infer_times)))
            log.flush()

if __name__ == "__main__":
    main()
