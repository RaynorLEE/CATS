import os
import argparse
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from data_manager import DataManager
from prompt_templates import ONTOLOGY_REASON_PROMPT, CLOSE_PATH_REASON_PROMPT, OPEN_PATH_REASON_PROMPT, EXPLAINING_PROMPT

from llm_api import run_llm_chat
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset", help="Name of the dataset")
    parser.add_argument("--setting", type=str, choices=["inductive", "transductive"], default="inductive", help="Inductive or Transductive setting")
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")
    args = parser.parse_args()

    data_manager = DataManager(dataset=args.dataset, setting=args.setting, train_size=args.train_size)
    test_batches = data_manager.get_test_batches()
    
    model_path = "/data/FinAi_Mapping_Knowledge/finllm/LLMs/Qwen2-7B-Instruct"
    # model_path = "/data/FinAi_Mapping_Knowledge/finllm/LLMs/Qwen1.5-14B-Chat"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    generation_config = dict(
        temperature=0,
        top_k=0,
        top_p=0,
        do_sample=False,
        max_new_tokens=1,
    )

    device = "cuda"
    Y_id = tokenizer.encode("Y", add_special_tokens=False)[0]
    N_id = tokenizer.encode("N", add_special_tokens=False)[0]
    
    hits_result_ontology = []
    hits_result_path = []
    hits_result_all = []

    for idx, batch in enumerate(tqdm(test_batches, desc="Processing batches")):
        ontology_prob_in_batch = []
        path_prob_in_batch = []

        for batch_idx, test_triple in enumerate(batch):
            fewshot_triples = data_manager.diverse_fewshot_triple_finder(test_triple)
            test_triple_sentence = data_manager.triple_to_sentence(test_triple)
            fewshot_triples_sentence = '\n'.join(data_manager.triple_to_sentence(triple) for triple in fewshot_triples)
            ontology_prompt = ONTOLOGY_REASON_PROMPT.format(fewshot_triples=fewshot_triples_sentence, test_triple=test_triple_sentence)
            # print(ontology_prompt)
            ontology_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ontology_prompt}
            ]
            ontology_text = tokenizer.apply_chat_template(
                ontology_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(ontology_text, return_tensors="pt").to(device)

            generated_output = model.generate(
                input_ids=inputs.input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_config
            )

            scores = generated_output.scores[0]
            probs = scores.softmax(dim=-1)
            Y_prob = probs[0, Y_id].item()
            ontology_prob_in_batch.append(Y_prob)
        sorted_ontology_indices = sorted(range(len(ontology_prob_in_batch)), key=lambda i: ontology_prob_in_batch[i], reverse=True)
        hits_position_ontology = sorted_ontology_indices.index(0) + 1 if 0 in sorted_ontology_indices else 0
        hits_result_ontology.append(hits_position_ontology)
        continue
        for batch_idx, test_triple in enumerate(batch):
            test_close_paths = data_manager.close_path_finder(test_triple)

            if test_close_paths != []:
                reasoning_paths = "\n".join(
                    " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
                    for path in test_close_paths
                )
                path_prompt = CLOSE_PATH_REASON_PROMPT.format(reasoning_paths=reasoning_paths, test_triple=data_manager.triple_to_sentence(test_triple))
            else:
                known_triples = data_manager.open_path_finder(test_triple)
                path_prompt = OPEN_PATH_REASON_PROMPT.format(known_triples='\n'.join(known_triples), test_triple=data_manager.triple_to_sentence(test_triple))
            # print(path_prompt)
            path_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": path_prompt}
            ]
            path_text = tokenizer.apply_chat_template(
                path_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(path_text, return_tensors="pt").to(device)
            generated_output = model.generate(
                input_ids=inputs.input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_config
            )
            probs = generated_output.scores[0].softmax(dim=-1)
            Y_prob = probs[0, Y_id].item()
            path_prob_in_batch.append(Y_prob)

        sorted_path_indices = sorted(range(len(path_prob_in_batch)), key=lambda i: path_prob_in_batch[i], reverse=True)
        hits_position_path = sorted_path_indices.index(0) + 1 if 0 in sorted_path_indices else 0
        hits_result_path.append(hits_position_path)

        # Ensemble
        combined_ranks = [sorted_ontology_indices.index(i) + sorted_path_indices.index(i) for i in range(len(sorted_ontology_indices))]
        sorted_combined_indices = sorted(range(len(combined_ranks)), key=lambda i: combined_ranks[i])
        hits_position_all = sorted_combined_indices.index(0) + 1 if 0 in sorted_combined_indices else 0
        hits_result_all.append(hits_position_all)
    
    print("Propotion of top 10:", sum(1 for hits in hits_result_ontology if hits <= 10) / len(hits_result_ontology))
    print("Propotion of top 5:", sum(1 for hits in hits_result_ontology if hits <= 5) / len(hits_result_ontology))
    
    print("Ontology Hits results:", hits_result_ontology)
    print("Ontology Hit@1:", sum(1 for hits in hits_result_ontology if hits == 1) / len(hits_result_ontology))
    print("Ontology MRR:", sum(1 / hits for hits in hits_result_ontology if hits != 0) / len(hits_result_ontology))

    # print("Path Hits results:", hits_result_path)
    # print("Path Hit@1:", sum(1 for hits in hits_result_path if hits == 1) / len(hits_result_path))
    # print("Path MRR:", sum(1 / hits for hits in hits_result_path if hits != 0) / len(hits_result_path))

    # print("Ensemble Hits results:", hits_result_all)
    # print("Ensemble Hit@1:", sum(1 for hits in hits_result_all if hits == 1) / len(hits_result_all))
    # print("Ensemble MRR:", sum(1 / hits for hits in hits_result_all if hits != 0) / len(hits_result_all))

    

if __name__ == "__main__":
    main()
