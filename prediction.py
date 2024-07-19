import argparse
from tqdm import tqdm

from llm_api import run_llm_chat
from data_manager import DataManager
from prompt_templates import HYPER_PATTERN_BASED_SELECTOR_PROMPT

def main():
    parser = argparse.ArgumentParser(description="Triple Prediction System")
    parser.add_argument("--dataset", type=str, default="FB15k-237-subset", help="Name of the dataset")
    parser.add_argument("--inductive", type=str, choices=["True", "False"], default="True", help="Whether to use the inductive setting")
    parser.add_argument("--direction", type=str, choices=["head", "tail"], default="head", help="Direction for ranking")
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")
    
    args = parser.parse_args()

    data_manager = DataManager(args.dataset, args.inductive, args.direction, args.train_size)
    test_batches = data_manager.get_test_batches()[2:3]
    
    result_record = []
    for idx, batch in enumerate(tqdm(test_batches, desc="Processing batches")):
        response_record = []
        for test_triple in tqdm(batch, desc=f"Processing batch {idx+1}"):
            test_relation = test_triple[1]
            fewshot_triples = data_manager.fewshot_triple_finder(test_relation, 5)
            test_triple_input = data_manager.triple_to_sentence(test_triple)
            fewshot_triple_input = "\n".join(data_manager.triple_to_sentence(triple) for triple in fewshot_triples)
            prompt = HYPER_PATTERN_BASED_SELECTOR_PROMPT.format(fewshot_triple_input=fewshot_triple_input, test_triple_input=test_triple_input)
            print(prompt)
            response = run_llm_chat([{"role": "user", "content": prompt}])
            print(response)
            if "true" in response:
                response_record.append(1)
            else:
                response_record.append(0)
        result_record.append(response_record)
    
    num_batches = len(result_record)
    count_first_ones = sum(1 for record in result_record if record[0] == 1)
    proportion_first_ones = count_first_ones / num_batches
    ones_per_batch = [sum(record) for record in result_record]
    total_ones = sum(ones_per_batch)
    total_elements = sum(len(record) for record in result_record)
    proportion_ones = total_ones / total_elements

    print(f"Proportion of batches where the first element is 1: {proportion_first_ones:.2f}")
    print(f"Number of ones in each batch: {ones_per_batch}")
    print(f"Proportion of ones across all batches: {proportion_ones:.2f}")

    for result in result_record:
        print(result)

if __name__ == "__main__":
    main()
