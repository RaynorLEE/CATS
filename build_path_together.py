import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import datetime

def build_path(params):
    dataset, setting, train_size, max_path_hops = params
    model_params = [
        '--dataset', dataset,
        '--setting', setting,
        '--train_size', train_size,
        '--max_path_hops', max_path_hops
    ]
    command = ["python", "build_close_path.py"] + model_params
    subprocess.run(command)

def main():
    datasets = ["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"]
    settings = ["transductive"]
    train_sizes = ["1000", "2000"]
    max_path_hops = "3"
    # datasets = ["FB15k-237-subset"]
    # settings = ["inductive"]
    # train_sizes = ["full"]
    
    parameter_sets = []
    for setting in settings:
        if setting == "inductive":
            for dataset in datasets:
                parameter_sets.append((dataset, setting, "full", max_path_hops))
                    
        if setting == "transductive":
            for dataset in datasets:
                for train_size in train_sizes:
                    parameter_sets.append((dataset, setting, train_size, max_path_hops))
    
    with ThreadPoolExecutor(max_workers=len(parameter_sets)) as executor:
        futures = [executor.submit(build_path, params) for params in parameter_sets]
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
