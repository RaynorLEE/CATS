import os
import json
import random
import numpy as np
from collections import defaultdict, deque
from sentence_transformers import SentenceTransformer
from prompt_templates import TYPE_REASON_PROMPT, SUBGRAPH_REASON_PROMPT, NEIGHBOR_REASON_PROMPT, CLOSE_PATH_REASON_PROMPT, BASE_REASON_PROMPT, ALL_REASON_PROMPT

LLM_PATH = ""

class DataManager:
    def __init__(self, dataset="FB15k-237-subset", setting="inductive", train_size="full", model_name="Qwen2-7B-Instruct", llm_type="sft"):
        self.dataset = dataset
        self.model_name = model_name
        self.dataset_name = dataset.split("-")[0]
        self.dataset_path = f"datasets/{dataset}" + ("-inductive" if setting=="inductive" else "")
        self.train_size = train_size
        self.model_path = f"{LLM_PATH}/{self.model_name}-{self.dataset_name}-{train_size}" if llm_type == "sft" else f"{LLM_PATH}/{self.model_name}"
        
        self.test_batch_size = 50                                    # 测试集中每50个sample为一个batch，并计算MRR和Hits@1
        self.max_type_triples = 5                                    # Type Reasoning阶段最多使用5个fewshot triples
        self.max_reason_paths = 6                                    # Path Reasoning阶段最多使用6个path，其中neighbor_triples和close_paths都最多六个
        self.max_path_hops = 3                                       # bfs搜索close_path的最大深度
        
        self.entity2text = self._load_text_file("entity2text.txt")
        self.relation2text = self._load_text_file("relation2text.txt")
        
        self.train_set = self._load_triples(f"train_{self.train_size}.txt")
        self.path_set = self._load_triples("inductive_graph.txt") if setting=="inductive" else self.train_set
        self.valid_set = self._load_triples(f"valid.txt")
        self.test_set_head = self._load_triples(f"ranking_head.txt")
        self.test_set_tail = self._load_triples(f"ranking_tail.txt")
        self.test_set = self.test_set_head + self.test_set_tail
        
        self.relation2headtail_dict = self._load_relation2headtail_dict(self.path_set)
        self.entity2relationtail_dict = self._load_entity2relationtail_dict(self.path_set)
        self.relation_degree_dict = self._load_relation_degree_dict(self.path_set)
        self.close_path_file = f"paths/close_path.json" if setting=="inductive" else f"paths/close_path_train_size_{self.train_size}.json"
        self.close_path_dict = self._load_close_path_dict(self.close_path_file)
        
        self.embedding_model = SentenceTransformer(
            model_name_or_path='BAAI/bge-small-en-v1.5',
            device="cuda"
        )

    def _load_text_file(self, filename):
        filepath = f"{self.dataset_path}/{filename}"
        with open(filepath, "r", encoding="utf-8") as file:
            return dict(line.strip().split('\t', 1) for line in file if line.strip())

    def _load_triples(self, filename):
        filepath = f"{self.dataset_path}/{filename}"
        with open(filepath, "r", encoding="utf-8") as file:
            return [line.strip().split('\t') for line in file if line.strip()]

    def _load_relation2headtail_dict(self, triple_set):
        relation2headtail_dict = defaultdict(list)
        for head, relation, tail in triple_set:
            relation2headtail_dict[relation].append([head, tail])
        return relation2headtail_dict
    
    def _load_entity2relationtail_dict(self, triple_set):
        entity2relationtail_dict = defaultdict(list)
        for head, relation, tail in triple_set:
            entity2relationtail_dict[head].append((relation, tail, 1))
            entity2relationtail_dict[tail].append((relation, head, -1))
        return entity2relationtail_dict

    def _load_relation_degree_dict(self, triple_set):
        relation_degree_dict = defaultdict(int)
        for _, relation, _ in triple_set:
            relation_degree_dict[relation] += 1
        return relation_degree_dict

    def _load_close_path_dict(self, filename):
        filepath = f"{self.dataset_path}/{filename}" 
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        return {}

    # 输入head entity和tail entity，使用bfs遍历搜索所有close_paths
    def bfs_paths(self, start, goal):
        queue = deque([(start, [], 0, set([start]))])
        paths = []
        while queue:
            current, path, hops, visited = queue.popleft()
            if hops < self.max_path_hops:
                for relation, neighbor, direction in self.entity2relationtail_dict[current]:
                    if direction == 1:
                        new_path = path + [(current, relation, neighbor)]
                    else:
                        new_path = path + [(neighbor, relation, current)]
                    if neighbor == goal:
                        paths.append(new_path)
                    elif neighbor not in visited:
                        queue.append((neighbor, new_path, hops + 1, visited | set([neighbor])))
        return paths
    
    # 用一个relation_degree计算所有close_paths的degree和，然后排序，取最小的几个，这样能排除"gender","ethnicity"等高频relation
    def close_path_finder(self, triple):
        head, relation, tail = triple
        head_tail = f"{head}-{tail}"
        close_paths = self.close_path_dict[head_tail]

        if close_paths:
            path_degrees = []
            for path in close_paths:
                degree_sum = sum(self.relation_degree_dict[rel] for _, rel, _ in path)
                path_degrees.append((degree_sum, path))
            path_degrees.sort(key=lambda x: x[0])
            
            top_paths = [path for _, path in path_degrees[:self.max_reason_paths]]
            top_paths.reverse()
            return top_paths

        return []
    
    def close_path_finder_no_degree(self, triple):
        head, relation, tail = triple
        head_tail = f"{head}-{tail}"
        close_paths = self.close_path_dict[head_tail]

        if close_paths:
            return close_paths[:self.max_reason_paths]

        return []

    def linearize_triple(self, triple):
        return f"({self.entity2text[triple[0]]}, {self.relation2text[triple[1]]}, {self.entity2text[triple[2]]})"
    
    def triple_to_sentence(self, triple):
        head, relation, tail = triple
        if self.dataset == "FB15k-237-subset":
            head_property = relation.split('/')[2]
            tail_property = relation.split('/')[-1]
            return f"('{self.entity2text[tail]}' is the {tail_property} of {head_property} '{self.entity2text[head]}')"
        elif self.dataset == "WN18RR-subset":
            return f"('{self.entity2text[head]}' {self.relation2text[relation]} '{self.entity2text[tail]}')"
        elif self.dataset == "NELL-995-subset":
            return f"('{self.entity2text[head]}' {self.relation2text[relation]} '{self.entity2text[tail]}')"
    
    def build_type_prompt(self, triple):
        fewshot_triples = self.diverse_fewshot_triple_finder(triple)
        fewshot_triples_sentence = '\n'.join(self.triple_to_sentence(triple) for triple in fewshot_triples)
        return TYPE_REASON_PROMPT.format(fewshot_triples=fewshot_triples_sentence, test_triple=self.triple_to_sentence(triple))

    def build_subgraph_prompt(self, triple):
        neighbor_triples = self.neighbor_triple_finder(triple)
        close_paths = self.close_path_finder(triple)
        reasoning_paths = "\n".join(
            " -> ".join(self.triple_to_sentence(triple) for triple in path)
            for path in close_paths
        )
        return SUBGRAPH_REASON_PROMPT.format(neighbor_triples="\n".join(neighbor_triples), reasoning_paths=reasoning_paths, test_triple=self.triple_to_sentence(triple))
    
    def build_neighbor_prompt(self, triple):
        neighbor_triples = self.neighbor_triple_finder(triple)
        return NEIGHBOR_REASON_PROMPT.format(neighbor_triples="\n".join(neighbor_triples), test_triple=self.triple_to_sentence(triple))
    
    def build_close_path_prompt(self, triple):
        close_paths = self.close_path_finder(triple)
        reasoning_paths = "\n".join(
            " -> ".join(self.triple_to_sentence(triple) for triple in path)
            for path in close_paths
        )
        return CLOSE_PATH_REASON_PROMPT.format(reasoning_paths=reasoning_paths, test_triple=self.triple_to_sentence(triple))
    
    def build_close_path_no_degree_prompt(self, triple):
        close_paths = self.close_path_finder_no_degree(triple)
        reasoning_paths = "\n".join(
            " -> ".join(self.triple_to_sentence(triple) for triple in path)
            for path in close_paths
        )
        return CLOSE_PATH_REASON_PROMPT.format(reasoning_paths=reasoning_paths, test_triple=self.triple_to_sentence(triple))
    
    def build_vanilla_prompt(self, triple):
        return BASE_REASON_PROMPT.format(test_triple=self.triple_to_sentence(triple))
    
    def build_all_prompt(self, triple):
        fewshot_triples = self.diverse_fewshot_triple_finder(triple)
        neighbor_triples = self.neighbor_triple_finder(triple)
        close_paths = self.close_path_finder(triple)
        fewshot_triples_sentence = '\n'.join(self.triple_to_sentence(triple) for triple in fewshot_triples)
        reasoning_paths = "\n".join(
            " -> ".join(self.triple_to_sentence(triple) for triple in path)
            for path in close_paths
        )
        return ALL_REASON_PROMPT.format(fewshot_triples=fewshot_triples_sentence, neighbor_triples="\n".join(neighbor_triples), reasoning_paths=reasoning_paths, test_triple=self.triple_to_sentence(triple))
    
    def get_test_batches(self):
        return [self.test_set[i:i + self.test_batch_size] for i in range(0, len(self.test_set), self.test_batch_size)]
    
    def diverse_fewshot_triple_finder(self, test_triple):
        test_head, relation, test_tail = test_triple
        head_tail_pairs = self.relation2headtail_dict[relation]
        
        if len(head_tail_pairs) <= self.max_type_triples:
            return [[head, relation, tail] for head, tail in head_tail_pairs]
        
        used_heads = {test_head, test_tail}
        used_tails = {test_tail, test_head}
        used_pairs = set()
        selected_triples = []
        
        for head, tail in head_tail_pairs:
            if head not in used_heads and tail not in used_tails:
                selected_triples.append([head, relation, tail])
                used_heads.add(head)
                used_tails.add(tail)
                used_pairs.add((head, tail))
                if len(selected_triples) == self.max_type_triples:
                    return selected_triples
         
        for head, tail in head_tail_pairs:
            if (head, tail) not in used_pairs:
                if len(selected_triples) < self.max_type_triples:
                    selected_triples.append([head, relation, tail])
                    used_heads.add(head)
                    used_tails.add(tail)
                    used_pairs.add((head, tail))
                else:
                    break
        
        return selected_triples
    
    # path_reasoning里面的neighbor triple，尽可能找到与当前triple相关的neighbor triple
    def neighbor_triple_finder(self, triple):
        head, relation, tail = triple
        head_triples = self.entity2relationtail_dict[head]
        tail_triples = self.entity2relationtail_dict[tail]

        triple_sentence = self.triple_to_sentence(triple)
        head_sentences = [self.triple_to_sentence((head, rel, t)) if direction == 1 else self.triple_to_sentence((t, rel, head))
                          for rel, t, direction in head_triples]
        tail_sentences = [self.triple_to_sentence((tail, rel, h)) if direction == 1 else self.triple_to_sentence((h, rel, tail))
                          for rel, h, direction in tail_triples]
        
        all_head_sentences = [triple_sentence] + head_sentences
        all_tail_sentences = [triple_sentence] + tail_sentences
        
        each_count = self.max_reason_paths // 2
        
        top_head_sentences = head_sentences
        top_tail_sentences = tail_sentences

        if len(head_sentences) > each_count:
            head_embeddings = self.embedding_model.encode(all_head_sentences, normalize_embeddings=True)
            head_similarity = head_embeddings[0] @ head_embeddings[1:].T
            top_head_indices = np.argsort(-head_similarity)[:each_count]
            top_head_sentences = [head_sentences[i] for i in top_head_indices]

        if len(tail_sentences) > each_count:
            tail_embeddings = self.embedding_model.encode(all_tail_sentences, normalize_embeddings=True)
            tail_similarity = tail_embeddings[0] @ tail_embeddings[1:].T
            top_tail_indices = np.argsort(-tail_similarity)[:each_count]
            top_tail_sentences = [tail_sentences[i] for i in top_tail_indices]
        
        return top_head_sentences + top_tail_sentences
    
    # 负采样：对于正样本triple，分别破坏head, relaton和tail，并为它们随机采样。
    def neg_sampling(self, pos_triple, count):
        head, relation, tail = pos_triple
        
        entities = set()
        for triple in self.path_set:
            entities.add(triple[0])
            entities.add(triple[2])
        
        candidate_entities = entities - {head, tail}
        seen_triples = {tuple(triple) for triple in self.path_set}
        negative_samples = []
        
        # 破坏head
        for _ in range(count):
            while True:
                new_head = random.choice(list(candidate_entities))
                if (new_head, relation, tail) not in seen_triples:
                    seen_triples.add((new_head, relation, tail))
                    negative_samples.append((new_head, relation, tail))
                    break
        
        # 破坏tail
        for _ in range(count):
            while True:
                new_tail = random.choice(list(candidate_entities))
                if (head, relation, new_tail) not in seen_triples:
                    seen_triples.add((head, relation, new_tail))
                    negative_samples.append((head, relation, new_tail))
                    break
                
        # # 破坏relation
        # candidate_relations = {triple[1] for triple in self.path_set} - {relation}
        # for _ in range(count):
        #     while True:
        #         new_relation = random.choice(list(candidate_relations))
        #         if (head, new_relation, tail) not in seen_triples:
        #             seen_triples.add((head, new_relation, tail))
        #             negative_samples.append((head, new_relation, tail))
        #             break
        
        return negative_samples
