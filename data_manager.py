from collections import defaultdict

class DataManager:
    def __init__(self, dataset="FB15k-237-subset", inductive="True", direction="head", train_size="full"):
        self.dataset_path = f"data/{dataset}" + ("-inductive" if inductive else "")
        self.direction = direction
        self.train_size = train_size
        self.test_batch_size = 50
        
        self.entity2text = self._load_text_file("entity2text.txt")
        self.relation2text = self._load_text_file("relation2text.txt")
        
        self.train_set = self._load_triples(f"train_{self.train_size}.txt")
        self.inductive_set = self._load_triples("inductive_graph.txt") if inductive else self.train_set
        self.test_set = self._load_triples(f"ranking_{self.direction}.txt")
        
        self.train_dict = self._load_relation_dict(self.train_set)

    def _load_text_file(self, filename):
        filepath = f"{self.dataset_path}/{filename}"
        with open(filepath, "r", encoding="utf-8") as file:
            return dict(line.strip().split('\t', 1) for line in file if line.strip())

    def _load_triples(self, filename):
        filepath = f"{self.dataset_path}/{filename}"
        with open(filepath, "r", encoding="utf-8") as file:
            return [line.strip().split('\t') for line in file if line.strip()]

    def _load_relation_dict(self, triple_set):
        relation2dict = defaultdict(list)
        for head, relation, tail in triple_set:
            relation2dict[relation].append([head, tail])
        return relation2dict

    def get_test_batches(self):
        return [self.test_set[i:i + self.test_batch_size] for i in range(0, len(self.test_set), self.test_batch_size)]

    def fewshot_triple_finder(self, relation, fewshot_count):
        head_tail_pairs = self.train_graph[relation][:fewshot_count]
        return [[head, relation, tail] for head, tail in head_tail_pairs]
    
    def triple_to_sentence(self, triple):
        return f"[{self.entity2text[triple[0]]}, {self.relation2text[triple[1]]}, {self.entity2text[triple[2]]}]"
    
    def triple_to_question(self, triple):
        return ""
    