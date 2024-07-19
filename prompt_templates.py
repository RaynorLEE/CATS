HYPER_PATTERN_BASED_SELECTOR_PROMPT = """Your task is to learn an ontology schema from example triples, where an ontology schema can be formally defined as [type, relation, type].
{fewshot_triple_input}
Determine whether the input triple matches the ontology schema you learned, which means each head and tail entity belongs to the same type as entity in the same position above. The input triple is:
{test_triple_input}
Your response should be "true" or "false". Do not say anything else except your determination.
"""
