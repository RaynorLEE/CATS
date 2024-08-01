ONTOLOGY_REASON_PROMPT = """Please determine whether the entities in the input triples are consistent in entity type with a set of known triples in the knowledge graph provided.
A set of known triples are:
{fewshot_triples}
The triple to be determined is:
{test_triple}
Please return 'Y' if the input triple is consistent in entity type, otherwise return 'N'. Do not say anything else except your determination.
"""

PATH_REASON_PROMPT = """Please determine whether the relation in the input can be reliably inferred between the head and tail entities, based on a set of neighbor triples and reasoning paths from the knowledge graph.
A set of neighbor triples from the knowledge graph are:
{neighbor_triples}
A set of reasoning paths from the knowledge graph are:
{reasoning_paths}
The relation to be inferred is:
{test_triple}
Please return 'Y' if there is sufficient evidence from the knowledge graph to infer the relation, otherwise return 'N'. Do not say anything else except your determination.
"""

EXPLAINING_PROMPT = """Please explain why the test triple can be reliably inferred based on the provided reasoning evidence from the knowledge graph. If it cannot be directly inferred from the evidence, provide a detailed justification for its correctness, including potential reasons. 
A set of neighbor triples from the knowledge graph are:
{known_triples}
A set of reasoning paths from the knowledge graph are:
{reasoning_paths}
The test triple to be explained is:
{test_triple}
Please provide a detailed explanation justifying why.
"""
