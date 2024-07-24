# 下面这个prompt在FB237上，所有batch的groundtruth保留率为0.92（越高越好），过滤样本剩余比例为0.18（越低越好）
# SELECTING_PROMPT = """Determine whether the input triple is consistent within the same ontological framework, given a set of known triples of the knowledge graph.
# A set of known triples are:
# {fewshot_triples}
# The triple to be determined is:
# {test_triple}
# Please return 'Y' if the input triple is consistent, otherwise return 'N'. Do not say anything else except your determination.
# """

# 下面这个prompt在FB237上，所有batch的groundtruth保留率为0.93（越高越好），过滤样本剩余比例为0.20（越低越好）
SELECTING_PROMPT = """Please determine whether the entities in the input triples are consistent in type with a set of known triples in the knowledge graph provided.
A set of known triples are:
{fewshot_triples}
The triple to be determined is:
{test_triple}
Please return 'Y' if the input triple is consistent, otherwise return 'N'. Do not say anything else except your determination.
"""

REASONING_PROMPT = """Determine whether the relation in the input can be reliably inferred, given a set of reasoning paths between two entities of known triples of the knowledge graph.
A set of reasoning paths are:
{reasoning_paths}
The relation to be inferred is:
{test_triple}
Return 'Y' if there is sufficient evidence from the reasoning paths to infer the relation, otherwise return 'N'. Do not say anything else except your determination.
"""

REASONING_LONGTAIL_PROMPT = """Determine whether the relation in the input can be reliably inferred, given a set of known triples including two entities of the knowledge graph.
A set of known triples are:
{known_triples}
The relation to be inferred is:
{test_triple}
Return 'Y' if there is sufficient evidence from the known triples to infer the relation, otherwise return 'N'. Do not say anything else except your determination.
"""

EXPLAINING_PROMPT = """Explain why the test triple can be reliably inferred based on the provided reasoning evidence. If it cannot be directly inferred from the evidence, provide a detailed justification for its correctness, including potential reasons. 
The reasoning evidence is:
{reasoning_evidence}
The test triple to be explained is:
{test_triple}
Please provide a detailed explanation justifying why.
"""
