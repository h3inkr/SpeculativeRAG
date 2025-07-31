# Templates for the system instructions.
# * Each explanations are seperated by "*".
CSQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 5 options (labeled A, B, C, D, and E). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 5 options (labeled A, B, C, D, and E), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 5 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 5 options, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge into high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and accurate explanation.
* The explanations must be simple and concise.
Do you understand the task?\
""",
#Your task is to select the most relevant external knowledge for answering the given question. \
'knowledge_selection': """\
You are given a question, 5 options, explanations, and a list of external knowledge. \
Your task is to select the most relevant external knowledge to refine given explanations. \
Note that:
* You should answer with the number of documents.
* If there is no relevant knowledge, you should select "None".
* If given explanations are sufficient to answer for the question, you should select "None".
Do you understand the task?\
""",
#Based on the external knowledge, your task is to refine the given explanations into high-quality explanations that support the most likely option. \
#* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
'knowledge_refinement': """\
You are given a question, 5 options, explanations, and a list of external knowledge. \
Based on the external knowledge and explanations, your task is to synthesize these knowledge into high-quality explanations that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be simple and concise.
Do you understand the task?\
""",
}
#* The explanations must be concise and accurate (max 15 words).

CSQA2 = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 2 options (labeled A and B). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 2 options (labeled A and B), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 2 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 2 options, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge into one or more high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question, 2 options, explanations, and a list of external knowledge. \
Your task is to select the most relevant external knowledge to refine the explanations that supports the most likely option. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
'knowledge_refinement': """\
You are given a question, 2 options, explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}


PIQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question or a text to complete and 2 possible solutions (labeled A and B). \
Your task is to choose the label corresponding to the best solution. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question or a text to complete, 2 possible solutions (labeled A and B), and a list of explanations. \
Your task is to choose the label corresponding to the best solution based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question or a text to complete and 2 possible solutions. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely solution. \
Note that:
* Each explanations are seperated by "*".
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question or a text to complete, 2 possible solutions, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge into one or more high-quality explanations that support the most likely solution. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 2 possible solutions, explanations, and a list of external knowledge. \
Your task is to select the most useful external knowledge to refine the explanations that supports the most likely solution. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
'knowledge_refinement': """\
You are given a question or a text to complete, 2 solutions, explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}

ARC = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and up to 5 options (labeled A, B, C, D, and E). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, up to 5 options (labeled A, B, C, D, and E), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and up to 5 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* the explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 5 options, and external knowledge references. \
Based on the external knowledge, your task is to synthesize these knowledge into one or more high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question, 5 options, explanations, and a list of external knowledge. \
Your task is to select the most relevant external knowledge to refine the explanations that supports the most likely option. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
#Based on the external knowledge, your task is to refine the given explanations into high-quality explanations that support the most likely option. \
#* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
'knowledge_refinement': """\
You are given a question, 5 options, explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}

OBQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 4 options (labeled A, B, C, and D). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 4 options (labeled A, B, C, and D), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 4 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* the explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 4 options, and external knowledge references. \
Based on the external knowledge, your task is to synthesize these knowledge into one or more high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question, 4 options, explanations, and a list of external knowledge. \
Your task is to select the most relevant external knowledge to refine the explanations that supports the most likely option. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
#Based on the external knowledge, your task is to refine the given explanations into high-quality explanations that support the most likely option. \
#* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
'knowledge_refinement': """\
You are given a question, 4 options, explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}


QASC = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 8 options (labeled A, B, C, D, E, F, G and H). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 8 options (labeled A, B, C, D, E, F, G and H), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 8 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* the explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",    
"knowledge_generation_external": """\
You are given a question, 8 options, and external knowledge references. \
Based on the external knowledge, your task is to synthesize these knowledge into one or more high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question, 8 options, explanations, and a list of external knowledge. \
Your task is to select the most relevant external knowledge to refine the explanations that supports the most likely option. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
#Based on the external knowledge, your task is to refine the given explanations into high-quality explanations that support the most likely option. \
#* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
'knowledge_refinement': """\
You are given a question, 8 options, explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}

WG = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question where one word has been replaced with \"_\" and 2 options (labeled A and B) to replace \"_\". \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 2 options (labeled A and B) to replace \"_\", and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question where one word has been replaced with \"_\" and 2 options to replace \"_\". \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* the explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
 "knowledge_generation_external": """\
You are given a question where one word has been replaced with \"_\", 2 options to replace \"_\", and external knowledge references. \
Based on the external knowledge, your task is to synthesize these knowledge into one or more high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question where one word has been replaced with \"_\", 2 options to replace \"_\", explanations, and a list of external knowledge. \
Your task is to select the most relevant external knowledge to refine the explanations that supports the most likely option. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
#Based on the external knowledge, your task is to refine the given explanations into high-quality explanations that support the most likely option. \
#* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
'knowledge_refinement': """\
You are given a question where one word has been replaced with \"_\", 2 options to replace \"_\", explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}


##############OOD Dataset Prompts #######################

SIQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 3 options (labeled A, B and C). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 3 options (labeled A, B and C), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 3 options. \
Your task is to write one or more explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* the explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 3 options, and external knowledge references. \
Based on the external knowledge, your task is to synthesize these knowledge into comprehensive high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 3 options, explanations, and a list of external knowledge. \
Your task is to select the most useful external knowledge to refine the explanations that supports the most likely option. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
'knowledge_refinement': """\
You are given a question or a text to complete, 3 options, explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}


HellaSWAG = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 4 options (labeled A, B, C, and D). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 4 options (labeled A, B, C, and D), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 4 options. \
Your task is to write one or more explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* the explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 4 options, and external knowledge references. \
Based on the external knowledge, your task is to synthesize these knowledge into comprehensive high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 4 options, explanations, and a list of external knowledge. \
Your task is to select the most useful external knowledge to refine the explanations that supports the most likely option. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
'knowledge_refinement': """\
You are given a question or a text to complete, 4 options, explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}

COM2SENSE = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 2 options (labeled A and B). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 2 options (labeled A and B), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 2 options. \
Your task is to write one or more explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* the explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 2 options, and external knowledge references. \
Based on the external knowledge, your task is to synthesize these knowledge into comprehensive high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 2 options, explanations, and a list of external knowledge. \
Your task is to select the most useful external knowledge to refine the explanations that supports the most likely option. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
'knowledge_refinement': """\
You are given a question or a text to complete, 2 options, explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}

RIDDLESENSE = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 5 options (labeled A, B, C, D, and E). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 5 options (labeled A, B, C, D, and E), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 5 options. \
Your task is to write one or more explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 5 options, and external knowledge references. \
Based on the external knowledge, your task is to synthesize these knowledge into comprehensive high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 5 options, explanations, and a list of external knowledge. \
Your task is to select the most useful external knowledge to refine the explanations that supports the most likely option. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
'knowledge_refinement': """\
You are given a question or a text to complete, 5 options, explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}


NUMERSENSE = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 12 options (labeled A, B, C, D, E, F, G, H, I, J, K, and L). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 12 options (labeled A, B, C, D, E, F, G, H, I, J, K, and L), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 12 options. \
Your task is to write one or more explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 12 options, and external knowledge references. \
Based on the external knowledge, your task is to synthesize these knowledge into comprehensive high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 12 options, explanations, and a list of external knowledge. \
Your task is to select the most useful external knowledge to refine the explanations that supports the most likely option. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
'knowledge_refinement': """\
You are given a question or a text to complete, 12 options, explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}


QUARTZ = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 2 options (labeled A and B). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 2 options (labeled A and B), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 2 options. \
Your task is to write one or more explanations that support the most likely option. \
Note that:
* Each explanations are seperated by "*".
* the explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 2 options, and external knowledge references. \
Based on the external knowledge, your task is to synthesize these knowledge into comprehensive high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given knowledge, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 2 options, explanations, and a list of external knowledge. \
Your task is to select the most useful external knowledge to refine the explanations that supports the most likely option. \
Note that:
* If there is no relevant knowledge, you should select "None".
* If the given explanation is sufficient to choose the label corresponding to the best answer for the question, you should select "None".
Do you understand the task?\
""",
'knowledge_refinement': """\
You are given a question or a text to complete, 2 options, explanations, and a list of external knowledge. \
Based on the external knowledge, your task is to synthesize these knowledge and the given explanation into the high-quality explanation that support the most likely options. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
* Do not simply replicate the given explanation, but you should offer a refined and accurate explanation.
* The explanations must be concise and accurate.
Do you understand the task?\
""",
}



#HellaSwag / Com2sense / Nummersense ... 


# Templates for the fewshot examples.
MCQ_EXAMPLE_TEMPLATE = """\
Question:
{question}

Options:
{choices}\
"""

MCQ_WITH_KNOWLEDGE_EXAMPLE_TEMPLATE = """\
Question:
{question}

Options:
{choices}

Explanations:
{knowledge}\
"""

KNOWLEDGE_GENERATION_EXAMPLE_TEMPLATE = """\
Question:
{question}

Options:
{choices}\
"""

KNOWLEDGE_GENERATION_EXAMPLE_TEMPLATE_EXTERNAL = """\
Question:
{question}

Options:
{choices}\
"""

KNOWLEDGE_SELECTION = """\
Question:
{question}

Options:
{choices}

Explanations:
{knowledge}\
"""


SHOT_TEMPLATES = {
    "mcq": MCQ_EXAMPLE_TEMPLATE,
    "mcq_with_kg": MCQ_WITH_KNOWLEDGE_EXAMPLE_TEMPLATE,
    "knowledge_generation": KNOWLEDGE_GENERATION_EXAMPLE_TEMPLATE,
    "knowledge_generation_external" : KNOWLEDGE_GENERATION_EXAMPLE_TEMPLATE_EXTERNAL,
    "knowledge_selection" : KNOWLEDGE_SELECTION
}

# Mapping from dataset names to their respective instruction prompts.
OUR_DATASET_TAGS = {
    "csqa": CSQA,
    "csqa2": CSQA2,
    "piqa": PIQA,
    "siqa": SIQA,
    "hellaswag" : HellaSWAG,
    "com2sense" : COM2SENSE,
    "riddlesense" : RIDDLESENSE,
    "obqa" : OBQA,
    "qasc" : QASC,
    "wg" : WG,
    "arc-challenge" : ARC,
    "arc-easy" : ARC,
    "numersense" : NUMERSENSE,
    "quartz" : QUARTZ,
}