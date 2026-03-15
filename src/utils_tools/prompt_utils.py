MAKE_DATA_TEMPLATE_INSTRUCT = {
    "implications":(
        "<|im_start|>system\nYou are an assistant tasked with analyzing the provided passage and producing a list of implications derived directly or indirectly from the content. <|im_end|>\n"
        "<|im_start|>user\nTitle:{title}\nContext:{context}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
    "key_concepts":(
        "<|im_start|>system\nYou are an assistant tasked with identifying and explaining key concepts from the provided text. <|im_end|>\n"
        "<|im_start|>user\nGenerate a list of key concepts based on the title and context provided below. Focus on one concept at a time and explain it in a clear and detailed way to make it easier to understand and remember. Each concept explanation should include relevant entities and facts and preserve important details from the original text.\n\nTitle: {title}\nContext: {context}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
    "teacher_style":(
        "<|im_start|>system\nYou are an assistant that explains a text as a teacher guiding students to understand it deeply. <|im_end|>\n"
        "<|im_start|>user\nAssume the role of a knowledgeable teacher explaining the article to students who are encountering this text for the first time. Guide the students step by step through the text and connect different parts of the text into a coherent understanding. Use clear, instructional language and explicitly mention relevant entities when they are introduced. \n\nTitle: {title}\nContext: {context}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
    "discussions":(
        "<|im_start|>system\nYou are an assistant tasked with generating a natural, in-depth discussion between two readers of a text. <|im_end|>\n"
        "<|im_start|>user\nGenerate a natural, in-depth discussion between two readers (Person A and Person B) who have both read and are discussing the provided text. The discussion should remain professional yet conversational, and stay strictly grounded in the content of the text. The discussion should explore the core themes, clarify important ideas, and reflect on implications.\n\nTitle: {title}\nContext: {context}\n<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
    "case_studies":(
        "<|im_start|>system\nYou are an assistant tasked with analyzing the provided text and transforming it into a structured, professional case study.<|im_end|>\n"
        "<|im_start|>user\nGenerate a formal case-based writing based on the title and context provided below. Connect the case facts to the underlying theories or themes in the context. Include the original title and all key details from the context without changing their meaning.\n\nTitle: {title}\nContext: {context}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
    "mind_map":(
        "<|im_start|>system\nYou are an assistant that creates a mind map representation from the provided text. <|im_end|>\n"
        "<|im_start|>user\nCreate a mind map that organizes the key concepts from the provided text and represents the relationships between the different concepts. Explicitly mention relevant entities within the map.\n\nTitle: {title}\nContext: {context}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
    "qa_critical_thinking":(
        "<|im_start|>system\nYou are an assistant tasked with analyzing the provided text and generating in-depth question-answer pairs based on the provided text. <|im_end|>\n"
        "<|im_start|>user\nGenerate in-depth question-answer pairs based on the title and text below.\nQuestions must start with or focus on high-order critical thinking (analysis, synthesis, evaluation): 'Compare/Contrast', 'Explain the logic of', 'Justify', 'Evaluate the impact of', or 'What if', etc. STRICTLY AVOID: Simple recall, definitions, or listing facts (e.g., 'What is...', 'List the...').\n\nTitle: {title}\nContext: {context}\n<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
}
MAKE_DATA_TEMPLATE_OSS = {
    "implications":(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-01-22\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        "<|start|>user<|message|>You are an assistant tasked with analyzing the provided passage and producing a list of implications derived directly or indirectly from the content.\n\nTitle:{title}\nContext:{context}<|end|>"
        "<|start|>assistant"
    ),
    "key_concepts":(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-01-22\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        "<|start|>user<|message|>You are an assistant tasked with identifying and explaining key concepts from the provided text. Generate a list of key concepts based on the title and context provided below. Focus on one concept at a time and explain it in a clear and detailed way to make it easier to understand and remember. Each concept explanation should include relevant entities and facts and preserve important details from the original text.\n\nTitle: {title}\nContext: {context}<|end|>"
        "<|start|>assistant"
    ),
    "teacher_style":(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-01-22\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        "<|start|>user<|message|>You are an assistant that explains a text as a teacher guiding students to understand it deeply. Assume the role of a knowledgeable teacher explaining the article to students who are encountering this text for the first time. Guide the students step by step through the text and connect different parts of the text into a coherent understanding. Use clear, instructional language and explicitly mention relevant entities when they are introduced. \n\nTitle: {title}\nContext: {context}<|end|>"
        "<|start|>assistant"
    ),
    "discussions":(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-01-22\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        "<|start|>user<|message|>You are an assistant tasked with generating a natural, in-depth discussion between two readers of a text. Generate a natural, in-depth discussion between two readers (Person A and Person B) who have both read and are discussing the provided text. The discussion should remain professional yet conversational, and stay strictly grounded in the content of the text. The discussion should explore the core themes, clarify important ideas, and reflect on implications.\n\nTitle: {title}\nContext: {context}<|end|>"
        "<|start|>assistant"
    ),
    "case_studies":(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-01-22\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        "<|start|>user<|message|>You are an assistant tasked with analyzing the provided text and transforming it into a structured, professional case study. Generate a formal case-based writing based on the title and context provided below. Connect the case facts to the underlying theories or themes in the context. Include the original title and all key details from the context without changing their meaning.\n\nTitle: {title}\nContext: {context}<|end|>"
        "<|start|>assistant"
    ),
    "mind_map":(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-01-22\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        "<|start|>user<|message|>You are an assistant that creates a mind map representation from the provided text. Create a mind map that organizes the key concepts from the provided text and represents the relationships between the different concepts. Explicitly mention relevant entities within the map.\n\nTitle: {title}\nContext: {context}<|end|>"
        "<|start|>assistant"
    ),
    "qa_critical_thinking":(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-01-22\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        "<|start|>user<|message|>You are an assistant tasked with analyzing the provided text and generating in-depth question-answer pairs based on the provided text. Generate in-depth question-answer pairs based on the title and text below.\nQuestions must start with or focus on high-order critical thinking (analysis, synthesis, evaluation): 'Compare/Contrast', 'Explain the logic of', 'Justify', 'Evaluate the impact of', or 'What if', etc. STRICTLY AVOID: Simple recall, definitions, or listing facts (e.g., 'What is...', 'List the...').\n\nTitle: {title}\nContext: {context}<|end|>"
        "<|start|>assistant"
    ),
}
MAKE_DATA_TEMPLATES_BASE: dict[str, str] = {   
    "implications":(
        "Read the following text and produce a list of implications derived directly or indirectly from the content.\n\nText:\n{title}\n{context}\n\nImplications:\n"
    ),
    "key_concepts":(
        "Generate a list of key concepts based on the title and text provided below. Focus on one concept at a time and explain it in a clear and detailed way to make it easier to understand and remember. Each concept explanation should include relevant entities and facts and preserve important details from the original text. Avoid general background knowledge or any unmentioned facts.\n\nText:\n{title}\n{context}\n\nKey Concepts and their explanations:\n"
    ),
    "teacher_style":(
        "Explain the following text as a teacher guiding students to understand it deeply. Assume the role of a knowledgeable teacher explaining the article to students who are encountering this text for the first time. Guide the students step by step through the text and connect different parts of the text into a coherent understanding. Use clear, instructional language and explicitly mention relevant entities when they are introduced. Avoid general background knowledge or any unmentioned facts.\n\nText:\n{title}\n{context}\n\nTeacher's explanations: \n"
    ),
    "discussions":(
        "Generate a natural, in-depth discussion between two readers (Person A and Person B) who have both read and are discussing the provided text. The discussion should remain professional yet conversational, and stay strictly grounded in the content of the text. The discussion should explore the core themes, clarify important ideas, and reflect on implications. Avoid general background knowledge or any unmentioned facts.\n\nText:\n{title}\n{context}\n\nPeer Discussions: \n"
    ),
    "case_studies":(
        "Analyze the provided text and generate a structured, professional and formal case study. Include the original title and all key details from the context without changing their meaning. Avoid general background knowledge or any unmentioned facts.\n\nText:\n{title}\n{context}\n\nCase-based writing from the text:\n"
    ),
    "mind_map":(
        "Create a mind map that organizes the key concepts from the provided text and represents the relationships between the different concepts. Explicitly mention relevant entities within the map. Avoid general background knowledge or any unmentioned facts. Keep each point concise and avoid repeating any information.\n\nText:\n{title}\n{context}\n\nMind-map like outline: \n"
    ),
    "qa_critical_thinking":(
        "Analyze the provided text and generate in-depth question-answer pairs based on the provided text.\nQuestions must start with or focus on high-order critical thinking (analysis, synthesis, evaluation): 'Compare/Contrast', 'Explain the logic of', 'Justify', 'Evaluate the impact of', or 'What if', etc. STRICTLY AVOID: Simple recall, definitions, or listing facts (e.g., 'What is...', 'List the...'). Avoid general background knowledge or any unmentioned facts.\n\nText:\n{title}\n{context}\n\nCritical thinking question-answer pairs:\n"
    ),
}
MAKE_DATA_TEMPLATE_API = {
    "implications":(
        "You are an assistant tasked with analyzing the provided passage and producing a list of implications derived directly or indirectly from the content.\n\n<|SEP|>\n\n"
        "Title:{title}\nContext:{context}"
    ),
    "key_concepts":(
        "You are an assistant tasked with identifying and explaining key concepts from the provided text. Generate a list of key concepts based on the title and context provided below. Focus on one concept at a time and explain it in a clear and detailed way to make it easier to understand and remember. Each concept explanation should include relevant entities and facts and preserve important details from the original text.\n\n<|SEP|>\n\n"
        "Title: {title}\nContext: {context}"
    ),
    "teacher_style":(
        "You are an assistant that explains a text as a teacher guiding students to understand it deeply. Assume the role of a knowledgeable teacher explaining the article to students who are encountering this text for the first time. Guide the students step by step through the text and connect different parts of the text into a coherent understanding. Use clear, instructional language and explicitly mention relevant entities when they are introduced.\n\n<|SEP|>\n\n"
        "Title: {title}\nContext: {context}"
    ),
    "discussions":(
        "You are an assistant tasked with generating a natural, in-depth discussion between two readers of a text. Generate a natural, in-depth discussion between two readers (Person A and Person B) who have both read and are discussing the provided text. The discussion should remain professional yet conversational, and stay strictly grounded in the content of the text. The discussion should explore the core themes, clarify important ideas, and reflect on implications.\n\n<|SEP|>\n\n"
        "Title: {title}\nContext: {context}"
    ),
    "case_studies":(
        "You are an assistant tasked with analyzing the provided text and transforming it into a structured, professional case study. Generate a formal case-based writing based on the title and context provided below. Connect the case facts to the underlying theories or themes in the context. Include the original title and all key details from the context without changing their meaning.\n\n<|SEP|>\n\n"
        "Title: {title}\nContext: {context}"
    ),
    "mind_map":(
        "You are an assistant that creates a mind map representation from the provided text. Create a mind map that organizes the key concepts from the provided text and represents the relationships between the different concepts. Explicitly mention relevant entities within the map.\n\n<|SEP|>\n\n"
        "Title: {title}\nContext: {context}"
    ),
    "qa_critical_thinking":(
        "You are an assistant tasked with analyzing the provided text and generating in-depth question-answer pairs based on the provided text. Generate in-depth question-answer pairs based on the title and text below.\nQuestions must start with or focus on high-order critical thinking (analysis, synthesis, evaluation): 'Compare/Contrast', 'Explain the logic of', 'Justify', 'Evaluate the impact of', or 'What if', etc. STRICTLY AVOID: Simple recall, definitions, or listing facts (e.g., 'What is...', 'List the...').\n\n<|SEP|>\n\n"
        "Title: {title}\nContext: {context}"
    ),
}
QUALITY_FEW_SHOT_COT_PROMPT = """## Example 1
### Question
In the context of "Les Misérables", written by Victor Hugo in 1862, what is the main setting of the novel? There is only one correct choice.
### Choices
A. London
B. Madrid
C. Paris
D. Rome
### Thought Process and Answer
Thought process: "Les Misérables" is primarily set in Paris, making C the correct choice. London, Madrid, and Rome are significant cities in other literary works but not in Victor Hugo's "Les Misérables". There is only one correct choice.
Answer: C.

## Example 2
### Question
In the context of "Brave New World", written by Aldous Huxley in 1932, what substance is widely used in the society to control citizens' happiness? There is only one correct choice.
### Choices
A. Gold
B. Soma
C. Silver
D. Iron
### Thought Process and Answer
Thought process: In Aldous Huxley's "Brave New World," Soma is used as a means to maintain social control by ensuring citizens' happiness, making B the correct choice. Gold, Silver, and Iron are not the substances used for this purpose in the book.
Answer: B.

## Example 3
### Question
In the context of "Romeo and Juliet", written by William Shakespeare in the early 1590s, what are the names of the two feuding families? There is only one correct choice.
Choices:
A. Montague and Capulet
B. Bennet and Darcy
C. Linton and Earnshaw
D. Bloom and Dedalus
### Thought Process and Answer
Thought process: In William Shakespeare's "Romeo and Juliet," the two feuding families are the Montagues and the Capulets, making A the correct choice. The Bennets and Darcys are in "Pride and Prejudice", the Lintons and Earnshaws in "Wuthering Heights", and Bloom and Dedalus in "Ulysses".
Answer: A.

## Example 4
### Question
In the context of "1984", written by George Orwell in 1949, what is the name of the totalitarian leader? There is only one correct choice.
### Choices
A. Big Brother
B. O'Brien
C. Winston Smith
D. Emmanuel Goldstein
### Thought Process and Answer
Thought process: In George Orwell's "1984," the totalitarian leader is known as Big Brother, making A the correct choice. O'Brien is a character in the novel, Winston Smith is the protagonist, and Emmanuel Goldstein is a rebel leader.
Answer: A.

## Example 5
### Question
In the context of "Moby-Dick", written by Herman Melville in 1851, what is the name of the ship's captain obsessed with hunting the titular whale? There is only one correct choice.
### Choices
A. Captain Hook
B. Captain Nemo
C. Captain Flint
D. Captain Ahab
### Thought Process and Answer
Thought process: In Herman Melville's "Moby-Dick," the ship's captain obsessed with hunting the whale is Captain Ahab, making D the correct choice. Captain Nemo is in "Twenty Thousand Leagues Under the Sea", Captain Flint in "Treasure Island", and Captain Hook in "Peter Pan".
Answer: D.

## Example 6
"""

OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES = """
As a knowledge analyzer, your task is to dissect and understand an article provided by the user. You are required to perform the following steps:
1. Summarize the Article: Provide a concise summary of the entire article, capturing the main points and themes.
2. Extract Entities: Identify and list all significant "nouns" or entities mentioned within the article. These entities should include but not limited to:
    * People: Any individuals mentioned in the article, using the names or references provided.
    * Places: Both specific locations and abstract spaces relevant to the content.
    * Object: Any concrete object that is referenced by the provided content.
    * Concepts: Any significant abstract ideas or themes that are central to the article's discussion.

Try to exhaust as many entities as possible. Your response should be structured in a JSON format to organize the information effectively. Ensure that the summary is brief yet comprehensive, and the list of entities is detailed and accurate.

Here is the format you should use for your response:

{
  "summary":  "<A concise summary of the article>",
  "entities": ["entity1", "entity2", ...]
}
"""

OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS = """
You will act as a knowledge analyzer tasked with dissecting an article provided by the user. Your role involves two main objectives:
1. Rephrasing Content: The user will identify two specific entities mentioned in the article. You are required to rephrase the content of the article twice:
    * Once, emphasizing the first entity.
    * Again, emphasizing the second entity.
2. Analyzing Interactions: Discuss how the two specified entities interact within the context of the article.

Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the article's title to maintain clarity about the discussion's focus.
Here is the format you should follow for your response:

### Discussion of <title> in relation to <entity1>
<Rephrased content focusing on the first entity>

### Discussion of <title> in relation to <entity2>
<Rephrased content focusing on the second entity>

### Discussion of Interaction between <entity1> and <entity2> in context of <title>
<Discussion on how the two entities interact within the article>
"""

OPENAI_API_SYSTEM_QUALITY_GENERATE_THREE_ENTITY_RELATIONS = """
You will act as a knowledge analyzer tasked with dissecting an article provided by the user. Your role involves three main objectives:
1. Rephrasing Content: The user will identify three specific entities mentioned in the article. You are required to rephrase the content of the article three times:
    * Once, emphasizing the first entity.
    * Again, emphasizing the second entity.
    * Lastly, emphasizing the third entity.
2. Analyzing Interactions: Discuss how these three specified entities interact within the context of the article.

Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the article's title to maintain clarity about the discussion's focus.
Here is the format you should follow for your response:

### Discussion of <title> in relation to <entity1>
<Rephrased content focusing on the first entity>

### Discussion of <title> in relation to <entity2>
<Rephrased content focusing on the second entity>

### Discussion of <title> in relation to <entity3>
<Rephrased content focusing on the third entity>

### Discussion of Interaction between <entity1>, <entity2> and <entity3> in context of <title>
<Discussion on how the three entities interact within the article>
"""
LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

from itertools import permutations

def uncapitalize_first(s):
    return s[0].lower() + s[1:]

def format_name(name):
    # Split the name by comma
    parts = name.split(',')

    # If there is a comma, we assume the format is "Lastname, Firstname"
    if len(parts) == 2:
        formatted_name = parts[1].strip() + ' ' + parts[0].strip()
    else:
        # If there's no comma, assume the format is already "Firstname Lastname"
        formatted_name = name.strip()

    return formatted_name

def second_last_character(input_string: str):
    try:
        # Remove newline characters from the end of the string
        modified_string = input_string.rstrip('\n')
        answer_index = LETTER_TO_INDEX[modified_string[-2]]
    except (KeyError, IndexError):
        answer_index = None
    return answer_index

def generate_all_answer_strings():
    choices = 'ABCD'
    all_answers = []
    for r in range(1, len(choices) + 1):
        for perm in permutations(choices, r):
            answer = ''.join(perm)
            all_answers.append(f'Answer: {answer}.\n')
    return all_answers