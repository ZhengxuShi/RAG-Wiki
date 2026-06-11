from hypster import HP
from textwrap import dedent

prompt_filter = dedent("""\
    Task: Please extract the possible key words in the context of the question from it, no more than three, including both Chinese and English versions:
    Note: Key words should be provided in the form of list.
    examples:

    example1: What is the definition of an article under REACH?
    [物品, article]

    example2: When will REACH come into mandatory effect?
    [强制生效，enter into force,时间, time]
    Please output the key words of the following question:
    Question:{{query}}                   
    """)


def response_config(hp: HP):

    prompt_response = dedent("""\
    Given the following information,
    answer the question concisely in one to two sentences, pay attention, answer in Chinese.
    using only the relevant details provided in the documents.
    Support your answer with a brief, word-for-word quote from the most pertinent document. 
    Note that some documents may not be relevant to the question.
    No need to include the source of the answer in the response.(Such as:(1)This information is derived from...)
    ========================================
    Context:
    {% for document in documents %}
    Document {{loop.index}}:
    {{ document.meta.llm_extracted_info }}
    {{ document.content }}
    ---
    {% endfor %}
    ========================================
    Question: {{query}}

    Answer:
    """)

    prompt_hyde = dedent("""\
    Given a question, generate a paragraph of text that answers the question.
    Question: {{query}}
    Paragraph:                    
    """)

    prompt_language = dedent("""\
        Prompt Template:
        ========================================
        To enhance retrieval, the question is now being rephrased.
        Given a Chinese question, translate it into English as the answer.
        Question: {{query}}
        Answer:                    
        """)

    prompt_association = dedent("""\
    Prompt Template:
    ========================================
    Given Question:  {{query}}
    Task: Generate three associated questions based on the given question. The nouns in the given question are crucial; retain the most distinguishing nouns. Each associated question must include these nouns, and each should align with the theme or purpose of the original question but differ in approach or specific details.
    Note: The following three associated questions should be provided in the form of key-value pairs.
    {
    "Associated Question 1": "{{associated_question_1}}",
    "Associated Question 2": "{{associated_question_2}}",
    "Associated Question 3": "{{associated_question_3}}"
    }
    ========================================
    """)

    prompt_enrich_doc_key = dedent("""
    According to the document's context. 
    Then list 3-5 keywords or acronyms that best \
    represent its content for search purposes.
    The extracted keywords must be in the same language as the document's context! Do not translate arbitrarily.
    ========================================
    Context:
    {{ document.content[:1500] }}

    ============================

    Output format:

    Keywords:
    """)

    prompt_enrich_doc_summary = dedent("""
    Summarize the document's main topic in one sentence (100 words max).
    Represent its content for search purposes.
    The extracted summary must be in the same language as the document's context! Do not translate arbitrarily.
    ========================================
    Context:
    {{ document.content[:1500] }}

    ============================

    Output format:

    Summary:
    """)
    prompt_enrich_doc_link_summary = dedent("""
        As a RAG retrieval augmentation expert, please generate a retrieval-optimized summary representation based on the following contextual content. The summary length should range from 0 to 500 characters (dynamically adjusted according to content value), must be in simplified Chinese, and strictly adhere to the original information without adding external content.
        ========================================
        Context:
        {{ document.meta['metadata']['content_link'][:12000] }}
        ============================
        [Requirements]
        1. Extract core entities, key events, and data metrics
        2. Maintain factual accuracy and semantic integrity of the original text
        3. Adopt a dense expression style using noun phrase combinations
        4. Prioritize preserving professional terminology and specific expressions
        5. Avoid decorative language and subjective evaluations
        ============================
        Output format:
        Summary:
        """)

    prompt_enrich_doc_summary_key = dedent("""
    According to the document's context. 
    Then list 3-5 keywords or acronyms that best \
    represent its content for search purposes.
    The extracted keywords must be in the same language as the document's context! Do not translate arbitrarily.
    ========================================
    Context:
    {{ document[:1000] }}

    ============================

    Output format:

    Keywords:
    """)

    prompt_is_second_retrieval = dedent("""
    Based on the content provided in the document, 
    assess whether the questions posed can be answered correctly.
    Do not provide the answers, only indicate the answerability of the questions, only answer yes or no.
    ========================================
    Context:
    {% for document in documents %}
    Document {{loop.index}}:
    {{ document.meta.llm_extracted_info }}
    {{ document.content }}
    ---
    {% endfor %}
    ========================================
    Question: {{query}}

    Answer:yes/no
    """)
    return {
        "prompt_response": prompt_response,
        "prompt_hyde": prompt_hyde,
        "prompt_language": prompt_language,
        "prompt_association": prompt_association,
        "prompt_enrich_doc_key": prompt_enrich_doc_key,
        "prompt_enrich_doc_summary": prompt_enrich_doc_summary,
        "prompt_enrich_doc_summary_key": prompt_enrich_doc_summary_key,
        "prompt_is_second_retrieval": prompt_is_second_retrieval
    }
prompt_is_homepage = dedent("""
        You are now a professional search result evaluator. Your task is to determine whether the search recall is correct (whether the recalled document segments can answer the question) based on the document segments recalled by the search, combined with the question, and return the source of the answer.
        Note:
        1.All document segments come from the first page of the documents, and you need to combine all the information from the first pages to determine whether the question can be answered.
        2.Determine whether the question posed can be answered correctly. Do not use your own knowledge, only based on the content of the document segments! Do not provide the answer! Only answer “yes” or “no”!
        3.The source of the answer can only be the recalled document segments. When the question cannot be answered, the source of the answer can be left blank.
        ========================================
        Context:
        {% for document in documents %}
        Document {{loop.index}}:
        {{document[:1000]}}
        ---
        {% endfor %}
        ========================================
        Question: {{query}}
        Answer:yes/no
        Source:[] or [Document1:"",Document2:""](The number of Document depends on the actual situation, The document and its content must be provided.)
        """)

prompt_is_relevant = dedent("""
        You are now a professional search result evaluator. Your task is to individually evaluate whether each recalled document segment is semantically relevant to the question (whether the document segment could be part of or the complete answer to the question).
        Note:
        1. Evaluate each document segment independently. Do not combine information across documents.
        2. Determine relevance based solely on the content of each document segment. Do not use external knowledge.
        3. For each document, provide only a "yes" or "no" judgment.
        4. No need to specify sources or provide excerpts for relevant documents.
        5. Apply lenient similarity matching; When uncertain about relevance, give yes; only give no when clearly unrelated
        ========================================
        Context:
        {% for document in documents %}
        Document {{loop.index}}:
        {{document.meta['metadata']['file_name']+":"+document.meta['metadata']['link_summary'][:1000]}}
        ---
        {% endfor %}
        ========================================
        Question: {{query}}

        Evaluation:
        {% for document in documents %}
        Document {{loop.index}}: [yes/no]
        {% endfor %}
        """)