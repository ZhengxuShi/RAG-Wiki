from hypster import HP


def prompt_config(hp: HP):
    from textwrap import dedent

    template_1 = dedent("""\
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

    template_2 = dedent("""\
    基于下面内容，回答问题，简洁明了，用一到两句话回答，注意，用中文回答，不要扩展。
    ========================================
    内容:
    {% for document in documents %}
    Document {{loop.index}}:
    {{ document.meta.llm_extracted_info }}
    {{ document.content }}
    ---
    {% endfor %}
    ========================================
    问题: {{query}}

    回答:
    """)
    
    template_options = {
        "template_1": template_1,
        "template_2": template_2,
    }

    template = hp.select(template_options, default="template_2")

    from haystack.components.builders.prompt_builder import PromptBuilder
    prompt_builder = PromptBuilder(template=template)