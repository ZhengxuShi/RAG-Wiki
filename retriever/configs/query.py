from hypster import HP

from retriever.src.haystack_utils import PassThroughText, Word_Filter, Query2query
from haystack_integrations.components.generators.ollama.generator import OllamaGenerator


def query(hp: HP):
    from haystack import Pipeline

    query_methods = hp.select(["none", "query_hyde", "query_language", "query_association",
                               "query_association+query_hyde", "query_association+query_language",
                               "query_language+query_hyde", "query_language+query_association",
                               "query_language+query_association+query_hyde",
                               "query_association+query_language+query_hyde"],
                              default="none", name="query_methods")

    anthropic_models = {"qwen2:72b": "qwen2:72b",
                        "deepseek-r1:32b": "deepseek-r1:32b",
                        "deepseek-r1:32b-qwen-distill-q8_0": "deepseek-r1:32b-qwen-distill-q8_0"}
    openai_models = {"qwen2.5": "qwen2.5"}
    model_options = {**anthropic_models, **openai_models}
    model_platform = hp.text(default="openai", name="model_platform")
    model = hp.select(model_options, default="qwen2:72b", name="model")
    temperature = hp.int(default=0, name="temperature")
    num_predict = hp.int(default=200, name="num_predict")
    timeout = hp.int(default=300, name="timeout")
    model_url = hp.text(default="None", name="model_url")

    if model_platform == "openai":
        from haystack.components.generators import OpenAIGenerator

        llm_query = OpenAIGenerator(model=model, api_base_url=model_url,
                                    generation_kwargs={"temperature": temperature, "max_tokens": num_predict,
                                                       "timeout": timeout})
    elif model_platform == "ollama":
        from haystack_integrations.components.generators.ollama.generator import OllamaGenerator

        llm_query = OllamaGenerator(model=model, url=model_url,
                                    generation_kwargs={"temperature": temperature, "num_predict": num_predict,
                                                       "timeout": timeout})
    pipeline = Pipeline()
    pipeline.add_component("query", PassThroughText())
    pipeline.add_component("word_filter", Word_Filter())
    pipeline.add_component("query2query", Query2query(hp=hp, query_methods=query_methods, llm_query=llm_query))
    pipeline.connect("query", "word_filter")
    pipeline.connect("word_filter", "query2query")
    return {
        "pipeline": pipeline,
        "llm_query": llm_query,
        "model": model,
        "query_methods": query_methods,
        "anthropic_models": anthropic_models,
        "model_platform": model_platform,
        "temperature": temperature,
        "num_predict": num_predict,
        "timeout": timeout,
        "model_url": model_url
    }
