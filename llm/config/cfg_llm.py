from hypster import HP


def llm_config(hp: HP):
    ollama_models = {"qwen2:72b":"qwen2:72b"}
    openai_models = {"gpt-4o-mini": "gpt-4o-mini", "gpt-4o": "gpt-4o", "gpt-4o-latest": "gpt-4o-2024-08-06"}
    model_options = {**ollama_models, **openai_models}

    model = hp.select(model_options, default="qwen2:72b")
    temperature = hp.number(0.0, min=0.0, max=1.0)
    num_predict = hp.number(500, min=1, max=1000)

    if model in openai_models.values():
        from haystack.components.generators import OpenAIGenerator
        llm = OpenAIGenerator(model=model, generation_kwargs={"temperature": temperature})
    elif model in ollama_models.values():
        from haystack_integrations.components.generators.ollama.generator import OllamaGenerator
        llm = OllamaGenerator(model=model, url = "http://127.0.0.1:11434",generation_kwargs={"temperature": temperature,"num_predict": num_predict,})
    else:
        raise ValueError(f"Model {model} not supported")