from hypster import HP


def llm_config(hp: HP):
    anthropic_models = {"haiku": "claude-3-haiku-20240307", "sonnet": "claude-3-5-sonnet-20240620",
                        "qwen2:72b": "qwen2:72b", "qwen2.5:latest": "qwen2.5:latest",
                        "deepseek-r1:32b": "deepseek-r1:32b",
                        "deepseek-r1:32b-qwen-distill-q8_0": "deepseek-r1:32b-qwen-distill-q8_0"}
    openai_models = {"gpt-4o-mini": "gpt-4o-mini", "gpt-4o": "gpt-4o", "gpt-4o-latest": "gpt-4o-2024-08-06"}
    model_options = {**anthropic_models, **openai_models}

    model = hp.select(model_options, default="qwen2:72b",name="model")
    temperature = hp.int(0,name="temperature")
    if model in openai_models.values():
        from haystack.components.generators import OpenAIGenerator

        llm = OpenAIGenerator(model=model, generation_kwargs={"temperature": temperature})
    else:
        from haystack_integrations.components.generators.ollama.generator import OllamaGenerator
        llm = OllamaGenerator(model=model, url="http://127.0.0.1:11434",
                              generation_kwargs={"temperature": temperature, "num_predict": 200,"timeout": 300})
    return {
            "llm": llm,
            "model": model
        }
