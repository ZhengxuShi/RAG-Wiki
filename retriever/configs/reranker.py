from hypster import HP
from retriever.src.haystack_utils import FastAPIRanker, OpenAIReranker
from haystack.components.rankers import TransformersSimilarityRanker
def reranker(hp: HP):

    transformers_models = {
        "tiny-bert-v2": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "minilm-v2": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        "Cohere": "rerank-multilingual-v3.0",
        "bge-reranker": "./retriever/models/bge-reranker-v2-m3",
        "openai-bge-reranker": "bge-reranker-v2-m3",
        "openai-bge-reranker-int4": "bge-reranker-v2-m3-int4-ov",
        "bge-reranker-finetune": "./retriever/models/bge-rerank-batch4-en-neg",
        "gte_passage-ranking_multilingual-base": "./retriever/models/iic/gte_passage-ranking_multilingual-base"                                     
        ""
    }
    model = hp.select({**transformers_models}, default="bge-reranker", name="model")
    top_k = hp.int(default=30, name="top_k")
    rerank_url = hp.text(default="None", name="rerank_url")
    rerank_model_platform = hp.text(default="openai", name="rerank_model_platform")
    timeout = hp.int(default=300, name="timeout")
    default_config_path = hp.text(default="", name="default_config_path")
    if rerank_model_platform == "openai":
        reranker = OpenAIReranker(
            base_url=rerank_url,
            model=model,
            top_k=top_k,
            timeout=timeout,
            default_config_path=default_config_path
        )
    elif rerank_model_platform == "local_api":
        reranker = FastAPIRanker(url=rerank_url, top_k=top_k)
    elif rerank_model_platform == "local_st":
        reranker = TransformersSimilarityRanker(model=model, top_k=top_k,batch_size=1)
    else:
        # 提供清晰的错误信息和可用的选项
        available_platforms = ["openai", "local_api", "local_st"]
        raise ValueError(
            f"Unsupported rerank_model_platform: '{rerank_model_platform}'. "
            f"Available options: {available_platforms}"
        )
    return {
        "model": model,
        "reranker": reranker,
        "rerank_model_platform": rerank_model_platform
    }


