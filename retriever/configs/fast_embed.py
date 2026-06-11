from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from hypster import HP
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.embedders import OpenAITextEmbedder
from retriever.src.haystack_utils import  OpenAISparseTextEmbedder,OpenAISparseDocumentEmbedder
def fast_embed(hp: HP):
    model = hp.select(
        {"bge-m3:latest": "bge-m3:latest", "bge-m3": "bge-m3","gte-finetune-api": "bge-negative-train",
         "gte": "retriever/models/gte_sentence-embedding_multilingual-base",
         "gte-qwen2-1.5b": "retriever/models/gte_Qwen2-1.5B-instruct",
         "qwen3": "retriever/models/Qwen3-Embedding-8B",
         "bge-m3_local": "retriever/models/bge-m3",
            },
        default="bge-m3",
        name="model"
    )
    sparse_model = hp.select(
        {"bm42": "Qdrant/bm42-all-minilm-l6-v2-attentions","bm42_openai": "all-minilm-l6-v2","all_miniLM_L6_v2_with_attentions": "all_miniLM_L6_v2_with_attentions"},
        default="all_miniLM_L6_v2_with_attentions",name="sparse_model"
    )
    from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
    model_url = hp.text(default="None",name="model_url")
    sparse_embedder_model_url = hp.text(default="None", name="sparse_embedder_model_url")
    embedder_timeout = hp.int(default=300,name="embedder_timeout")
    sparse_timeout = hp.int(default=300,name="sparse_timeout")
    embedder_model_platform = hp.text(default="openai",name="embedder_model_platform")
    sparse_embedder_model_platform = hp.text(default="openai", name="sparse_embedder_model_platform")
    if embedder_model_platform == "openai":
        text_embedder = OpenAITextEmbedder(model=model, api_base_url=model_url, timeout=embedder_timeout)
    elif embedder_model_platform == "ollama":
        from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
        from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
        text_embedder = OllamaTextEmbedder(model=model, url=model_url, timeout=embedder_timeout)

    elif embedder_model_platform == "local_st":
        from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
        from retriever.src.haystack_utils import SentenceTransformersTextEmbedderClient
        # text_embedder = SentenceTransformersTextEmbedder(model=model, trust_remote_code=True)
        text_embedder = SentenceTransformersTextEmbedderClient(api_url=model_url)
    else:
        # 提供清晰的错误信息和可用的选项
        available_platforms = ["openai", "ollama", "local_st"]
        raise ValueError(
            f"Unsupported embedder_model_platform: '{embedder_model_platform}'. "
            f"Available options: {available_platforms}"
        )
    # 稀疏模型
    if sparse_embedder_model_platform == "openai":
        # 自定义远程稀疏模型
        sparse_text_embedder = OpenAISparseTextEmbedder(model=sparse_model, api_base_url=sparse_embedder_model_url,
                                                        timeout=sparse_timeout)
    elif sparse_embedder_model_platform == "local_st":
        # 本地稀疏模型
        sparse_text_embedder = FastembedSparseTextEmbedder(model=sparse_model, local_files_only=True)
    else:
        # 提供清晰的错误信息和可用的选项
        available_platforms = ["openai", "local_st"]
        raise ValueError(
            f"Unsupported sparse_embedder_model_platform: '{sparse_embedder_model_platform}'. "
            f"Available options: {available_platforms}"
        )
    return {
        "sparse_model": sparse_model,
        "model_url": model_url,
        "model": model,
        "embedder_model_platform": embedder_model_platform,
        "sparse_embedder_model_platform": sparse_embedder_model_platform,
        "text_embedder": text_embedder,
        "sparse_text_embedder": sparse_text_embedder,
    }
