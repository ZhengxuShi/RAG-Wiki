from hypster import HP


def indexing_config(hp: HP):
    from haystack import Pipeline

    from haystack.components.converters import JSONConverter
    from retriever.src.haystack_utils import Enrich_Doc_LLM, AddLLMMetadata, Updated_Meta_Info, Data_To_Written
    from haystack.components.writers import DocumentWriter
    from haystack.document_stores.types import DuplicatePolicy
    from milvus_haystack import MilvusDocumentStore

    pipeline = Pipeline()
    pipeline.add_component("loader",
                           JSONConverter(content_key="text", extra_meta_fields={"page_idx", "img_path", "unique_id",
                                                                                "next_text_level", "next_text_level",
                                                                                "previous_text_level",
                                                                                "parallel_text_level",
                                                                                "file_name", "type", "text_level",
                                                                                "table_caption", "table_footnote",
                                                                                "sheet", "version", "url", "ori_id_1"}))

    indexing_method = hp.select(["enrich_doc_key", "enrich_doc_summary", "enrich_doc_summary_key", "none"],
                                default="none", name="indexing_method")

    embedder_url = hp.text(default="None", name="embedder_url")
    sparse_embedder_model_url = hp.text(default="None", name="sparse_embedder_model_url")
    embedder_timeout = hp.int(default=300, name="embedder_timeout")
    sparse_timeout = hp.int(default=300, name="sparse_timeout")
    retrieval_fetchType = hp.select(["local", "api"], default="local", name="retrieval_fetchType")

    # embedder
    embedder_model = hp.select(
        {"bge-m3:latest": "bge-m3:latest", "bge-m3": "bge-m3", "gte-finetune-api": "bge-negative-train",
         "gte": "retriever/models/gte_sentence-embedding_multilingual-base",
         "gte-qwen2-1.5b": "retriever/models/gte_Qwen2-1.5B-instruct",
         "qwen3": "retriever/models/Qwen3-Embedding-8B",
         "gte_1.5b": "retriever/models/gte-Qwen2-1.5B-instruct",
         "bge-m3_local": "retriever/models/bge-m3"},
        default="bge-m3", name="embedder_model"
    )

    embedder_sparse_model = hp.select(
        {"bm42": "Qdrant/bm42-all-minilm-l6-v2-attentions", "bm42_openai": "all-minilm-l6-v2",
         "all_miniLM_L6_v2_with_attentions": "all_miniLM_L6_v2_with_attentions"},
        default="bm42", name="embedder_sparse_model"
    )

    embedder_model_platform = hp.text(default="openai", name="embedder_model_platform")
    sparse_embedder_model_platform = hp.text(default="openai", name="sparse_embedder_model_platform")

    if embedder_model_platform == "openai":
        from haystack.components.embedders import OpenAIDocumentEmbedder
        doc_embedder = OpenAIDocumentEmbedder(model=embedder_model, api_base_url=embedder_url, timeout=embedder_timeout)
    elif embedder_model_platform == "ollama":
        from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
        doc_embedder = OllamaDocumentEmbedder(model=embedder_model, url=embedder_url, timeout=embedder_timeout)
    elif embedder_model_platform == "local_st":
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder
        from retriever.src.haystack_utils import SentenceTransformersEmbedderClient
        # doc_embedder = SentenceTransformersDocumentEmbedder(model=embedder_model, trust_remote_code=True, batch_size=4)
        doc_embedder = SentenceTransformersEmbedderClient(api_url=embedder_url)
    else:
        # 提供清晰的错误信息和可用的选项
        available_platforms = ["openai", "ollama", "local_st"]
        raise ValueError(
            f"Unsupported embedder_model_platform: '{embedder_model_platform}'. "
            f"Available options: {available_platforms}"
        )

    from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
    from retriever.src.haystack_utils import OpenAISparseDocumentEmbedder

    # 稀疏模型
    if sparse_embedder_model_platform == "openai":
        # 自定义远程稀疏模型
        sparse_doc_embedder = OpenAISparseDocumentEmbedder(model=embedder_sparse_model,
                                                           api_base_url=sparse_embedder_model_url,
                                                           timeout=sparse_timeout)
    elif sparse_embedder_model_platform == "local_st":
        # 本地稀疏模型
        sparse_doc_embedder = FastembedSparseDocumentEmbedder(model=embedder_sparse_model, local_files_only=True)
    else:
        # 提供清晰的错误信息和可用的选项
        available_platforms = ["openai", "local_st"]
        raise ValueError(
            f"Unsupported sparse_embedder_model_platform: '{sparse_embedder_model_platform}'. "
            f"Available options: {available_platforms}"
        )

    db_name = hp.text(default="milvus", name="db_name")
    db_path_url = hp.text(default="None", name="db_path_url")
    milvus_db_name = hp.text(default="authenticated_agents_second", name="milvus_db_name")

    # 大模型
    anthropic_models = {"qwen2:72b": "qwen2:72b",
                        "deepseek-r1:32b": "deepseek-r1:32b",
                        "deepseek-r1:32b-qwen-distill-q8_0": "deepseek-r1:32b-qwen-distill-q8_0"}
    openai_models = {"qwen2.5": "qwen2.5"}
    model_options = {**anthropic_models, **openai_models}

    model_platform = hp.text(default="openai", name="model_platform")
    model = hp.select(model_options, default="qwen2:72b", name="model")
    llm_enrich_model_url = hp.text(default="None", name="llm_enrich_model_url")
    temperature = hp.int(default=0, name="temperature")
    num_predict = hp.int(default=400, name="num_predict")
    timeout = hp.int(default=300, name="timeout")
    if model_platform == "openai":
        from haystack.components.generators import OpenAIGenerator
        llm_enrich = OpenAIGenerator(model=model, api_base_url=llm_enrich_model_url,
                                     generation_kwargs={"temperature": temperature, "max_tokens": num_predict,
                                                        "timeout": timeout})
    elif model_platform == "ollama":
        from haystack_integrations.components.generators.ollama.generator import OllamaGenerator
        llm_enrich = OllamaGenerator(model=model, url=llm_enrich_model_url,
                                     generation_kwargs={"temperature": temperature, "num_predict": num_predict,
                                                        "timeout": timeout})
    else:
        # 提供清晰的错误信息和可用的选项
        available_platforms = ["openai", "ollama"]
        raise ValueError(
            f"Unsupported model_platform: '{model_platform}'. "
            f"Available options: {available_platforms}"
        )

    # 只有当 retrieval_fetchType 为 "local" 时才初始化 document_store
    document_store = None
    if retrieval_fetchType == "local":
        document_store = MilvusDocumentStore(
            connection_args={"uri": db_path_url, "db_name": milvus_db_name},
            collection_name=db_name,
            vector_field="embedding",
            primary_field="doc_id",
            text_field="content",
            index_params={
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {"M": 24, "efConstruction": 100, "ef": 50},
            },
            sparse_vector_field="sparse_embedding",
            sparse_index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
                "params": {"M": 24, "efConstruction": 100, "ef": 50},
            },
            search_params={"metric_type": "L2", "params": {"ef": 50}},
            drop_old=False,
        )

    pipeline.add_component("updated_meta_info", Updated_Meta_Info())
    pipeline.connect("loader", "updated_meta_info")

    if indexing_method == "none":
        splitter_source = "updated_meta_info"
    else:
        pipeline.add_component("enrich_doc_llm", Enrich_Doc_LLM(hp, indexing_method, llm_enrich, db_name))
        pipeline.add_component("document_enricher", AddLLMMetadata())

        pipeline.connect("updated_meta_info", "enrich_doc_llm")
        pipeline.connect("enrich_doc_llm", "document_enricher")
        pipeline.connect("updated_meta_info", "document_enricher")

        splitter_source = "document_enricher"

    # 默认都是以混合检索方式构建知识库
    pipeline.add_component("doc_embedder", doc_embedder)
    pipeline.add_component("sparse_doc_embedder", sparse_doc_embedder)

    pipeline.connect(splitter_source, "sparse_doc_embedder")
    pipeline.connect("sparse_doc_embedder", "doc_embedder")

    # 统一管道构建逻辑：根据 retrieval_fetchType 决定最终的写入组件
    if retrieval_fetchType == "local":
        # 本地测试：使用 DocumentWriter 直接写入 document_store
        pipeline.add_component("document_writer", DocumentWriter(document_store))
        pipeline.connect("doc_embedder", "document_writer")
    else:
        # API 调用：使用 Data_To_Written
        pipeline.add_component("Data_To_Written", Data_To_Written())
        pipeline.connect("doc_embedder", "Data_To_Written")

    # 构建返回结果
    result = {
        "pipeline": pipeline,
        "doc_embedder": doc_embedder,
        "model": model,
        "embedder_model_platform": embedder_model_platform,
        "sparse_doc_embedder": sparse_doc_embedder,
        "llm_enrich": llm_enrich,
    }

    # 只有当 retrieval_fetchType 为 "local" 时才返回 document_store
    if retrieval_fetchType == "local":
        result["document_store"] = document_store

    return result