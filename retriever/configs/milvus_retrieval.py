from hypster import HP
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever, MilvusSparseEmbeddingRetriever, \
    MilvusHybridRetriever
from retriever.configs.filter_and_hybrid_search import MilvusFilterHybridRetriever


def milvus_retrieval(hp: HP):
    retrieval_method = hp.select(
        ["embedding_retriever", "sparse_embedding_retriever", "hybrid_retriever", "sparse_retriever"],
        default="embedding_retriever", name="retrieval_method")
    retrieval_fetchType = hp.select(
        ["local", "api"], default="local", name="retrieval_fetchType")
    collection_name = hp.text(default="milvus", name="collection_name")
    db_path_url = hp.text(default="None", name="db_path_url")
    milvus_db_name = hp.text(default="default", name="milvus_db_name")
    milvus_filename = hp.text(default="file_name", name="milvus_filename")
    multiRouteRetrieval = hp.bool(default=False, name="multiRouteRetrieval")

    # 参数设置
    embedding_retriever_top_k = hp.int(default=10, name="embedding_retriever_top_k")
    retrieval_sparse_embedding_retriever_top_k = hp.int(default=10, name="retrieval_sparse_embedding_retriever_top_k")
    retrieval_hybrid_retriever_top_k = hp.int(default=10, name="retrieval_hybrid_retriever_top_k")

    # 只有当 retrieval_fetchType 为 "local" 时才初始化 document_store
    document_store = None
    embedding_retriever = None

    if retrieval_fetchType == "local":
        # 知识库 - 仅本地检索时初始化
        document_store = MilvusDocumentStore(
            connection_args={"uri": db_path_url, "db_name": milvus_db_name},
            # Milvus standalone docker service.
            collection_name=collection_name,
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
            search_params={"metric_type": "L2", "params": {"ef": 150}},
            drop_old=False,
        )

        # 根据检索方法初始化对应的检索器（仅本地模式）
        if retrieval_method == "embedding_retriever":
            embedding_retriever = MilvusEmbeddingRetriever(
                document_store=document_store, top_k=embedding_retriever_top_k)
        elif retrieval_method == "sparse_embedding_retriever":
            embedding_retriever = MilvusSparseEmbeddingRetriever(
                document_store=document_store, top_k=retrieval_sparse_embedding_retriever_top_k)
        elif retrieval_method == "hybrid_retriever":
            embedding_retriever = MilvusFilterHybridRetriever(
                document_store=document_store, top_k=retrieval_hybrid_retriever_top_k,
                filename=milvus_filename, retrieval_fetchType=retrieval_fetchType)

    elif retrieval_fetchType == "api":
        # API 模式下，只初始化混合检索器
        if retrieval_method == "hybrid_retriever":
            embedding_retriever = MilvusFilterHybridRetriever(
                document_store=None, top_k=retrieval_hybrid_retriever_top_k,
                filename=milvus_filename, retrieval_fetchType=retrieval_fetchType)
        else:
            # 其他检索方法在 API 模式下可能不支持，或者需要不同的实现
            raise ValueError(f"Retrieval method '{retrieval_method}' is not supported in API mode. "
                             f"Only 'hybrid_retriever' is supported for API calls.")

    # 构建返回结果
    result = {
        "retrieval_method": retrieval_method,
        "retrieval_fetchType": retrieval_fetchType,
        "retrieval_hybrid_retriever_top_k": retrieval_hybrid_retriever_top_k,
        "multiRouteRetrieval": multiRouteRetrieval
    }

    # 只有当 retrieval_fetchType 为 "local" 时才返回 document_store
    if retrieval_fetchType == "local":
        result["document_store"] = document_store
        result["embedding_retriever"] = embedding_retriever
    else:
        # API 模式下返回 embedding_retriever（如果已初始化）
        if embedding_retriever:
            result["embedding_retriever"] = embedding_retriever

    return result