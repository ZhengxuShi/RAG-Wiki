from haystack import Pipeline

from llm.llm import LlmModule
from retriever.retriever import RetrieverModule, Retriever_Recall_Module, Knowledge_Base


class rag_module:
    def __init__(self):
        self.retriever_module = RetrieverModule()
        self.llm_module = LlmModule()

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="retriever", instance=self.retriever_module)
        self.pipeline.add_component(name="llm", instance=self.llm_module)
        self.pipeline.connect(sender="retriever", receiver="llm.documents")

    def run(self, query: str, retrieval_config: dict):
        result = self.pipeline.run(
            {
                "retriever": {"query": query, "retrieval_config": retrieval_config},
                "llm": {"query": query}
            }
        )
        return result


if __name__ == "__main__":
    retrieval_config = {
        "indexing_method": "enrich_doc_summary_key",
        "retrieval_embedding_similarity_function": "cosine",

        "embedder_model": "bge-m3",
        "embedder_sparse_model": "bm42",
        "embedder_embedding_dim": 1024,

        "query_methods": "none",

        "retrieval_top_k": 10,
        "retrieval_method": "hybrid_retriever",

        "use_reranker": True,
        "reranker_model": "bge-reranker",
        "reranker_top_k": 5,

        "llm_model": "qwen2:72b",
        "llm_temperature": 0,

        "use_second_retrieval": False,
        "second_retrieval_retrieval": "General",
        "second_indexing_method": "none",
        "second_query_methods": "query_hyde",
        "second_retrieval_method": "embedding_retriever",

    }
    rag = rag_module()
    result = rag.run(query="笔记本对塑胶外壳材料有什么要求？", retrieval_config=retrieval_config)
    print(result)

