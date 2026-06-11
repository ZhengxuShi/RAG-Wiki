import json
import requests
from milvus_haystack.milvus_embedding_retriever import MilvusHybridRetriever
from typing import Optional, Dict, Any, List
from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict
from milvus_haystack import MilvusDocumentStore
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from pymilvus.client.abstract import BaseRanker
from pymilvus import AnnSearchRequest, RRFRanker
from milvus_haystack.document_store import  MilvusStoreError
from retriever.src.utils import MilvusQuery_hybrid_retriever
import logger_local
from logger_local import app_logger as logging
from retriever.retriever import LoggedTime
class MilvusFilterHybridRetriever(MilvusHybridRetriever):

    """
    A component for retrieving documents using hybrid search with optional filtering from a Milvus Document Store.
    """

    def __init__(self, document_store: MilvusDocumentStore, top_k: int = 10,
                 reranker: Optional[BaseRanker] = None, filename=[],retrieval_fetchType= None):
        super().__init__(document_store)
        self.top_k = top_k
        self.filename = filename
        self.document_store = document_store
        self.retrieval_fetchType = retrieval_fetchType
        if reranker is None:
            reranker = RRFRanker()
        self.reranker = reranker
        self.knowledgeUUID=""
        self._vector_field = "embedding"
        self._sparse_vector_field = "sparse_embedding"
    def close_search(self):
        self.document_store.client.close()
    def split_knowledge_uuid(self,knowledge_uuid_str):
        """
        按照英文逗号切分 knowledgeUUID 字符串

        Args:
            knowledge_uuid_str: 包含 UUID 的字符串，用逗号分隔

        Returns:
            list: 切分后的 UUID 列表
        """
        if not knowledge_uuid_str:
            return []

        # 去除可能的空格后按逗号切分
        return [uuid.strip() for uuid in knowledge_uuid_str.split(',') if uuid.strip()]

    def _local_retrieval(self, query_embedding: List[float], query_sparse_embedding: SparseEmbedding, filename=[], knowledgeUUID=None, logging=None) -> \
    List[Document]:
        """本地向量库检索"""
        self.filename = filename
        self.knowledgeUUID = knowledgeUUID

        # 构建过滤条件
        filter_conditions = []

        # 添加文件名过滤
        if self.filename:
            file_name_conditions = [
                {
                    "field": "metadata['file_name']",
                    "operator": "==",
                    "value": fname
                }
                for fname in self.filename
            ]
            filter_conditions.append({
                "operator": "OR",
                "conditions": file_name_conditions
            })

        # 添加knowledgeUUID过滤
        if self.knowledgeUUID:
            knowledgeUUID_list = self.split_knowledge_uuid(knowledge_uuid_str=self.knowledgeUUID)
            uuid_conditions = [
                {
                    "field": "metadata['knowledge_uuid']",
                    "operator": "==",
                    "value": uuid
                }
                for uuid in knowledgeUUID_list
            ]
            filter_conditions.append({
                "operator": "OR",
                "conditions": uuid_conditions
            })
            logging.info(f"进行本地混合检索,筛选向量库为{self.knowledgeUUID}文件名为{self.filename}的chunks")
        else:
            logging.info(f"进行本地混合检索,筛选文件名为{self.filename}的chunks")

        # 组合所有过滤条件
        filters_files = {}
        if filter_conditions:
            if len(filter_conditions) == 1:
                filters_files = filter_conditions[0]
            else:
                filters_files = {
                    "operator": "AND",
                    "conditions": filter_conditions
                }

        if filters_files:
            logging.info(f"应用过滤条件: {filters_files}")

        with LoggedTime("本地检索接口调用耗时统计", logging):
            docs = self.document_store._hybrid_retrieval(
                query_embedding=query_embedding,
                query_sparse_embedding=query_sparse_embedding,
                top_k=self.top_k,
                filters=filters_files,
                reranker=self.reranker,
            )

        self.close_search()
        return {"documents": docs}

    def _api_retrieval(self, query_embedding: List[float], query_sparse_embedding: SparseEmbedding,
                       filename=[], knowledgeUUID=None, query_hybrid_handler=None, parseType=None) -> List[Document]:
        """API接口检索"""
        self.filename= filename
        self.knowledgeUUID = knowledgeUUID
        self.query_hybrid_handler = query_hybrid_handler

        if not self.knowledgeUUID:
            self.knowledgeUUID=""
            logging.warning("No knowledge UUID,检索结果返回空!")
            docs=[]
        else:
            knowledgeUUID_list=self.split_knowledge_uuid(knowledge_uuid_str=self.knowledgeUUID)
            logging.info(f"进行API混合检索,筛选向量库为{knowledgeUUID}文件名为{self.filename}的chunks")

            limit = self.top_k
            result_limit = self.top_k
            # 构造数据
            search_queries = self.query_hybrid_handler.processed_data_for_hybrid_retrieval(
                query_embedding, query_sparse_embedding, limit, knowledgeUUID_list
            )

            with LoggedTime("混合检索接口调用耗时统计", logging):
                docs = self.query_hybrid_handler.call_milvus_hybrid_retrieval(search_queries, result_limit, knowledgeUUID=knowledgeUUID, parseType=parseType)

        return {"documents": docs}

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], query_sparse_embedding: SparseEmbedding,
        filename=[], knowledgeUUID=None, query_hybrid_handler=None,logging=None, parseType=None):
        """
        Retrieve documents from the `MilvusDocumentStore`, based on their dense and sparse embeddings and a filter.

        :param query_embedding: Dense Embedding of the query.
        :param query_sparse_embedding: Sparse Embedding of the query.
        :param filename: 文件名过滤列表
        :param knowledgeUUID: 知识库UUID
        :param query_hybrid_handler: 混合检索处理器
        :param logging: 日志记录器
        :param parseType: 解析类型参数
        :param flag: "local" for local retrieval, "api" for API retrieval.
        :param use_filter: Boolean to determine whether to use filtering in the search.
        :return: List of Document similar to `query_embedding`.
        """
        if not filename:
            filename = []
        if logging is None:
            logging = logger_local.app_logger

        if self.retrieval_fetchType == "local":
            return self._local_retrieval(query_embedding, query_sparse_embedding, filename, knowledgeUUID, logging)
        elif self.retrieval_fetchType == "api":
            return self._api_retrieval(query_embedding, query_sparse_embedding, filename, knowledgeUUID,
                                       query_hybrid_handler, parseType)
        else:
            raise ValueError(f"Unsupported retrieval_fetchType: {self.retrieval_fetchType}. Use 'local' or 'api'.")