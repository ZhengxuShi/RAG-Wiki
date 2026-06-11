import json
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from milvus_haystack import MilvusDocumentStore
from haystack import Document
from typing import Any, Dict, List
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from logger_local import app_logger as logging

class MilvusQueryHandler:
    def __init__(self, uri, db_name, collection_name, logging):
        """
        初始化Milvus查询处理器

        Args:
            uri: Milvus连接URI
            db_name: 数据库名称
            collection_name: 集合名称
            logging: 日志对象
        """
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.logging = logging

        # 初始化时创建DocumentStore连接
        self.document_store = MilvusDocumentStore(
            connection_args={"uri": self.uri, "db_name": self.db_name},
            collection_name=self.collection_name,
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
            },
            search_params={"metric_type": "L2", "params": {"M": 24, "efConstruction": 100, "ef": 50}},
            drop_old=False,
        )
        self.logging.info(f"Milvus查询处理器初始化完成，连接至集合: {self.collection_name}")

    def query_documents(self, filters):
        """
        查询符合过滤条件的文档

        Args:
            filters: 过滤条件字典，格式为 {"field": "...", "operator": "...", "value": ...}

        Returns:
            list: 包含符合条件文档内容的列表
        """
        # self.logging.info(f"开始检索文档，过滤条件: {filters}")

        try:
            # 从 fields 中移除 text_bm25_embedding，避免检索时访问该字段
            if 'text_bm25_embedding' in self.document_store.fields:
                original_fields = self.document_store.fields[:]
                self.document_store.fields = [f for f in original_fields if f != 'text_bm25_embedding']
                self.logging.info(f"已从检索字段中移除 text_bm25_embedding，剩余字段: {self.document_store.fields}")

            try:
                # self.logging.info(f"执行过滤条件: {filters}")
                filter_documents = self.document_store.filter_documents(filters=filters)
                return filter_documents
            finally:
                # 恢复原始 fields 列表
                if 'original_fields' in locals():
                    self.document_store.fields = original_fields

        except Exception as e:
            self.logging.error(f"查询文档时发生错误: {str(e)}")
            raise
    def is_filename_empty(self,original_name) -> bool:
        """
        判断集合是否存在

        Returns:
            bool: 如果集合不存在返回True，否则返回False
        """
        self.logging.info("开始判断文件名是否存在")
        filename_docs = self.query_documents(
            filters={"field": "metadata['file_name']", "operator": "==", "value": original_name}
        )
        if filename_docs :
            return False
        else:
            return True

    def is_collection_empty(self) -> bool:
        """
        判断集合是否存在

        Returns:
            bool: 如果集合不存在返回True，否则返回False
        """
        self.logging.info("开始判断collection是否存在")
        if self.document_store.col is None:
            return True
        else:
            return False

class MilvusQuery_filter_retriever:
    def __init__(self,filter_retriever_url):
        self.current_data_img_paths = []
        self.expr = ""
        self._vector_field = "embedding"
        self._sparse_vector_field = "sparse_embedding"
        self._primary_field = "doc_id"
        self._text_field = "content"
        self._dummy_value = 999.0
        # logging = logging
        self.filter_retriever_url = filter_retriever_url
    def _extract_fields(self,data) -> None:
        """Grab the existing fields from the Collection"""
        if isinstance(data, list):
            if len(data) == 0:
                 logging.info(f"filter_retriever检索的内容为空！")
            else:
                self.fields = list(data[0].keys())
    def _parse_document(self, data: dict) -> Document:
        # we store dummy vectors during writing documents if they are not provided,
        # so we don't return them if they are dummy vectors
        embedding = data.pop(self._vector_field)
        if all(x == self._dummy_value for x in embedding):
            embedding = None

        sparse_embedding = None
        sparse_dict = data.pop(self._sparse_vector_field, None)
        if sparse_dict:
            sparse_embedding = self._convert_dict_to_sparse(sparse_dict)
            if sparse_embedding.values == [self._dummy_value] and sparse_embedding.indices == [0]:
                sparse_embedding = None
        # nas修改
        data["metadata"] = json.loads(data["metadata"])  #nas要注释
        return Document(
            id=data.pop(self._primary_field),
            content=data.pop(self._text_field),
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            meta=data,
        )
    def _convert_dict_to_sparse(self, sparse_dict: Dict) -> SparseEmbedding:
        return SparseEmbedding(indices=list(sparse_dict.keys()), values=list(sparse_dict.values()))
    def call_milvus_filtered_retrieval(self, expr, output_fields=None, limit=None,knowledgeUUID_list:list=[],parse_Type=None):
        knowledgeUUID = ",".join(knowledgeUUID_list)
        # 接口URL
        if output_fields is None:
            output_fields = ["*"]
        # 请求头
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        # 请求体
        payload = {
            "filter": expr,
            "outputFields": output_fields,
            "knowledgeUUID": knowledgeUUID,
            "parse_Type": parse_Type
        }
        # nas修改
        if limit is None:
            payload["limit"] = 50
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                logging.warning("警告: 'limit' 必须是一个正整数，将忽略此参数。")
            else:
                payload["limit"] = limit
        try:
            # 发送POST请求
            response = requests.post(self.filter_retriever_url, headers=headers, data=json.dumps(payload), verify=False)

            # 检查响应状态码
            if response.status_code == 200:
                result = response.json()
                if result.get("code") != 200:
                    logging.error(f"混合检索接口请求失败，错误代码: {result.get('code')}")
                    logging.error(f"错误信息: {result.get('msg', '无')}")
                    return []
                # # nas修改-nas要将以上注释，用下面注释的
                # if result.get("code") != 0:
                #     logging.error(f"混合检索接口请求失败，错误代码: {result.get('code')}")
                #     logging.error(f"错误信息: {result.get('msg', '无')}")
                #     return []

                # # nas修改
                result=result["data"] #nas要注释
                # result=result["data"]
                # 检查返回代码是否为0（成功）
                if result.get("code") == 0:
                    docs = []
                    self._extract_fields(result["data"])
                    output_fields=self.fields
                    for item in result.get("data", []):
                        # 解析文档数据
                        data = {field: item.get(field) for field in output_fields}
                        doc = self._parse_document(data)
                        docs.append(doc)
                    return docs

                else:
                    logging.error(f"过滤检索请求接口失败，错误代码: {result.get('code')}")
                    return []
            else:
                logging.error(f"过滤检索请求接口失败，状态码: {response.status_code}")
                logging.error(f"响应内容: {response.text}")
                return []

        except requests.exceptions.RequestException as e:
            logging.error(f"过滤检索请求发生异常: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            logging.error(f"过滤检索结果JSON解析错误: {str(e)}")
            return []

class MilvusQuery_hybrid_retriever:
    def __init__(self,hybrid_retrieval_url):
        self.current_data_img_paths = []
        self.expr = ""
        self._vector_field = "embedding"
        self._sparse_vector_field = "sparse_embedding"
        self._primary_field = "doc_id"
        self._text_field = "content"
        self._dummy_value = 999.0
        self.fields= ['*']
        self.hybrid_retrieval_url = hybrid_retrieval_url
    def _extract_fields(self,data) -> None:
        """Grab the existing fields from the Collection"""
        if isinstance(data, list):
            if len(data) == 0:
                 logging.info(f"hybrid_retriever检索的内容为空！")
            else:
                self.fields = list(data[0].keys())

    def parse_knowledgeUUID_filters(self, knowledgeUUID_list):
        # 将关键词字符串转换为列表
        try:
            # 构建 SQL 表达式
            expr_list = [f'metadata["knowledge_uuid"] == "{knowledgeUUID}" && metadata["indexFlag"] == 1' for knowledgeUUID in knowledgeUUID_list]
            # nas修改-nas用下面的，不需要indexFlag
            # expr_list = [f'metadata["knowledge_uuid"] == "{knowledgeUUID}"' for
            #              knowledgeUUID in knowledgeUUID_list]

            expr = " || ".join(expr_list)
            # 输出结果
            logging.info(f"混合检索过滤符：{expr}")
        except Exception as e:
            logging.error(f"混合检索构建过滤符错误: {e}")
            return ""
        return expr
    def _parse_document_hybrid_retrieval(self, data: dict) -> Document:
        embedding = data.pop(self._vector_field)
        if all(x == self._dummy_value for x in embedding):
            embedding = None

        sparse_embedding = None
        sparse_dict = data.pop(self._sparse_vector_field, None)
        if sparse_dict:
            sparse_embedding = self._convert_dict_to_sparse(sparse_dict)
            if sparse_embedding.values == [self._dummy_value] and sparse_embedding.indices == [0]:
                sparse_embedding = None
        data.pop('distance', None)
        data.pop('id', None)
        # nas修改
        data["metadata"] = json.loads(data["metadata"]) #nas要注释这个
        return Document(
            id=data.pop(self._primary_field),
            content=data.pop(self._text_field),
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            meta=data,
        )
    def _convert_dict_to_sparse(self, sparse_dict: Dict) -> SparseEmbedding:
        return SparseEmbedding(indices=list(sparse_dict.keys()), values=list(sparse_dict.values()))
    def call_milvus_hybrid_retrieval(self,search_queries,result_limit=3,knowledgeUUID:str=None,distance_to_score_fn=lambda x: x,parse_Type=None) -> List[Document]:
        """
        调用Milvus混合检索接口并返回Haystack Document列表

        参数:
            search_queries (list): 搜索查询列表
            rerank_config (dict): 重排序配置
            output_fields (list): 需要返回的字段列表，如果为None则返回所有字段
            limit (int): 最终返回结果的最大数量
            distance_to_score_fn (callable): 距离值到分数值的转换函数

        返回:
            List[Document]: Haystack Document列表
        """
        # 接口URL
        # url = "https://YOUR_RETRIEVAL_API/endpoint"
        rerank_config = {
            "strategy": "rrf",
            "params": {"k": result_limit}
        }
        output_fields = ["*"]
        # 请求头
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        # 请求体
        payload = {
            "search": search_queries,
            "rerank": rerank_config,
            "limit": result_limit,
            "outputFields": output_fields,
            "knowledgeUUID": knowledgeUUID,
            "parse_Type": parse_Type
        }
        try:
            # 发送POST请求
            response = requests.post(self.hybrid_retrieval_url, headers=headers, data=json.dumps(payload), verify=False)

            # 检查响应状态码
            if response.status_code == 200:
                result = response.json()
                if result.get("code") != 200:
                    logging.error(f"混合检索接口请求失败，错误代码: {result.get('code')}")
                    logging.error(f"错误信息: {result.get('msg', '无')}")
                    return []
                # # nas修改-nas要将以上注释，用下面注释的
                # if result.get("code") != 0:
                #     logging.error(f"混合检索接口请求失败，错误代码: {result.get('code')}")
                #     logging.error(f"错误信息: {result.get('msg', '无')}")
                #     return []

                # # nas修改
                result = result["data"]  # nas要注释这个

                # 检查返回代码是否为0（成功）
                if result.get("code") == 0:
                    docs = []
                    self._extract_fields(result["data"])
                    output_fields=self.fields
                    for item in result.get("data", []):
                        # 解析文档数据
                        data = {field: item.get(field) for field in output_fields}
                        doc = self._parse_document_hybrid_retrieval(data)
                        doc.score = distance_to_score_fn(item.get("distance", 0.0))
                        docs.append(doc)
                    return docs

                else:
                    logging.error(f"混合检索接口请求失败，错误代码: {result.get('code')}")
                    return []
            else:
                logging.error(f"混合检索接口请求，状态码: {response.status_code}")
                logging.error(f"响应内容: {response.text}")
                return []

        except requests.exceptions.RequestException as e:
            logging.error(f"混合检索接口请求发生异常: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            logging.error(f"混合检索返回结果JSON解析错误: {str(e)}")
            return []
    def processed_data_for_hybrid_retrieval(self, query_embedding, query_sparse_embedding, limit,knowledgeUUID_list):
        """
        构建混合检索的查询请求结构（稠密 + 稀疏向量）

        参数:
            query_embedding: 稠密向量（用于向量检索）
            query_sparse_embedding: 稀疏向量（用于稀疏检索）
            limit: 返回结果的数量限制

        返回:
            包含两个查询字典的列表，分别用于稠密和稀疏检索
        """
        # 构造UUID过滤符
        self.expr = self.parse_knowledgeUUID_filters(knowledgeUUID_list)

        search_queries = [
            {
                "data": [query_embedding],  # 注意：保持嵌套列表结构 [[...]]
                "annsField": "embedding",  # 指定稠密向量的字段名
                "limit": limit,  # 结果数量限制
                "outputFields": ["*"],  # 返回所有字段
                "filter": self.expr # 过滤器
            },
            {
                "data": [{"values":query_sparse_embedding.values,"indices":query_sparse_embedding.indices}],  # 稀疏向量也保持相同结构
                "annsField": "sparse_embedding",  # 指定稀疏向量的字段名
                "limit": limit,
                "outputFields": ["*"],
                "filter": self.expr  # 过滤器
            }
        ]
        return search_queries
