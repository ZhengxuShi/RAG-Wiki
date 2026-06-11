import os
import time
from pathlib import Path
import sys
# import jieba
import torch
from dotenv import load_dotenv
# 首先加载环境变量
load_dotenv()
# 然后设置CUDA设备
# os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES")

import copy
import json
import re
# import logging
from logger_local import app_logger as logging
from datetime import datetime
from haystack.dataclasses import ByteStream
from typing import Any, Dict, List
from haystack import Document, Pipeline, component
from hypster import HP, instantiate
from milvus_haystack import MilvusDocumentStore

from retriever.configs.files_routing import run_classification_router
from retriever.src.haystack_utils import Word_Filter_Muiltiple
from retriever.src.utils import MilvusQueryHandler
from retriever.src.utils import MilvusQuery_filter_retriever,MilvusQuery_hybrid_retriever

# LLM Wiki imports
from llm_wiki.graph_relevance import build_retrieval_graph, get_related_nodes
from llm_wiki.json_source_adapter import json_chunks_to_source_content
from llm_wiki.ingest import auto_ingest
from llm_wiki.wiki_models import LlmConfig
from llm_wiki.wiki_utils import read_file_utf8, normalize_path

class LoggedTime:
    def __init__(self, task_name, logging):
        self.task_name = task_name
        self.logging = logging
        self.elapsed_time = None

    def __enter__(self):
        self.start = time.perf_counter()
        self.logging.info(f"Starting {self.task_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.perf_counter() - self.start
        self.logging.info(f"Finished {self.task_name} in {self.elapsed_time:.2f} seconds")

    def get_elapsed_time(self):
        """获取耗时"""
        return self.elapsed_time

@component
class RetrieverModule:
    """
  检索器模块，封装检索器所有功能，提供对外接口

  输入：查询问题,检索方法配置文件 retrieval_config: dict
  输出：检索到的文档列表 List[Document]
  """

    def __init__(self):
        self.first_router = None
        self.use_routing = None
        self.rerank_tokenizer = None
        self.rerank_url = None
        self.rerank_timeout = None
        self.rerank_model_platform = None
        self.sparse_embedder_model_url = None
        self.sparse_timeout = None
        self.hybrid_retrieval_url = None
        self.filter_retriever_url = None
        self.query_hybrid_handler = None
        self.query_filter_handler = None
        self.retrieval_fetchType = None
        self.multiRouteRetrieval = None
        self.seg = None
        self.milvus_filename = None
        self.milvus_filename_2 = None
        self.results = None
        self.query_handler = None
        self.use_table_recovery = None
        self.json_data_path = None
        self.doc_score = None
        self.use_homepage_context = None
        self.use_concat_context = None
        self.use_compressed = None
        self.retain_count = None
        self.reranker_top_k = None
        self.reranker_model = None
        self.use_reranker = None
        self.milvus_db_name = None
        self.retrieval_db_path_url = None
        self.retrieval_embedding_similarity_function = None
        self.retrieval_indexing_method = None
        self.retrieval_method = None
        self.retrieval_hybrid_retriever_top_k = None
        self.retrieval_sparse_embedding_retriever_top_k = None
        self.query_timeout = None
        self.query_num_predict = None
        self.query_temperature = None
        self.model_url = None
        self.retrieval_embedding_retriever_top_k = None
        self.query_model_platform = None
        self.query_model = None
        self.query_methods = None
        self.embedder_timeout = None
        self.embedder_model_platform = None
        self.sparse_embedder_model_platform = None
        self.sparse_embedding_model = None
        self.embedder_model_url = None
        self.embedder_model = None
        self.llm_enrich_model = None
        self.first_retrieval_method = None
        self.milvus_collection_name = None
        self.default_config_path = "configs/retriever_config_v3.json"
        # LLM Wiki config
        self.use_llm_wiki = False
        self.wiki_config_path = None
        self.wiki_expansion_hops = 2
        self.wiki_expansion_limit = 3
        self.wiki_expansion_min_relevance = 2.0
        self.wiki_llm_config = None
    def check_gpu_availability(self,logging=None):
        """检查GPU可用性"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logging.info(f"检测到 {gpu_count} 个GPU设备")

            # 显示环境变量设置
            cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "未设置")
            logging.info(f"CUDA_VISIBLE_DEVICES 环境变量: {cuda_visible_devices}")

            for i in range(gpu_count):
                device_name = torch.cuda.get_device_name(i)
                # 获取GPU的UUID来唯一标识
                device_uuid = torch.cuda.get_device_properties(i).uuid
                logging.info(f"逻辑GPU {i}: {device_name} (UUID: {device_uuid})")

            current_device = torch.cuda.current_device()
            logging.info(f"当前使用的逻辑GPU: {current_device}")

            # 显示GPU内存使用情况
            logging.info(
                f"GPU内存使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        else:
            logging.info("未检测到可用的GPU设备")
    def init_system(self,retrieval_config: dict = None,logging=None):
        PROJECT_ROOT = find_project_root()
        os.chdir(PROJECT_ROOT)
        #
        logging.info(f"切换到项目根目录：{PROJECT_ROOT}")
        # 创建日志目录（如果不存在）
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.check_gpu_availability(logging=logging)
        # 没提供则使用默认配置
        if retrieval_config is None:
            retrieval_config = self.read_json_file(self.default_config_path)
            logging.info("未提供检索配置，使用默认配置文件")
        else:
            retrieval_config = self.read_json_file(retrieval_config)
            logging.info("已加载用户提供的检索配置")
        # 配置参数-首次检索
        first_retrieval_method_path_list = ["1000.1st_retriever", "1001.value"]
        logging.info("读取配置参数-首次检索配置值")
        self.first_retrieval_method = self.get_value_from_key_path(retrieval_config,
                                                              first_retrieval_method_path_list)
        logging.info(f"first_retrieval_method: {self.first_retrieval_method}")

        llm_enrich_model_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                      "1005.llm_enrich", "1001.value"]
        logging.info("读取配置参数-LLM丰富化模型")
        self.llm_enrich_model = self.get_value_from_key_path(retrieval_config,
                                                        llm_enrich_model_path_list)
        logging.info(f"llm_enrich_model: {self.llm_enrich_model}")

        # 配置参数-embedder
        logging.info("读取配置参数-embedder参数")
        embedder_model_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                    "1002.retrieval",
                                    "1005.detail", "1000.embedding_retriever", "1005.detail", "1000.embedding_model",
                                    "1001.value"]
        embedder_model_url_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                        "1002.retrieval",
                                        "1005.detail", "1000.embedding_retriever", "1005.detail",
                                        "1000.embedding_model", "1005.detail", "1001.embedder_model_url",
                                        "1001.value"]
        sparse_embedder_model_url_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                                 "1002.retrieval",
                                                 "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                                 "1000.sparse_embedding_model", "1005.detail", "1000.sparse_embedder_model_url",
                                                 "1001.value"]

        sparse_embedding_model_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                            "1002.retrieval",
                                            "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                            "1000.sparse_embedding_model",
                                            "1001.value"]
        embedder_model_platform_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                             "1002.retrieval",
                                             "1005.detail", "1000.embedding_retriever", "1005.detail",
                                             "1000.embedding_model", "1005.detail", "1002.embedder_model_platform",
                                             "1001.value"]
        sparse_embedder_model_platform_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                             "1002.retrieval",
                                             "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                             "1000.sparse_embedding_model", "1005.detail", "1001.sparse_embedder_model_platform",
                                             "1001.value"]

        embedder_timeout_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                      "1002.retrieval",
                                      "1005.detail", "1000.embedding_retriever", "1005.detail",
                                      "1000.embedding_model", "1005.detail", "1003.timeout",
                                      "1001.value"]
        sparse_timeout_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                     "1002.retrieval",
                                     "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                     "1000.sparse_embedding_model", "1005.detail", "1002.sparse_timeout",
                                     "1001.value"]

        self.embedder_model = self.get_value_from_key_path(retrieval_config,
                                                      embedder_model_path_list)
        logging.info(f"embedder_model：{self.embedder_model}")
        self.embedder_model_url = self.get_value_from_key_path(retrieval_config,
                                                          embedder_model_url_path_list)
        logging.info(f"embedder_model_url：{self.embedder_model_url}")
        self.sparse_embedder_model_url = self.get_value_from_key_path(retrieval_config,
                                                               sparse_embedder_model_url_path_list)
        logging.info(f"sparse_embedder_model_url：{self.sparse_embedder_model_url}")
        self. sparse_embedding_model = self.get_value_from_key_path(retrieval_config,
                                                              sparse_embedding_model_path_list)
        logging.info(f"sparse_embedding_model：{self.sparse_embedding_model}")
        self.embedder_model_platform = self.get_value_from_key_path(retrieval_config,
                                                               embedder_model_platform_path_list)
        logging.info(f"embedder_model_platform：{self.embedder_model_platform}")
        self.sparse_embedder_model_platform = self.get_value_from_key_path(retrieval_config,
                                                                    sparse_embedder_model_platform_path_list)
        logging.info(f"sparse_embedder_model_platform：{self.sparse_embedder_model_platform}")
        self.embedder_timeout = self.get_value_from_key_path(retrieval_config,
                                                        embedder_timeout_path_list)
        logging.info(f"embedder_timeout：{self.embedder_timeout}")
        self.sparse_timeout = self.get_value_from_key_path(retrieval_config,
                                                             sparse_timeout_path_list)
        logging.info(f"sparse_timeout：{self.sparse_timeout}")

        # 配置参数-query
        logging.info("读取配置参数-query参数")
        query_methods_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail", "1001.query",
                                   "1001.value"]
        query_model_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail", "1004.llm_query",
                                 "1001.value"]
        query_model_platform_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                          "1004.llm_query", "1005.detail", "1006.model_platform", "1001.value"]
        model_url_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                               "1004.llm_query",
                               "1005.detail", "1004.model_url", "1001.value"]
        query_temperature_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                       "1004.llm_query",
                                       "1005.detail", "1000.temperature", "1001.value"]
        query_num_predict_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                       "1004.llm_query",
                                       "1005.detail", "1001.num_predict", "1001.value"]
        query_timeout_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                   "1004.llm_query",
                                   "1005.detail", "1005.timeout", "1001.value"]

        self.query_methods = self.get_value_from_key_path(retrieval_config,
                                                     query_methods_path_list)
        logging.info(f"query_methods：{self.query_methods}")
        self.query_model = self.get_value_from_key_path(retrieval_config,
                                                   query_model_path_list)
        logging.info(f"query_model：{self.query_model}")
        self.query_model_platform = self.get_value_from_key_path(retrieval_config,
                                                            query_model_platform_path_list)
        logging.info(f"query_model_platform：{self.query_model_platform}")
        self.model_url = self.get_value_from_key_path(retrieval_config,
                                                 model_url_path_list)
        logging.info(f"model_url：{self.model_url}")
        self.query_temperature = self.get_value_from_key_path(retrieval_config,
                                                         query_temperature_path_list)
        logging.info(f"query_temperature：{self.query_temperature}")
        self.query_num_predict = self.get_value_from_key_path(retrieval_config,
                                                         query_num_predict_path_list)
        logging.info(f"query_num_predict：{self.query_num_predict}")
        self.query_timeout = self.get_value_from_key_path(retrieval_config,
                                                     query_timeout_path_list)
        logging.info(f"query_timeout：{self.query_timeout}")
        # 配置参数-retrieval
        logging.info("读取配置参数-retrieval参数")
        multiRouteRetrieval_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                      "1002.retrieval", "1005.detail","1004.use_multiRouteRetrieval","1001.value"]
        retrieval_embedding_retriever_top_k_path_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                                         "1005.detail", "1002.retrieval", "1005.detail",
                                                         "1000.embedding_retriever", "1005.detail", "1001.top_k",
                                                         "1001.value"]
        retrieval_sparse_embedding_retriever_top_k_path_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                                                "1005.detail", "1002.retrieval", "1005.detail",
                                                                "1001.sparse_embedding_retriever", "1005.detail",
                                                                "1001.top_k",
                                                                "1001.value"]
        filter_retriever_url_path_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                          "1005.detail", "1002.retrieval", "1005.detail",
                                          "1003.hybrid_retriever", "1005.detail", "1007.filter_retriever_url",
                                          "1001.value"]
        hybrid_retrieval_url_path_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                          "1005.detail", "1002.retrieval", "1005.detail",
                                          "1003.hybrid_retriever", "1005.detail", "1008.hybrid_retrieval_url",
                                          "1001.value"]
        retrieval_hybrid_retriever_top_k_path_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                                      "1005.detail", "1002.retrieval", "1005.detail",
                                                      "1003.hybrid_retriever", "1005.detail", "1003.top_k",
                                                      "1001.value"]

        retrieval_method_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                      "1002.retrieval", "1001.value"]
        retrieval_fetchType_path_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                          "1005.detail", "1002.retrieval", "1005.detail",
                                          "1003.hybrid_retriever", "1005.detail", "1009.retrieval_fetchType",
                                          "1001.value"]
        retrieval_indexing_method_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                               "1000.indexing", "1001.value"]
        retrieval_embedding_similarity_function_path_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                                             "1005.detail",
                                                             "1000.indexing", "1005.detail",
                                                             "1000.embedding_similarity_function", "1001.value"]
        retrieval_db_path_url_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                      "1005.detail",
                                      "1000.indexing", "1005.detail",
                                      "1001.db_path_url", "1001.value"]

        milvus_db_name_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                               "1005.detail",
                               "1000.indexing", "1005.detail",
                               "1002.milvus_db_name", "1001.value"]
        milvus_collection_name_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                       "1005.detail",
                                       "1000.indexing", "1005.detail",
                                       "1003.milvus_collection_name", "1001.value"]
        self.retrieval_embedding_retriever_top_k = self.get_value_from_key_path(retrieval_config,
                                                                           retrieval_embedding_retriever_top_k_path_list)
        logging.info(f"retrieval_embedding_retriever_top_k：{self.retrieval_embedding_retriever_top_k}")
        self.retrieval_sparse_embedding_retriever_top_k = self.get_value_from_key_path(retrieval_config,
                                                                                  retrieval_sparse_embedding_retriever_top_k_path_list)
        logging.info(f"retrieval_sparse_embedding_retriever_top_k：{self.retrieval_sparse_embedding_retriever_top_k}")
        self.retrieval_hybrid_retriever_top_k = self.get_value_from_key_path(retrieval_config,
                                                                        retrieval_hybrid_retriever_top_k_path_list)
        self.filter_retriever_url = self.get_value_from_key_path(retrieval_config,
                                                                 filter_retriever_url_path_list)
        self.hybrid_retrieval_url = self.get_value_from_key_path(retrieval_config,
                                                                 hybrid_retrieval_url_path_list)
        logging.info(f"retrieval_hybrid_retriever_top_k：{self.retrieval_hybrid_retriever_top_k}")
        logging.info(f"retrieval_hybrid_retriever_top_k：{self.retrieval_hybrid_retriever_top_k}")
        logging.info(f"retrieval_hybrid_retriever_top_k：{self.retrieval_hybrid_retriever_top_k}")
        self.retrieval_method = self.get_value_from_key_path(retrieval_config,
                                                        retrieval_method_path_list)
        logging.info(f"retrieval_method：{self.retrieval_method}")
        self.retrieval_fetchType = self.get_value_from_key_path(retrieval_config,
                                                             retrieval_fetchType_path_list)
        logging.info(f"retrieval_fetchType：{self.retrieval_fetchType}")
        self.retrieval_indexing_method = self.get_value_from_key_path(retrieval_config,
                                                                 retrieval_indexing_method_path_list)
        logging.info(f"retrieval_indexing_method：{self.retrieval_indexing_method}")
        self.retrieval_embedding_similarity_function = self.get_value_from_key_path(retrieval_config,
                                                                               retrieval_embedding_similarity_function_path_list)
        logging.info(f"retrieval_embedding_similarity_function：{self.retrieval_embedding_similarity_function}")

        self.retrieval_db_path_url = self.get_value_from_key_path(retrieval_config,
                                                             retrieval_db_path_url_list)
        logging.info(f"retrieval_db_path_url：{self.retrieval_db_path_url}")
        self.milvus_db_name = self.get_value_from_key_path(retrieval_config,
                                                      milvus_db_name_list)
        logging.info(f"milvus_db_name：{self.milvus_db_name}")
        self.milvus_collection_name = self.get_value_from_key_path(retrieval_config,
                                                              milvus_collection_name_list)
        logging.info(f"milvus_collection_name：{self.milvus_collection_name}")
        #
        self.multiRouteRetrieval=self.get_value_from_key_path(retrieval_config,
                                                              multiRouteRetrieval_path_list)
        self.multiRouteRetrieval=self.change_bool(self.multiRouteRetrieval)
        logging.info(f"multiRouteRetrieval：{self.multiRouteRetrieval}")
        # 配置参数-reranker
        logging.info("读取配置参数-reranker参数")
        use_reranker_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                  "1003.use_reranker", "1001.value"]
        reranker_model_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                    "1003.use_reranker", "1005.detail", "1000.reranker_model", "1001.value"]
        reranker_top_k_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                    "1003.use_reranker", "1005.detail", "1001.reranker_top_k", "1001.value"]
        rerank_url_path_list=["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                "1003.use_reranker", "1005.detail", "1000.reranker_model", "1005.detail",
                              "1000.rerank_url", "1001.value"]
        rerank_model_platform_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                "1003.use_reranker", "1005.detail", "1000.reranker_model", "1005.detail",
                                "1001.rerank_model_platform", "1001.value"]
        rerank_timeout_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                           "1003.use_reranker", "1005.detail", "1000.reranker_model", "1005.detail",
                                           "1002.timeout", "1001.value"]

        use_reranker = self.get_value_from_key_path(retrieval_config,
                                                    use_reranker_path_list)
        self.use_reranker = self.change_bool(use_reranker)
        logging.info(f"use_reranker：{self.use_reranker}")
        self.reranker_model = self.get_value_from_key_path(retrieval_config,
                                                      reranker_model_path_list)
        logging.info(f"reranker_model：{self.reranker_model}")
        self.reranker_top_k = self.get_value_from_key_path(retrieval_config,
                                                      reranker_top_k_path_list)
        logging.info(f"reranker_top_k：{self.reranker_top_k}")
        self.rerank_url = self.get_value_from_key_path(retrieval_config,
                                                           rerank_url_path_list)
        logging.info(f"rerank_url：{self.rerank_url}")
        self.rerank_model_platform = self.get_value_from_key_path(retrieval_config,
                                                           rerank_model_platform_path_list)
        logging.info(f"rerank_model_platform：{self.rerank_model_platform}")
        self.rerank_timeout = self.get_value_from_key_path(retrieval_config,
                                                           rerank_timeout_path_list)
        logging.info(f"rerank_timeout：{self.rerank_timeout}")

        # 配置参数-压缩检索结果
        logging.info("读取配置参数-压缩检索结果")
        use_compressed_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                    "1006.use_compressed", "1001.value"]
        retain_count_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                  "1006.use_compressed", "1005.detail", "1000.retain_count", "1001.value"]
        use_compressed = self.get_value_from_key_path(retrieval_config,
                                                      use_compressed_path_list)
        self.retain_count = self.get_value_from_key_path(retrieval_config,
                                                    retain_count_path_list)
        self.use_compressed = self.change_bool(use_compressed)
        logging.info(f"use_compressed：{self.use_compressed}")
        logging.info(f"retain_count：{self.retain_count}")

        # 配置参数-增强检索信息相关配置-添加上下文
        use_concat_context_path_list = ["1003.enhanced_retrieval", "1005.detail", "1000.use_concat_context",
                                        "1001.value"]
        use_concat_context = self.get_value_from_key_path(retrieval_config, use_concat_context_path_list)
        self.use_concat_context = self.change_bool(use_concat_context)

        # 配置参数-首页信息相关配置
        use_homepage_context_path_list = ["1003.enhanced_retrieval", "1005.detail", "1002.use_homepage_context",
                                          "1001.value"]
        use_homepage_context = self.get_value_from_key_path(retrieval_config, use_homepage_context_path_list)
        self.use_homepage_context = self.change_bool(use_homepage_context)

        doc_score_path_list = ["1003.enhanced_retrieval", "1005.detail", "1000.use_concat_context", "1005.detail",
                               "1000.doc_score", "1001.value"]
        self.doc_score = self.get_value_from_key_path(retrieval_config, doc_score_path_list)
        json_data_path_path_list = ["1003.enhanced_retrieval", "1005.detail", "1000.use_concat_context", "1005.detail",
                                    "1001.json_data_path", "1001.value"]
        self.json_data_path = self.get_value_from_key_path(retrieval_config, json_data_path_path_list)
        # 配置参数-增强检索信息相关配置-复原表格
        use_table_recovery_path_list = ["1003.enhanced_retrieval", "1005.detail", "1001.use_table_recovery",
                                        "1001.value"]
        use_table_recovery = self.get_value_from_key_path(retrieval_config, use_table_recovery_path_list)
        self.use_table_recovery = self.change_bool(use_table_recovery)

        # LLM Wiki 配置读取（必须在 instantiate 之前完成，以决定 pipeline 是否禁用内部 reranker）
        try:
            llm_wiki_path_list = ["1007.llm_wiki", "1005.detail"]
            llm_wiki_detail = self.get_value_from_key_path(retrieval_config, llm_wiki_path_list)
            use_llm_wiki_path = ["1001.use_llm_wiki", "1001.value"]
            wiki_config_path_path = ["1002.wiki_config_path", "1001.value"]
            wiki_expansion_hops_path = ["1003.wiki_expansion_hops", "1001.value"]
            wiki_expansion_limit_path = ["1004.wiki_expansion_limit", "1001.value"]
            wiki_expansion_min_relevance_path = ["1005.wiki_expansion_min_relevance", "1001.value"]

            self.use_llm_wiki = self.change_bool(self.get_value_from_key_path(llm_wiki_detail, use_llm_wiki_path))
            self.wiki_config_path = self.get_value_from_key_path(llm_wiki_detail, wiki_config_path_path)
            self.wiki_expansion_hops = self.get_value_from_key_path(llm_wiki_detail, wiki_expansion_hops_path)
            self.wiki_expansion_limit = self.get_value_from_key_path(llm_wiki_detail, wiki_expansion_limit_path)
            self.wiki_expansion_min_relevance = self.get_value_from_key_path(llm_wiki_detail, wiki_expansion_min_relevance_path)

            wiki_search_top_k_path = ["1008.wiki_search_top_k", "1001.value"]
            wiki_search_top_k = self.get_value_from_key_path(llm_wiki_detail, wiki_search_top_k_path)
            self.wiki_search_top_k = int(wiki_search_top_k) if wiki_search_top_k is not None else 20

            logging.info(f"LLM Wiki配置: use_llm_wiki={self.use_llm_wiki}, config_path={self.wiki_config_path}, wiki_search_top_k={self.wiki_search_top_k}")
        except Exception as e:
            logging.warning(f"LLM Wiki配置读取失败（可能配置中未包含1007.llm_wiki节点）: {e}")
            self.use_llm_wiki = False
            self.wiki_search_top_k = 20

        if self.retrieval_fetchType == "api":
            self.query_filter_handler = MilvusQuery_filter_retriever(filter_retriever_url=self.filter_retriever_url)
            self.query_hybrid_handler = MilvusQuery_hybrid_retriever(hybrid_retrieval_url=self.hybrid_retrieval_url)
        else:
            # 初始化milvus操作类
            self.query_handler = MilvusQueryHandler(
                uri=self.retrieval_db_path_url,
                db_name=self.milvus_db_name,
                collection_name=self.milvus_collection_name,
                logging=logging
            )
        instantiate_values = {
                "embedder.model": self.embedder_model,
                "embedder.model_url": self.embedder_model_url,
                "embedder.sparse_model": self.sparse_embedding_model,
                "embedder.sparse_embedder_model_url": self.sparse_embedder_model_url,
                "embedder.embedder_model_platform": self.embedder_model_platform,
                "embedder.sparse_embedder_model_platform": self.sparse_embedder_model_platform,
                "embedder.embedder_timeout": self.embedder_timeout,
                "embedder.sparse_timeout": self.sparse_timeout,

                "query_preprocessing.query_methods": self.query_methods,
                "query_preprocessing.model": self.query_model,
                "query_preprocessing.model_platform": self.query_model_platform,
                "query_preprocessing.model_url": self.model_url,
                "query_preprocessing.temperature": self.query_temperature,
                "query_preprocessing.num_predict": self.query_num_predict,
                "query_preprocessing.timeout": self.query_timeout,

                "retrieval.embedding_retriever_top_k": self.retrieval_embedding_retriever_top_k,
                "retrieval.retrieval_sparse_embedding_retriever_top_k": self.retrieval_sparse_embedding_retriever_top_k,
                "retrieval.retrieval_hybrid_retriever_top_k": self.retrieval_hybrid_retriever_top_k,
                "retrieval.retrieval_method": self.retrieval_method,
                "retrieval.retrieval_fetchType": self.retrieval_fetchType,
                "retrieval.collection_name": self.milvus_collection_name,
                "retrieval.db_path_url": self.retrieval_db_path_url,
                "retrieval.milvus_db_name": self.milvus_db_name,
                "retrieval.multiRouteRetrieval": self.multiRouteRetrieval,
                "use_compressed": self.use_compressed,
                "retain_count": self.retain_count,

                "use_reranker": self.use_reranker,
                "external_rerank": self.use_llm_wiki,
            }
        # Only pass internal reranker params when pipeline actually includes the component
        if not self.use_llm_wiki:
            instantiate_values.update({
                "reranker.model": self.reranker_model,
                "reranker.top_k": self.reranker_top_k,
                "reranker.rerank_model_platform": self.rerank_model_platform,
                "reranker.rerank_url": self.rerank_url,
                "reranker.timeout": self.rerank_timeout,
                "reranker.default_config_path": self.default_config_path,
            })
        self.results = instantiate(self.modular_rag, values=instantiate_values)

        pipeline = self.results["pipeline"]
        pipeline.warm_up()

        # 初始化 WikiSearcher（当启用 LLM Wiki 时）
        if self.use_llm_wiki:
            try:
                from llm_wiki.wiki_searcher import WikiSearcher
                self.wiki_searcher = WikiSearcher(
                    project_path=normalize_path(os.path.join(PROJECT_ROOT, "llm_wiki")),
                    top_k=self.wiki_search_top_k,
                    expansion_hops=self.wiki_expansion_hops,
                    expansion_limit=self.wiki_expansion_limit,
                    expansion_min_relevance=self.wiki_expansion_min_relevance,
                )
                logging.info(f"WikiSearcher 初始化完成: top_k={self.wiki_search_top_k}, hops={self.wiki_expansion_hops}, limit={self.wiki_expansion_limit}, min_relevance={self.wiki_expansion_min_relevance}")
            except Exception as e:
                logging.warning(f"WikiSearcher 初始化失败: {e}")
                self.wiki_searcher = None
        else:
            self.wiki_searcher = None

        # # 创建一个jieba分词对象
        # self.seg = jieba.Tokenizer()
        # # 设置tmp_dir为您的指定文件夹路径
        # self.seg.tmp_dir = os.getenv("JIEBA_CACHE_FOLDER")
        # # 初始化，jieba会从指定的文件夹中读取或创建jieba.cache文件
        # self.seg.initialize()

        logging.info("路由项目根目录定位成功: %s", PROJECT_ROOT)

    def simple_keyword(self, query: str):
        """
         关键词检索模块，输入问题，分割关键词，从知识库进行关键词检索
        """
        documents = []
        return {"documents": documents}

    def change_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def graphRAG(self, query: str):
        """
         图检索模块，输入问题，检索知识图谱
        """
        documents = []
        return {"documents": documents}
        pass

    def get_value_from_key_path(self, data_dict, key_path_list):
        """
        通过键路径列表获取值
        :param data_dict: 要查询的字典
        :param key_path_list: 键路径列表，例如 ["1st_retriever", "1005.detail", "general", "1000.param_meaning"]
        :return: 对应的值
        """
        current_data = data_dict
        for key in key_path_list:
            current_data = current_data[key]
        return current_data

    def insert_new_query(self, new_query, file_name, documents,logging):
        logging.info(f"开始更新metadata信息，new_query: {new_query}, 文件名: {file_name}")
        for doc in documents:
            doc.meta["metadata"]["new_query"] = new_query
            # 如果是列表，转为字符串（或用逗号分隔）
            doc.meta["metadata"]["route_file"] = (
                ", ".join(file_name) if isinstance(file_name, list) else file_name
            )
            logging.debug(f"文档元数据已更新: {doc.meta['metadata']}")
        logging.info(f"成功更新了 {len(documents)} 个文档的metadata信息")
        return documents

    def documents_to_json(self, documents):
        """
        将文档列表转换为JSON格式字符串列表。

        Args:
            documents (list): 文档对象列表，每个文档对象包含content、content_type、id、score、embedding、meta属性。

        Returns:
            list: 包含JSON格式字符串的列表。

        """
        json_documents = []
        for doc in documents:
            # 将doc的content和metadata部分转换为字典
            doc_dict = {
                "content": doc.content,
                "content_type": doc.content_type,
                "id": doc.id,
                "score": doc.score,
                "metadata": doc.meta.get("metadata") if isinstance(doc.meta.get("metadata"), dict) else None,
            }
            # # 将字典转换为JSON格式字符串
            # json_doc = json.dumps(doc_dict, ensure_ascii=False)
            # 将JSON字符串添加到列表中
            json_documents.append(doc_dict)
        return json_documents
    def enhanced_info_retrieval(self, query_handler, documents, doc_score,retrieval_fetchType,knowledgeUUID_list,parseType, logging, skip_knowledge_uuids=None):
        """
        增强信息检索：通过批量查询IN操作符优化低分文档的上下文拼接

        Args:
            skip_knowledge_uuids: 需要跳过上下文拼接的知识库UUID集合
        """
        if skip_knowledge_uuids is None:
            skip_knowledge_uuids = set()

        if not documents:
            logging.warning("输入文档列表为空，跳过拼接上下文内容处理")
            return documents

        processed_count = 0

        # 第一步：预处理，收集所有需要查询的文档信息
        with LoggedTime("拼接上下文-预处理阶段耗时统计", logging):
            query_info = []  # 存储 (knowledge_uuid, file_name, unique_id, doc_index, query_type)
            valid_doc_indices = []  # 存储可以处理的文档索引

            for idx, doc in enumerate(documents):
                # 获取文档的knowledge_uuid
                metadata = doc.meta.get("metadata", {})
                doc_knowledge_uuid = metadata.get("knowledge_uuid", "")

                # 检查是否需要跳过该文档的上下文拼接
                if doc_knowledge_uuid in skip_knowledge_uuids:
                    logging.info(f"文档knowledge_uuid为{doc_knowledge_uuid}，跳过上下文拼接")
                    continue

                # 基本属性检查
                if not self._is_valid_document(doc, doc_score, logging):
                    continue
                doc_file_name = metadata.get("file_name")
                doc_uid = metadata.get("unique_id")

                # 根据检索类型确定 knowledge_uuid
                if retrieval_fetchType == "api":
                    # API模式下从文档获取或从knowledgeUUID_list获取
                    doc_knowledge_uuid = metadata.get("knowledge_uuid")
                    # 如果没有knowledge_uuid，赋值空字符串
                    if not doc_knowledge_uuid and knowledgeUUID_list:
                        doc_knowledge_uuid = ""
                        logging.warning(f"调用api接口-上下文信息增强-数据库文件：{doc.id}，无knowledge_uuid字段！已赋值为空字符串！！!")
                else:
                    # local模式下knowledge_uuid为空字符串
                    doc_knowledge_uuid = ""

                # 文档处理能力检查
                if not self._can_process_document(doc_uid, doc_file_name, logging):
                    continue

                # 收集查询信息
                for offset in [-1, 0, 1]:
                    query_type = "previous" if offset == -1 else "current" if offset == 0 else "below"
                    query_info.append((
                        doc_knowledge_uuid,
                        doc_file_name,
                        doc_uid + offset,
                        idx,
                        query_type
                    ))
                valid_doc_indices.append(idx)

            logging.info(f"拼接上下文-预处理完成，收集查询信息数量: {len(query_info)}, 有效文档索引数: {len(valid_doc_indices)}")

        # 第二步：批量查询所有需要的文档
        with LoggedTime("拼接上下文-批量查询阶段耗时统计", logging):
            context_docs_map = self._batch_query_with_in_operator(query_handler,query_info,retrieval_fetchType,knowledgeUUID_list, parseType,logging)
            logging.info(f"拼接上下文-批量查询完成，获取文档数量: {len(context_docs_map)}")

        # 第三步：处理文档内容，拼接上下文
        with LoggedTime("拼接上下文-内容拼接阶段耗时统计", logging):
            for doc_index in valid_doc_indices:
                try:
                    doc = documents[doc_index]
                    metadata = doc.meta.get("metadata", {})
                    doc_file_name = metadata.get("file_name")
                    doc_uid = metadata.get("unique_id")

                    # 根据检索类型确定如何从context_docs_map获取文档
                    if retrieval_fetchType == "local":
                        # local模式：key为 (file_name, unique_id, query_type)
                        current_docs = context_docs_map.get((doc_file_name, doc_uid, "current"), [])
                        previous_docs = context_docs_map.get((doc_file_name, doc_uid - 1, "previous"), [])
                        below_docs = context_docs_map.get((doc_file_name, doc_uid + 1, "below"), [])
                    else:
                        # API模式：key为 (knowledge_uuid, file_name, unique_id, query_type)
                        # 需要获取文档的knowledge_uuid
                        doc_knowledge_uuid = metadata.get("knowledge_uuid", "")
                        if not doc_knowledge_uuid and knowledgeUUID_list:
                            doc_knowledge_uuid = ""
                            logging.warning(f"调用api接口-上下文信息增强-数据库文件id：{doc_uid}，无knowledge_uuid字段！已赋值为空字符串！")

                        current_docs = context_docs_map.get((doc_knowledge_uuid, doc_file_name, doc_uid, "current"), [])
                        previous_docs = context_docs_map.get((doc_knowledge_uuid, doc_file_name, doc_uid - 1, "previous"),
                                                             [])
                        below_docs = context_docs_map.get((doc_knowledge_uuid, doc_file_name, doc_uid + 1, "below"), [])

                    # 验证当前文档内容一致性
                    if current_docs and not doc.content == current_docs[0].content:
                        logging.warning(f"文档{doc_file_name},doc_uid:{doc_uid},内容与Milvus查询结果不一致，跳过上下文拼接")
                        continue

                    # 拼接内容
                    combined, combined_uid = self._combine_documents(previous_docs, doc, below_docs, logging)

                    if combined:
                        doc.content = "\n".join(combined)
                        processed_count += 1
                        doc.meta["metadata"]["unique_id"] = combined_uid
                        logging.info(f"文档内容已增强，总长度: {len(doc.content)}")

                except Exception as e:
                    logging.error(f"处理文档 {doc_file_name} (UID: {doc_uid}) 时发生异常: {str(e)}")
                    continue

            logging.info(f"拼接上下文-内容拼接完成，成功处理文档数: {processed_count}/{len(documents)}")

        logging.info(f"成功完成增强信息检索流程，共处理文档数: {processed_count}/{len(documents)}")
        return documents

    def enhanced_info_filters(self, knowledgeUUID_list, file_name, unique_ids, logging):
        """为knowledgeUUID列表构建过滤表达式（用于多knowledge_uuid查询）"""
        try:
            if not knowledgeUUID_list:
                return ""

            # 为每个knowledgeUUID构建条件
            conditions = []
            for knowledgeUUID in knowledgeUUID_list:
                condition = f'(metadata["knowledge_uuid"] == "{knowledgeUUID}" && metadata["file_name"] == "{file_name}" && metadata["unique_id"] in {list(unique_ids)})'
                conditions.append(condition)

            # 用OR连接所有条件
            expr = " || ".join(conditions)
            logging.info(f"过滤检索接口-上下文信息增强-过滤表达式: {expr}")
            return expr
        except Exception as e:
            logging.error(f"过滤检索接口-上下文信息增强-构建过滤符错误: {e}")
            return ""

    def _batch_query_with_in_operator(self, query_handler, query_info, retrieval_fetchType, knowledgeUUID_list,
                                      parseType, logging):
        """
        使用IN操作符批量查询文档
        """
        context_docs_map = {}

        if not query_info:
            return context_docs_map

        try:
            # 根据检索类型决定query_info的结构
            if retrieval_fetchType == "local":
                # local模式：query_info结构为 (knowledge_uuid, file_name, unique_id, doc_index, query_type)
                # 按文件名分组
                with LoggedTime("批量查询-local模式分组耗时统计", logging):
                    file_query_groups = {}

                    for knowledge_uuid,file_name, unique_id, doc_index, query_type in query_info:
                        if file_name not in file_query_groups:
                            file_query_groups[file_name] = set()
                        file_query_groups[file_name].add(unique_id)

                    logging.info(f"批量查询-local模式分组完成，分组数量: {len(file_query_groups)}")

                # 对每个文件执行一次IN查询
                with LoggedTime("批量查询-local模式查询执行耗时统计", logging):
                    for file_name, unique_ids in file_query_groups.items():
                        try:
                            # 为每次local查询添加耗时统计
                            import time
                            local_call_start = time.perf_counter()

                            # 构建IN查询条件（local模式没有knowledge_uuid）
                            filters = {
                                "operator": "AND",
                                "conditions": [
                                    {
                                        "field": "metadata['file_name']",
                                        "operator": "==",
                                        "value": file_name
                                    },
                                    {
                                        "field": "metadata['unique_id']",
                                        "operator": "in",
                                        "value": list(unique_ids)
                                    }
                                ]
                            }
                            batch_results = query_handler.query_documents(filters=filters)

                            local_call_elapsed = time.perf_counter() - local_call_start
                            logging.info(
                                f"批量查询-local单次查询耗时统计: {local_call_elapsed:.4f} 秒, 文件={file_name}, 查询unique_id数={len(unique_ids)}, 命中结果数={len(batch_results)}")

                            # 将查询结果映射到对应的文档和类型
                            for doc in batch_results:
                                doc_metadata = doc.meta.get("metadata", {})
                                doc_file_name = doc_metadata.get("file_name", "")
                                doc_uid = doc_metadata.get("unique_id", "")

                                # 找到所有需要此文档的查询信息
                                for info in query_info:
                                    knowledge_uuid,info_file_name, info_uid, _, info_query_type = info
                                    if info_file_name == doc_file_name and info_uid == doc_uid:
                                        key = (doc_file_name, doc_uid, info_query_type)
                                        context_docs_map[key] = [doc]
                                        break

                        except Exception as e:
                            logging.error(f"批量查询文件 {file_name} 时出错: {str(e)}")
                            continue

            else:
                # API模式：query_info结构需要包含knowledge_uuid
                # 期望结构: (knowledge_uuid, file_name, unique_id, doc_index, query_type)

                # 按(knowledge_uuid, 文件名)分组
                with LoggedTime("批量查询-API模式分组耗时统计", logging):
                    query_groups = {}

                    for knowledge_uuid, file_name, unique_id, doc_index, query_type in query_info:
                        group_key = (knowledge_uuid, file_name)
                        if group_key not in query_groups:
                            query_groups[group_key] = set()
                        query_groups[group_key].add(unique_id)

                    logging.info(f"批量查询-API模式分组完成，分组数量: {len(query_groups)}")

                # 对每个(knowledge_uuid, 文件名)组合执行一次IN查询
                with LoggedTime("批量查询-API模式查询执行耗时统计", logging):
                    for (knowledge_uuid, file_name), unique_ids in query_groups.items():
                        try:
                            # API检索模式 - 为每次API调用添加耗时统计
                            import time
                            api_call_start = time.perf_counter()

                            expr = self.enhanced_info_filters(knowledgeUUID_list, file_name, unique_ids, logging)
                            batch_results = query_handler.call_milvus_filtered_retrieval(expr=expr,knowledgeUUID_list=knowledgeUUID_list,parseType=parseType)

                            api_call_elapsed = time.perf_counter() - api_call_start
                            logging.info(
                                f"批量查询-API单次调用耗时统计: {api_call_elapsed:.4f} 秒, knowledgeUUID_list={knowledgeUUID_list}, 文件={file_name}, 查询unique_id数={len(unique_ids)}, 命中结果数={len(batch_results)}")

                            # 将查询结果映射到对应的文档和类型
                            for doc in batch_results:
                                doc_metadata = doc.meta["metadata"]
                                doc_knowledge_uuid = doc_metadata.get("knowledge_uuid", "")
                                doc_file_name = doc_metadata.get("file_name", "")
                                doc_uid = doc_metadata.get("unique_id", "")

                                # 找到所有需要此文档的查询信息
                                for info in query_info:
                                    info_knowledge_uuid, info_file_name, info_uid, _, info_query_type = info
                                    if (info_knowledge_uuid == doc_knowledge_uuid and
                                            info_file_name == doc_file_name and
                                            info_uid == doc_uid):
                                        key = (doc_knowledge_uuid, doc_file_name, doc_uid, info_query_type)
                                        context_docs_map[key] = [doc]
                                        break

                        except Exception as e:
                            logging.error(f"批量查询 knowledge_uuid={knowledge_uuid}, 文件 {file_name} 时出错: {str(e)}")
                            continue

        except Exception as e:
            logging.error(f"批量查询过程中发生异常: {str(e)}")

        return context_docs_map

    def _is_valid_document(self, doc, doc_score, logging):
        """验证文档基本属性"""
        if not hasattr(doc, 'score') or not hasattr(doc, 'meta') or not doc.meta:
            logging.warning("文档缺少必要属性，跳过上下文拼接")
            return False

        if doc.score >= doc_score:
            logging.warning(f"文档分数{doc.score}高于阈值{doc_score}，跳过上下文拼接")
            return False

        return True

    def _is_valid_document_timed(self, doc, doc_score, logging):
        """验证文档基本属性（带耗时统计）"""
        import time
        start = time.perf_counter()
        result = self._is_valid_document(doc, doc_score, logging)
        elapsed = time.perf_counter() - start
        if elapsed > 0.001:  # 只在耗时超过1ms时记录
            logging.info(f"文档验证耗时统计: {elapsed:.4f} 秒")
        return result

    def _can_process_document(self, doc_uid, doc_file_name, logging):
        """检查文档是否可以处理"""
        if not doc_uid:
            logging.warning("文档unique_id为空，跳过上下文拼接")
            return False
        if isinstance(doc_uid, list):
            logging.warning("文档unique_id是列表类型，已经过表格还原，跳过上下文拼接")
            return False
        if not doc_file_name:
            logging.warning("文档file_name为空，跳过上下文拼接")
            return False
        if not isinstance(doc_uid, (int, float)):
            logging.warning(f"文档unique_id {doc_uid} 不是数字类型，跳过上下文拼接")
            return False

        return True

    def _combine_documents(self, previous_docs, current_doc, below_docs, logging):
        """合并文档内容"""
        combined = []
        combined_uid = []

        # 添加上文文档
        if previous_docs and hasattr(previous_docs[0], 'content'):
            combined.append(previous_docs[0].content)
            ori_id = previous_docs[0].meta["metadata"].get("ori_id_1", [])
            uid_to_append = ori_id if ori_id else [previous_docs[0].meta["metadata"]["unique_id"]]
            combined_uid.append(uid_to_append)
            logging.info(f"已获取上一条文档内容，长度: {len(previous_docs[0].content)}")

        # 添加当前文档
        if hasattr(current_doc, 'content'):
            combined.append(current_doc.content)
            ori_id = current_doc.meta["metadata"].get("ori_id_1", [])
            uid_to_append = ori_id if ori_id else [current_doc.meta["metadata"]["unique_id"]]
            combined_uid.append(uid_to_append)

        # 添加下文文档
        if below_docs and hasattr(below_docs[0], 'content'):
            combined.append(below_docs[0].content)
            ori_id = below_docs[0].meta["metadata"].get("ori_id_1", [])
            uid_to_append = ori_id if ori_id else [below_docs[0].meta["metadata"]["unique_id"]]
            combined_uid.append(uid_to_append)
            logging.info(f"已获取下一条文档内容，长度: {len(below_docs[0].content)}")

        return combined, combined_uid
    def enhanced_chunk_links(self, query_handler, documents, doc_score):
        if not documents:
            logging.warning("输入文档列表为空，跳过chunk_links处理")
            return documents

        processed_count = 0
        for doc in documents:
            doc_file_name = doc.meta["metadata"]["file_name"]
            try:
                if not hasattr(doc, 'score') or not hasattr(doc, 'meta') or not doc.meta:
                    logging.warning("文档缺少必要属性，跳过处理")
                    continue

                if doc.score >= doc_score:
                    logging.warning(f"文档分数{doc.score}高于阈值{doc_score}，跳过chunk_links拼接")
                    continue

                doc_uid = doc.meta.get("metadata", {}).get("unique_id")
                if not doc_uid:
                    logging.warning(f"文档unique_id为空，跳过chunk_links拼接")
                    continue
                if isinstance(doc_uid, list):
                    logging.warning(f"文档unique_id是列表类型，已经过表格还原，跳过chunk_links拼接")
                    continue
                if not doc_file_name:
                    logging.warning(f"文档{doc_file_name}的file_name为空，跳过chunk_links拼接")
                    continue

                logging.info(
                    f"当前文档分数{doc.score}低于阈值{doc_score}，准备获取chunk_links文档，文件名: {doc_file_name}，unique_id: {doc_uid}")

                # Get context documents from Milvus
                try:
                    this_doc = query_handler.query_documents(
                        filters={
                            "operator": "AND",
                            "conditions": [
                                {
                                    "field": "metadata['file_name']",
                                    "operator": "==",
                                    "value": doc_file_name
                                },
                                {
                                    "field": "metadata['unique_id']",
                                    "operator": "==",
                                    "value": doc_uid
                                }
                            ]
                        }
                    ) if isinstance(doc_uid, int) else []

                    if this_doc:
                        if not doc.content == this_doc[0].content:
                            logging.warning(f"文档内容与查询结果不一致，跳过chunk_links拼接")
                            continue
                    else:
                        logging.warning(
                            f"文档{doc_file_name}的unique_id：{doc_uid}在milvus中查询结果为空或不为int，跳过chunk_links拼接!")
                        continue

                    # 获取chunk_link
                    chunk_link = this_doc[0].meta["metadata"].get("chunk_link", [])
                    chunk_docs = []

                    if chunk_link:
                        # 如果chunk_link长度为1（无关联chunk），则使用上下文策略
                        if len(chunk_link) == 1:
                            logging.info(f"chunk_link长度为1，使用上下文策略替代")

                            # 获取上下文文档
                            previous_doc = query_handler.query_documents(
                                filters={
                                    "operator": "AND",
                                    "conditions": [
                                        {
                                            "field": "metadata['file_name']",
                                            "operator": "==",
                                            "value": doc_file_name
                                        },
                                        {
                                            "field": "metadata['unique_id']",
                                            "operator": "==",
                                            "value": doc_uid - 1
                                        }
                                    ]
                                }
                            ) if isinstance(doc_uid, (int, float)) else []

                            below_doc = query_handler.query_documents(
                                filters={
                                    "operator": "AND",
                                    "conditions": [
                                        {
                                            "field": "metadata['file_name']",
                                            "operator": "==",
                                            "value": doc_file_name
                                        },
                                        {
                                            "field": "metadata['unique_id']",
                                            "operator": "==",
                                            "value": doc_uid + 1
                                        }
                                    ]
                                }
                            ) if isinstance(doc_uid, (int, float)) else []

                            # 创建上下文文档列表
                            context_docs = []
                            if previous_doc:
                                context_docs.append(previous_doc[0])
                            context_docs.append(this_doc[0])  # 当前文档
                            if below_doc:
                                context_docs.append(below_doc[0])

                            chunk_docs = context_docs
                        else:
                            # 正常处理多个chunk_link
                            for chunk_id in chunk_link:
                                chunk_doc = query_handler.query_documents(
                                    filters={
                                        "operator": "AND",
                                        "conditions": [
                                            {
                                                "field": "metadata['file_name']",
                                                "operator": "==",
                                                "value": doc_file_name
                                            },
                                            {
                                                "field": "metadata['unique_id']",
                                                "operator": "==",
                                                "value": chunk_id
                                            }
                                        ]
                                    }
                                ) if isinstance(chunk_id, int) else []

                                if chunk_doc:
                                    chunk_docs.extend(chunk_doc)

                except Exception as e:
                    logging.error(f"查询chunk_links文档时出错: {str(e)}")
                    continue

                # Create a new list combining all documents
                combined = []
                combined_uid = []
                seen_uids = set()  # 用于跟踪已处理的UID
                for doc_ in chunk_docs:
                    if doc_:  # 确保doc_不为空
                        # 获取当前文档的UID
                        current_uid = doc_.meta["metadata"].get("unique_id")
                        # 检查UID是否已处理过
                        if current_uid in seen_uids:
                            logging.info(f"跳过重复的UID: {current_uid}")
                            continue
                        # 添加到已处理集合
                        seen_uids.add(current_uid)
                        combined.append(doc_.content)
                        ori_id = doc_.meta["metadata"].get("ori_id_1", [])
                        uid_to_append = ori_id if ori_id else [doc_.meta["metadata"]["unique_id"]]
                        combined_uid.append(uid_to_append)
                        logging.info(f"已获取关联文档内容，长度: {len(doc_.content)}")

                # 将增强信息添加到documents中
                if combined:
                    doc.content = "\n".join(combined)
                    processed_count += 1
                    doc.meta["metadata"]["unique_id"] = combined_uid
                    logging.info(f"文档内容已增强，总长度: {len(doc.content)}")
                    logging.info(f"文档内容已增强，最细粒度的uid: {combined_uid}")

            except Exception as e:
                logging.error(f"拼接chunk_links处理文档时发生异常: {str(e)}")
                continue

        logging.info(f"成功完成chunk_links检索流程，共处理文档数: {processed_count}/{len(documents)}")
        return documents

    def enhanced_info_retrieval_old(self, file_path, documents, doc_score):
        try:
            # 检查输入参数有效性
            if not documents:
                logging.info("输入文档列表为空，无需处理")
                return documents

            if not file_path:
                logging.error("文件路径不能为空")
                return documents

            # 读取JSON文件
            try:
                json_data = self.read_json_file(file_path)
                if not json_data:
                    logging.info("JSON文件内容为空")
                    return documents
            except Exception as e:
                logging.error(f"读取JSON文件失败: {str(e)}")
                return documents

            # 处理每个文档
            for doc in documents:
                try:
                    if not hasattr(doc, 'score') or not hasattr(doc, 'content'):
                        logging.warning(f"文档缺少必要属性: {doc}")
                        continue

                    if doc.score < doc_score:
                        matched_blocks = []

                        # 查找匹配块
                        for i, block in enumerate(json_data):
                            try:
                                if not isinstance(block, dict) or 'text' not in block:
                                    logging.warning(f"JSON块格式不正确，索引{i}")
                                    continue

                                if block["text"] == doc.content:
                                    # 获取匹配块及其相邻块
                                    matched_index = i

                                    # 前一块
                                    if matched_index > 0:
                                        prev_block = json_data[matched_index - 1]
                                        if isinstance(prev_block, dict) and 'text' in prev_block:
                                            matched_blocks.append(prev_block)
                                        else:
                                            logging.warning(f"前一块格式不正确，索引{matched_index - 1}")

                                    # 当前块
                                    matched_blocks.append(block)

                                    # 后一块
                                    if matched_index < len(json_data) - 1:
                                        next_block = json_data[matched_index + 1]
                                        if isinstance(next_block, dict) and 'text' in next_block:
                                            matched_blocks.append(next_block)
                                        else:
                                            logging.warning(f"后一块格式不正确，索引{matched_index + 1}")

                                    break

                            except Exception as e:
                                logging.error(f"处理JSON块时出错，索引{i}: {str(e)}")
                                continue

                        if matched_blocks:
                            try:
                                concatenated_text = '\n'.join(
                                    [block['text'] for block in matched_blocks if 'text' in block])
                                doc.content = concatenated_text
                                logging.info(f"成功拼接文档内容，原长度{len(doc.content)}，新长度{len(concatenated_text)}")
                            except Exception as e:
                                logging.error(f"拼接文本时出错: {str(e)}")

                except Exception as e:
                    logging.error(f"处理文档时出错: {str(e)}")
                    continue

        except Exception as e:
            logging.error(f"函数执行过程中发生未预期错误: {str(e)}")

        return documents

    def filter_content(self, content):
        # Remove all '\' and '-' characters from the string
        return content.replace('\\n', '').replace('\n', '').replace('\\', '').replace(' ', '')

    def filter_homepage_documents(self, query_handler, file_name,logging):
        # 从milvus读取首页信息
        try:
            homepage_docs = query_handler.query_documents(
                filters={} if not file_name else {
                    "operator": "OR",
                    "conditions": [
                        {
                            "operator": "AND",
                            "conditions": [
                                {
                                    "field": "metadata['file_name']",
                                    "operator": "==",
                                    "value": filename_
                                },
                                {
                                    "field": "metadata['page_idx']",
                                    "operator": "==",
                                    "value": 0
                                }
                            ]
                        }
                        for filename_ in file_name
                    ]
                }
            )

            # 处理可能的None或非可迭代返回值
            if homepage_docs is None:
                logging.warning("query_documents返回None而不是列表")
                return []

            if not isinstance(homepage_docs, (list, tuple)):
                logging.warning(f"query_documents返回了非列表类型: {type(homepage_docs)}")
                return []

            filter_docs = []
            for doc in homepage_docs:
                if doc is None:
                    logging.debug("跳过None文档")
                    continue

                if hasattr(doc, 'content'):
                    content = doc.content or ""  # 处理content为None的情况
                    filter_docs.append(content)
                    logging.debug(f"文档首页内容: {content[:10]}...")  # 只记录前10个字符以避免日志过长
                else:
                    logging.warning(f"文档缺少content属性: {doc}")

            logging.info(f"最终筛选出 {len(filter_docs)} 条文档内容")
            return filter_docs

        except Exception as e:
            logging.error(f"处理首页文档时发生错误: {str(e)}", exc_info=True)
            return []

    def homepage_info_retrieval(self, query_handler, file_name, homepage_documents,logging):

        # 从milvus读取首页信息
        homepage_docs = query_handler.query_documents(
            filters={} if not file_name else {
                "operator": "OR",
                "conditions": [
                    {
                        "operator": "AND",
                        "conditions": [
                            {
                                "field": "metadata['file_name']",
                                "operator": "==",
                                "value": filename_
                            },
                            {
                                "field": "metadata['page_idx']",
                                "operator": "==",
                                "value": 0
                            }
                        ]
                    }
                    for filename_ in file_name
                ]
            }
        )
        filter_docs = []
        matched_blocks = []
        for doc in homepage_docs:
            filter_docs.append(doc.content)
            logging.debug(f"文档首页内容: {doc.content[:10]}...")  # 只记录前10个字符以避免日志过长

        for doc in homepage_documents:
            doc = doc.replace("'", "")
            doc = self.filter_content(doc)

            for block in filter_docs:
                block_filter = self.filter_content(block)
                if doc in block_filter:
                    matched_blocks.append(block)
                    break
        return matched_blocks

    def read_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def filter_query_from_json(self, json_path, query):
        data = self.read_json_file(json_path)
        # 获取filter_words列表
        filter_words = data.get('filter_words', [])

        for word in filter_words:
            query = query.replace(word, "").strip()

        return query

    def contains_keyword(self, query: str, home_page_keywords: list[str]) -> bool:
        """
        判断查询字符串是否包含关键词列表中的任意一个元素。

        :param query: 用户查询字符串
        :param home_page_keywords: 关键词列表
        :return: 如果查询字符串包含至少一个关键词，则返回True，否则返回False
        """
        for keyword in home_page_keywords:
            if keyword in query:
                return True
        return False
    def calculate_similarity(self,reranker,query, documents):
        result = reranker.run(
            query=query,
            documents=documents
        )
        return result['documents'] if result else None
    def validate_and_correct_name(self, name):
        """
        Check if the first character of the name is an underscore or a letter.
        If not, prepend an underscore to the name.

        Args:
        name (str): The name to be checked and possibly corrected.

        Returns:
        str: The corrected name if it was invalid, or the original name if it was valid.
        """
        if not name[0].isalpha() and name[0] != '_':
            # Prepend an underscore if the first character is not a letter or an underscore
            return '_' + name
        else:
            # Return the original name if it's already valid
            return name

    def filter_documents_by_llm_judgment(self,documents, llm_answer):
        """
        根据LLM的判断结果过滤文档

        Args:
            documents: 原始文档列表
            llm_answer: LLM返回的判断字符串

        Returns:
            过滤后的文档列表
        """
        try:
            # 提取所有判断结果
            judgments = {}
            for line in llm_answer.strip().split('\n'):
                match = re.match(r'Document\s*(\d+):\s*(yes|no)', line, re.IGNORECASE)
                if match:
                    doc_num = int(match.group(1))
                    judgment = match.group(2).lower()
                    judgments[doc_num] = judgment

            # 过滤文档
            filtered_docs = []
            for i, doc in enumerate(documents):
                doc_num = i + 1  # 转换为1-based编号
                if doc_num in judgments and judgments[doc_num] == 'yes':
                    filtered_docs.append(doc)

            return filtered_docs

        except Exception as e:
            logging.error(f"过滤文档时发生错误: {e}")
            return documents  # 出错时返回原始文档
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
    def modular_rag(self, hp: HP):
        from retriever.src.haystack_utils import (
            PassThroughDocuments,
            TextEbd_EmbeddingRetriever,
            Query2TextEbd,
            Content_Restore,
            PassThroughDocuments_Rerank,
            Compressed_Search_Results,
            SimpleDocumentJoiner
        )
        from retriever.configs.fast_embed import fast_embed
        from retriever.configs.query import query
        from retriever.configs.milvus_retrieval import milvus_retrieval
        # 初始化参数
        query_preprocessing = hp.nest(query, name="query_preprocessing")
        # 初始化参数
        embedder = hp.nest(fast_embed, name="embedder")
        # 初始化参数
        retrieval = hp.nest(milvus_retrieval, name="retrieval")

        pipeline = query_preprocessing["pipeline"]

        # 添加通用组件
        pipeline.add_component("retrieved_documents", PassThroughDocuments())
        pipeline.add_component("content_restore", Content_Restore())

        if retrieval["multiRouteRetrieval"]:
            # === 多路检索模式 ===
            pipeline.add_component("word_filter_2", Word_Filter_Muiltiple())
            pipeline.add_component("query2text_ebd_1", Query2TextEbd(embedder, retrieval))
            pipeline.add_component("query2text_ebd_2", Query2TextEbd(embedder, retrieval))
            pipeline.add_component("textEbd_embedding_retriever_1", TextEbd_EmbeddingRetriever(retrieval))
            pipeline.add_component("textEbd_embedding_retriever_2", TextEbd_EmbeddingRetriever(retrieval))
            pipeline.add_component("document_merger", SimpleDocumentJoiner())

            # 连接数据流
            pipeline.connect("query2query", "query2text_ebd_1")
            pipeline.connect("word_filter_2", "query2text_ebd_2")
            pipeline.connect("query2text_ebd_1", "textEbd_embedding_retriever_1")
            pipeline.connect("query2text_ebd_2", "textEbd_embedding_retriever_2")
            pipeline.connect("textEbd_embedding_retriever_1.documents", "document_merger.documents1")
            pipeline.connect("textEbd_embedding_retriever_2.documents", "document_merger.documents2")
            pipeline.connect("document_merger", "content_restore")
        else:
            # === 单路检索模式 ===
            pipeline.add_component("query2text_ebd", Query2TextEbd(embedder, retrieval))
            pipeline.add_component("textEbd_embedding_retriever", TextEbd_EmbeddingRetriever(retrieval))

            # 连接数据流
            pipeline.connect("query2query", "query2text_ebd")
            pipeline.connect("query2text_ebd", "textEbd_embedding_retriever")
            pipeline.connect("textEbd_embedding_retriever", "content_restore")

        # 公共连接部分
        pipeline.connect("content_restore", "retrieved_documents")

        from retriever.src.haystack_utils import PassThroughDocuments_Rerank, Compressed_Search_Results

        retain_count = hp.int(default=20,name="retain_count")
        use_compressed = hp.select([True, False], default=True,name="use_compressed")

        if use_compressed:
            pipeline.add_component("compressed_search_results", Compressed_Search_Results(retain_count))
            pipeline.connect("retrieved_documents", "compressed_search_results")

        pipeline.add_component("docs_for_generation", PassThroughDocuments_Rerank())
        use_reranker = hp.select([True, False], default=True,name="use_reranker")
        external_rerank = hp.select([True, False], default=False, name="external_rerank")
        if use_reranker and not external_rerank and use_compressed:
            from retriever.configs.reranker import reranker
            reranker = hp.nest(reranker, name="reranker")
            pipeline.add_component("reranker", reranker["reranker"])
            pipeline.connect("compressed_search_results", "reranker")
            pipeline.connect("reranker", "docs_for_generation")
            pipeline.connect("query", "reranker")
        elif use_reranker and not external_rerank:
            from retriever.configs.reranker import reranker
            reranker = hp.nest(reranker, name="reranker")
            pipeline.add_component("reranker", reranker["reranker"])
            pipeline.connect("retrieved_documents", "reranker")
            pipeline.connect("reranker", "docs_for_generation")
            pipeline.connect("query", "reranker")
        elif use_compressed:
            pipeline.connect("compressed_search_results", "docs_for_generation")
        else:
            pipeline.connect("retrieved_documents", "docs_for_generation")
        return {"pipeline": pipeline}

    def _build_reranker_instance(self):
        """根据当前配置构造一个独立的 reranker 实例，供外部 RRF 后调用。"""
        from retriever.configs.reranker import reranker as reranker_config_fn
        from hypster import instantiate
        rr_config = instantiate(
            reranker_config_fn,
            values={
                "model": self.reranker_model,
                "top_k": self.reranker_top_k,
                "rerank_model_platform": self.rerank_model_platform,
                "rerank_url": self.rerank_url,
                "timeout": self.rerank_timeout,
                "default_config_path": self.default_config_path,
            },
        )
        return rr_config["reranker"]

    def _wiki_enhance_results(self, documents: list[Document], logging, min_relevance: float = None) -> list[Document]:
        """使用 LLM Wiki 图谱扩展增强检索结果（已废弃，请使用 WikiSearcher）。"""
        logging.warning("_wiki_enhance_results 已废弃，当前应使用 WikiSearcher 进行独立 Wiki 检索")
        if not documents:
            return documents

        # 使用传入的阈值，否则使用配置默认值
        relevance_threshold = min_relevance if min_relevance is not None else self.wiki_expansion_min_relevance

        # 1. 提取种子文件名并映射到 source summary page
        from llm_wiki.wiki_utils import source_summary_slug_from_identity
        project_path = normalize_path(os.path.join(find_project_root(), "llm_wiki"))
        seed_ids: set[str] = set()
        for doc in documents:
            file_name = doc.meta.get("metadata", {}).get("file_name", "")
            if file_name:
                seed_ids.add(source_summary_slug_from_identity(file_name))

        if not seed_ids:
            logging.info("Wiki 增强: 未从检索结果中提取到有效文件名，跳过图谱扩展")
            return documents

        # 2. 构建检索图（模块级缓存）
        try:
            graph = build_retrieval_graph(project_path, data_version=0)
        except Exception as e:
            logging.warning(f"Wiki 增强: 构建检索图失败: {e}")
            return documents

        if not graph.nodes:
            logging.info("Wiki 增强: 检索图为空，跳过")
            return documents

        # 3. 图谱扩展
        related_map: dict[str, float] = {}
        for seed_id in seed_ids:
            if seed_id not in graph.nodes:
                continue
            try:
                related = get_related_nodes(seed_id, graph, limit=self.wiki_expansion_limit)
                for item in related:
                    node = item["node"]
                    relevance = item["relevance"]
                    if relevance < relevance_threshold:
                        continue
                    if node.id in related_map:
                        related_map[node.id] = max(related_map[node.id], relevance)
                    else:
                        related_map[node.id] = relevance
            except Exception as e:
                logging.warning(f"Wiki 增强: 扩展节点 {seed_id} 失败: {e}")

        if not related_map:
            logging.info("Wiki 增强: 未找到关联节点，返回原始结果")
            return documents

        # 4. 读取扩展页面内容
        wiki_docs: list[Document] = []
        for node_id, relevance in sorted(related_map.items(), key=lambda x: x[1], reverse=True):
            node = graph.nodes.get(node_id)
            if not node or not node.path:
                continue
            try:
                content = read_file_utf8(node.path)
                # 简单截断，避免过长
                if len(content) > 8000:
                    content = content[:8000] + "\n\n[...truncated...]"
                wiki_doc = Document(
                    content=content,
                    meta={
                        "metadata": {
                            "source": "llm_wiki_graph",
                            "wiki_node_id": node_id,
                            "wiki_node_title": node.title,
                            "wiki_relevance": relevance,
                            "file_name": node_id,
                        }
                    },
                )
                wiki_docs.append(wiki_doc)
            except Exception as e:
                logging.warning(f"Wiki 增强: 读取页面 {node.path} 失败: {e}")

        if not wiki_docs:
            return documents

        # 5. RRF 融合
        def rrf_score(rank: int, k: int = 60) -> float:
            return 1.0 / (k + rank)

        scores: dict[int, float] = {}
        # FW_RAG 结果按原顺序赋分
        for rank, doc in enumerate(documents, start=1):
            scores[id(doc)] = rrf_score(rank)

        # Wiki 结果按 relevance 排序赋分
        for rank, doc in enumerate(wiki_docs, start=1):
            scores[id(doc)] = scores.get(id(doc), 0) + rrf_score(rank)

        all_docs = documents + wiki_docs
        all_docs.sort(key=lambda d: scores.get(id(d), 0), reverse=True)

        # 去重（基于 content 简单去重）
        seen_content: set[str] = set()
        fused: list[Document] = []
        for doc in all_docs:
            key = doc.content[:200]
            if key not in seen_content:
                seen_content.add(key)
                fused.append(doc)

        retain = getattr(self, "retain_count", len(fused))
        if retain and len(fused) > retain:
            fused = fused[:retain]

        logging.info(f"Wiki 增强: 原始 {len(documents)} 条，扩展 {len(wiki_docs)} 条，融合后 {len(fused)} 条")
        return fused

    def calculate_tokens(self,text: str):
        """计算单个文本的token数"""
        if not text or not isinstance(text, str):
            return 0
        # 使用tiktoken库计算tokens
        import tiktoken
        # 获取编码器
        encoding = tiktoken.get_encoding("cl100k_base")
        # 计算prompt tokens
        prompt_tokens = len(encoding.encode(text))
        return prompt_tokens
    @component.output_types(documents=List[Document])
    def run(self, query: str, retrieval_config: dict = None, logging=None,knowledgeUUID: str = None,parseType=None, use_wiki: bool = False, wiki_expansion_min_relevance: float = None):
        time_costs = {}
        # 没提供则使用默认配置
        if retrieval_config is None:
            retrieval_config = self.read_json_file(self.default_config_path)
            logging.info("未提供检索配置，使用默认配置文件")
        else:
            retrieval_config = self.read_json_file(retrieval_config)
            logging.info("已加载用户提供的检索配置")

        logging.info(f"原始用户问题：{query}")
        if knowledgeUUID :
            logging.info(f"检索接口入参向量库id：{knowledgeUUID}")

        # 读取路由配置
        use_routing_path_list = ["1002.files_routing", "1005.detail", "1006.use_routing","1001.value"]
        first_router_path_list = ["1002.files_routing", "1005.detail", "1006.use_routing",
                                  "1005.detail", "1000.first_router", "1001.value"]
        files_routing_path_path_list = ["1002.files_routing", "1005.detail", "1002.files_routing_path", "1001.value"]
        files_routing = self.get_value_from_key_path(retrieval_config,
                                                     files_routing_path_path_list)
        logging.info(f"读取文件路由文件路径参数: {files_routing}")
        use_routing = self.get_value_from_key_path(retrieval_config,
                                                     use_routing_path_list)
        first_router = self.get_value_from_key_path(retrieval_config,
                                                    first_router_path_list)
        self.use_routing = self.change_bool(use_routing)
        self.first_router = self.change_bool(first_router)

        if self.use_routing:
            logging.info(f"*****开启文件路由模块*****")
            if self.first_router:
                try:
                    logging.info(f"*****开启一级文件路由模块*****")
                    with LoggedTime("路由耗时统计", logging):
                        file_name = run_classification_router(query=query, retrieval_config=retrieval_config,
                                                              logging=logging)
                    # 多文件名
                    logging.info(f"成功通过一级路由获取文件名: {file_name}")
                    # 对问题进行去文件名处理
                    query_new = self.filter_query_from_json(files_routing, query)
                    logging.info(f"对问题进行去文件名处理后的问题: {query_new}")
                except ValueError as e:
                    logging.error(f"分类路由错误: {e}")

            logging.info(f"*****结束文件路由模块*****")
        else:
            logging.info(f"*****未开启文件路由模块*****")
            file_name=[]
            query_new = query
            logging.info(f"问题未进行去文件名处理: {query_new}")

        logging.info(f"本次检索要查询的表名: {self.milvus_collection_name}")
        logging.info(f"本次检索要查询的文件名: {file_name}")
        if self.multiRouteRetrieval:
            # 读取sop类文件名
            files_routing_data = self.read_json_file(files_routing)
            self.milvus_filename_2 = files_routing_data.get("summary_document", [])
        self.milvus_filename = file_name
        while True:
            # 首次检索
            if self.first_retrieval_method == "general":
                pipeline = self.results["pipeline"]
                homepage_documents = []
                # 读取首页问题关键词
                if self.use_homepage_context:
                    logging.info("*****开启首页问题策略*****")
                    from retriever.src.haystack_utils import LLM_Evaluation
                    # 获取 home_page_query 列表
                    data = self.read_json_file(files_routing)
                    # 获取filter_words列表
                    home_page_keywords = data.get('home_page_keywords', [])
                    logging.info(f"获取到的首页信息触发词: {home_page_keywords}")
                    if self.contains_keyword(query, home_page_keywords):
                        logging.info(f"问题包含首页信息触发词，问题为: {query}")
                        homepage_documents_list = self.filter_homepage_documents(self.query_handler, file_name,logging=logging)
                        if homepage_documents_list:
                            logging.info(f"成功检索到首页文档，数量: {len(homepage_documents_list)}")
                            LLM_Evaluation.__init__(self, self.query_model_platform, self.query_model, self.model_url,
                                                    self.query_temperature,
                                                    self.query_num_predict, self.query_timeout)
                            reply = LLM_Evaluation.run(self, documents=homepage_documents_list, query=query,flag="homepage",logging=logging)
                            # 提取模型回答
                            llm_answer = reply["reply"]["replies"][0]
                            logging.info(f"LLM评估完成，提取模型回答: {llm_answer}")
                            source_start_index = llm_answer.find("Source: ")

                            # 检查"Source: "是否存在于回答中
                            if source_start_index != -1:
                                # Extracting the content for Answer and Source
                                answer_content = llm_answer[len("Answer: "):source_start_index].strip()
                                source_content = llm_answer[source_start_index + len("Source: "):].strip()

                                if "yes" in answer_content.lower():
                                    # 检查source_content是否为空
                                    if source_content:
                                        logging.info(f"解析来源内容: {source_content}")
                                        source_list = source_content.strip("[]").split(", Document")
                                        for source in source_list:
                                            # 找到第一个和最后一个双引号的位置
                                            first_quote_index = source.find('"')
                                            last_quote_index = source.rfind('"')
                                            # 提取两个双引号之间的内容
                                            if first_quote_index != -1 and last_quote_index != -1:
                                                content_between_quotes = source[
                                                                         first_quote_index + 1:last_quote_index]
                                                homepage_documents.append(content_between_quotes)
                                                logging.info(
                                                    f"提取到来源内容并添加到首页文档: {content_between_quotes}")
                                    else:
                                        # Handle the case where source_content is empty
                                        logging.info(f"No source content available.")
                        else:
                            logging.info(f"该文件没有首页信息!")
                    logging.info("*****结束首页问题策略*****")
                with LoggedTime("检索模块耗时统计",logging) as lt_retrieval:
                    if self.multiRouteRetrieval:
                        response = pipeline.run(
                            {
                                "query": {"text": query_new},
                                "word_filter_2": {"query": query},
                                "textEbd_embedding_retriever_1": {"milvus_filename": self.milvus_filename,"knowledgeUUID":knowledgeUUID,"query_hybrid_handler":self.query_hybrid_handler,"parseType":parseType,"logging":logging},
                                "textEbd_embedding_retriever_2": {"milvus_filename": self.milvus_filename_2,"knowledgeUUID":knowledgeUUID,"query_hybrid_handler":self.query_hybrid_handler,"parseType":parseType,"logging":logging}
                            },
                            include_outputs_from=["docs_for_generation"]
                        )
                    else:
                        response = pipeline.run(
                            {
                                "query": {"text": query_new},
                                "textEbd_embedding_retriever": {"milvus_filename": self.milvus_filename,"knowledgeUUID":knowledgeUUID,"query_hybrid_handler":self.query_hybrid_handler,"parseType":parseType,"logging":logging}
                            },
                            include_outputs_from=["docs_for_generation"]
                        )
                # 添加首页信息
                if homepage_documents:
                    logging.info(f"*****开始添加首页信息！*****")
                    # chunk内容的还原
                    matched_blocks = self.homepage_info_retrieval(self.query_handler, file_name, homepage_documents,logging=logging)
                    if response:
                        # 获取列表
                        documents_insert = response['docs_for_generation']['documents']
                    else:
                        documents_insert = []
                    if documents_insert and matched_blocks:
                        logging.info(f"准备插入 {len(matched_blocks)} 条首页文档信息")
                        # 复制第一个元素matched_blocks长度的次数
                        first_element_copies = [copy.deepcopy(documents_insert[0]) for _ in range(len(matched_blocks))]
                        # 将复制的元素插入到列表的第一个位置
                        documents_insert[:0] = first_element_copies

                        for i in range(len(matched_blocks)):
                            documents_insert[i].content = matched_blocks[i]
                            documents_insert[i].score = 1
                            documents_insert[i].meta["metadata"]["page_idx"] = 0
                            logging.debug(f"已插入首页文档内容: {matched_blocks[i][:10]}...")  # 只记录前10个字符以避免日志过长
                        logging.info(f"*****结束添加首页信息！*****")
                    else:
                        if not matched_blocks:
                            logging.warning("未找到匹配的首页文档块，跳过插入操作")
                        if not documents_insert:
                            logging.warning("检索结果为空，无法插入首页文档信息")
                        logging.info(f"*****结束添加首页信息！*****")
                    # 更新响应字典
                    response['docs_for_generation']['documents'] = documents_insert

                time_costs["rag_retrieval"] = lt_retrieval.get_elapsed_time()
                documents = response["docs_for_generation"]["documents"]
                break
            elif self.first_retrieval_method == "simple_keyword":
                documents = self.simple_keyword(query=query_new)
                break
            elif self.first_retrieval_method == "graphRAG":
                documents = self.graphRAG(query=query_new)
                break
        # LLM后处理精排结果
        # use_llm_similarity=False
        # if use_llm_similarity:
        #     from retriever.src.haystack_utils import LLM_Evaluation
        #
        #     LLM_Evaluation.__init__(self, self.query_model_platform, self.query_model, self.model_url,
        #                         self.query_temperature,
        #                         self.query_num_predict, self.query_timeout)
        #     reply = LLM_Evaluation.run(self, documents=documents, query=query,flag="similarity")
        #     # 提取模型回答
        #     llm_answer = reply["reply"]["replies"][0]
        #     logging.info(f"LLM评估完成，提取模型回答: {llm_answer}")
        #     documents = self.filter_documents_by_llm_judgment(documents, llm_answer)

        # 切分knowledgeUUID
        knowledgeUUID_list = self.split_knowledge_uuid(knowledge_uuid_str=knowledgeUUID)
        # 定义需要跳过表格复原和拼接上下文的知识库UUID列表
        SKIP_KNOWLEDGE_UUIDS = {"2003026697833705474"}

        if self.use_table_recovery:
            from retriever.src.haystack_utils import TableRecoveryProcessor
            logging.info(f"*****开启表格复原策略！*****")
            processor = TableRecoveryProcessor()
            with LoggedTime("表格复原模块耗时统计",logging) as lt_table:
                if self.retrieval_fetchType == "api":
                    documents = processor.process_documents(documents, self.query_filter_handler,self.retrieval_fetchType,knowledgeUUID_list, parseType,logging=logging, skip_knowledge_uuids=SKIP_KNOWLEDGE_UUIDS)
                else:
                    documents = processor.process_documents(documents, self.query_handler,self.retrieval_fetchType,knowledgeUUID_list,parseType,logging=logging, skip_knowledge_uuids=SKIP_KNOWLEDGE_UUIDS)
            time_costs["table_recovery"] = lt_table.get_elapsed_time()
            logging.info(f"*****结束表格复原策略！*****")
        if self.use_concat_context:
            logging.info(f"*****开启拼接上下文策略！*****")
            with LoggedTime("拼接上下文策略耗时统计",logging) as lt_concat:
                if self.retrieval_fetchType == "api":
                    documents = self.enhanced_info_retrieval(self.query_filter_handler, documents, self.doc_score,self.retrieval_fetchType,knowledgeUUID_list,parseType,
                                                             logging=logging, skip_knowledge_uuids=SKIP_KNOWLEDGE_UUIDS)
                else:
                    documents = self.enhanced_info_retrieval(self.query_handler, documents, self.doc_score,self.retrieval_fetchType,knowledgeUUID_list,
                                                             parseType,logging=logging, skip_knowledge_uuids=SKIP_KNOWLEDGE_UUIDS)
            time_costs["concat_context"] = lt_concat.get_elapsed_time()
            logging.info(f"*****结束拼接上下文策略！*****")
        # metadata中插入新的问题
        if documents:
            documents = self.insert_new_query(query_new, file_name, documents,logging=logging)
        # 控制检索输出的字符长度
        content_list_text = ""
        documents_list = []
        for doc in documents:
            # logging.info(f"文档chunk长度：{len(doc.content)}")
            # content_list_text += doc.content
            # if self.calculate_tokens(content_list_text) > 13000:
            #     logging.info(f"检索chunk总长度超过13000，截断")
            #     # 如果是第一个文档就超过13000，截取前13000长度的内容
            #     if len(documents_list) == 0:
            #         # 反向计算需要保留的内容长度
            #         current_tokens = self.calculate_tokens(content_list_text)
            #         excess_tokens = current_tokens - 13000
            #         # 估算需要截断的字符数（粗略估计：1 token ≈ 3-4 字符，这里取保守值3）
            #         truncate_chars = excess_tokens * 3
            #         if truncate_chars < len(doc.content):
            #             doc.content = doc.content[:-truncate_chars]
            #         else:
            #             doc.content = doc.content[:len(doc.content) // 2]
            #         documents_list.append(doc)
            #     break
            documents_list.append(doc)

        # LLM Wiki 独立检索 + 统一 RRF + 外部 rerank
        if use_wiki and self.wiki_searcher:
            try:
                # Allow per-request override of the expansion threshold
                if wiki_expansion_min_relevance is not None:
                    original_threshold = self.wiki_searcher.expansion_min_relevance
                    self.wiki_searcher.expansion_min_relevance = wiki_expansion_min_relevance
                with LoggedTime("Wiki独立检索耗时统计", logging) as lt_wiki_search:
                    wiki_docs = self.wiki_searcher.search(query)
                if wiki_expansion_min_relevance is not None:
                    self.wiki_searcher.expansion_min_relevance = original_threshold
                time_costs["wiki_search"] = lt_wiki_search.get_elapsed_time()
                logging.info(f"Wiki 独立检索返回 {len(wiki_docs)} 条结果")

                with LoggedTime("RRF融合耗时统计", logging) as lt_rrf:
                    from llm_wiki.wiki_fusion import reciprocal_rank_fusion
                    documents_list = reciprocal_rank_fusion(
                        rag_docs=documents_list,
                        wiki_docs=wiki_docs,
                        top_k=None,  # 保留全部，让 reranker 做语义筛选
                    )
                time_costs["rrf_fusion"] = lt_rrf.get_elapsed_time()
                logging.info(f"RRF 融合后共 {len(documents_list)} 条结果")

                if self.use_reranker:
                    with LoggedTime("外部reranker耗时统计", logging) as lt_rerank:
                        reranker_instance = self._build_reranker_instance()
                        # OpenAIReranker accepts a logging kwarg; FastAPIRanker/Transformers do not.
                        from retriever.src.haystack_utils import OpenAIReranker
                        if isinstance(reranker_instance, OpenAIReranker):
                            rerank_result = reranker_instance.run(
                                query=query,
                                documents=documents_list,
                                logging=logging,
                            )
                        else:
                            rerank_result = reranker_instance.run(
                                query=query,
                                documents=documents_list,
                            )
                        documents_list = rerank_result.get("documents", documents_list)
                    time_costs["external_rerank"] = lt_rerank.get_elapsed_time()
                    logging.info(f"外部 rerank 后共 {len(documents_list)} 条结果")

                # 统一按 retain_count 截断，保证最终返回数量可控
                if self.retain_count and len(documents_list) > self.retain_count:
                    documents_list = documents_list[:self.retain_count]
                    logging.info(f"按 retain_count={self.retain_count} 截断后共 {len(documents_list)} 条结果")
            except Exception as e:
                logging.warning(f"Wiki 独立检索或 RRF 融合失败，返回原始结果: {e}")

        # 纯 RAG 模式下也走外部 reranker（当启用 reranker 时）
        if not use_wiki and self.use_reranker:
            try:
                with LoggedTime("外部reranker耗时统计", logging) as lt_rerank:
                    reranker_instance = self._build_reranker_instance()
                    from retriever.src.haystack_utils import OpenAIReranker
                    if isinstance(reranker_instance, OpenAIReranker):
                        rerank_result = reranker_instance.run(
                            query=query, documents=documents_list, logging=logging,
                        )
                    else:
                        rerank_result = reranker_instance.run(
                            query=query, documents=documents_list,
                        )
                    documents_list = rerank_result.get("documents", documents_list)
                time_costs["external_rerank"] = lt_rerank.get_elapsed_time()
                logging.info(f"外部 rerank 后共 {len(documents_list)} 条结果")
            except Exception as e:
                logging.warning(f"纯RAG模式下外部reranker调用失败，返回原始结果: {e}")

        # 纯 RAG 模式下（未走 wiki 分支）也需要按 retain_count 截断
        if self.retain_count and len(documents_list) > self.retain_count:
            documents_list = documents_list[:self.retain_count]
            logging.info(f"按 retain_count={self.retain_count} 截断后共 {len(documents_list)} 条结果")

        # documents转json形式，过滤embedding向量
        documents_list = self.documents_to_json(documents_list)

        return {"documents": documents_list, "time_costs": time_costs}

@component
class Retriever_Recall_Module:
    """
  检索器召回率统计模块，提供对外接口

  输入：查询问题列表 query:list,检索方法配置文件 retrieval_config: dict, 标注的上下文 content: list
  输出：问题列表对应问题的召回率列表 query_recall
  """
    def calculate_tokens(self,text: str):
        """计算单个文本的token数"""
        if not text or not isinstance(text, str):
            return 0
        # 使用tiktoken库计算tokens
        import tiktoken
        # 获取编码器
        encoding = tiktoken.get_encoding("cl100k_base")
        # 计算prompt tokens
        prompt_tokens = len(encoding.encode(text))
        return prompt_tokens
    # 定义计算召回率的函数
    def calculate_recall(self, query: list, file_data: dict, data_list_all: list):
        filename_uid_list=[]
        try:
            for item in file_data:
                filename_uid_list_ = []
                # 多出处映射还原
                if "multi_source_answer" in item:
                    for i in range(len(item["unique_ids"])):
                        key = str(item["unique_ids"][i])
                        if key in item["multi_source_answer"]:
                            item["unique_ids"][i] = item["multi_source_answer"][key]
                # 遍历列表中的每个元素，如果不是列表，就转换为列表
                item["unique_ids"] = [x if isinstance(x, list) else [x] for x in item["unique_ids"]]

                for uid_list in item["unique_ids"]:
                    u_list=[]
                    for uid in uid_list:
                        u_list.append(item["filename"]+"_"+str(uid))
                    filename_uid_list_.append(u_list)
                filename_uid_list.append(filename_uid_list_)
        except KeyError as e:
            logging.error(f"字段缺失错误：file_data中的某些条目缺少'unique_ids'或'filename'字段，错误详情: {str(e)}")
            raise ValueError("标注文件中的某些条目缺少'unique_ids'或'filename'字段")
        # 输入验证
        if not all(isinstance(lst, list) for lst in [query, filename_uid_list, data_list_all]):
            logging.error("输入参数类型错误：所有输入参数都应该是列表类型")
            raise ValueError("所有输入参数都应该是列表类型")

        if len(query) != len(filename_uid_list) or len(query) != len(data_list_all):
            logging.error("输入参数长度不匹配：输入列表长度不一致")
            raise ValueError("输入列表长度不一致")

        logging.info("开始计算召回率...")
        results = []  # 存储详细结果
        total_correct = 0

        # 预处理检索结果
        logging.info("预处理检索结果...")
        filename_query_uid_list_ = []
        try:
            for query_doc in data_list_all:
                query_filename_uid_list = []
                # 合块映射还原 仅限于unique_id类型不是list的(是list说明检索策略已经扩充过，且已经映射过),三种情况1.带ori_id_1字段且不为空 2.带ori_id_1字段但为空(只会第一个是) 3. 不带ori_id_1字段(excel)
                for doc in query_doc:
                    content_filename_uid_list=[]
                    if isinstance(doc.meta["metadata"]["unique_id"], list):
                        file_name = doc.meta["metadata"]["file_name"]
                        # content_filename_uid_list = [file_name + "_" + str(x) for x in
                        #                              doc.meta["metadata"]["unique_id"]]
                        content_filename_uid_list = [file_name + "_" + str(uid)
                                                     for uid_list in doc.meta["metadata"]["unique_id"]
                                                     for uid in uid_list]

                    else:
                        if "ori_id_1" in doc.meta["metadata"]:
                            if doc.meta["metadata"]["ori_id_1"]:
                                file_name = doc.meta["metadata"]["file_name"]
                                unique_id = doc.meta["metadata"]["unique_id"]
                                uid_sum_list = doc.meta["metadata"]["ori_id_1"]
                                doc.meta["metadata"]["unique_id"] = uid_sum_list
                                content_filename_uid_list = [file_name + "_" + str(x) for x in doc.meta["metadata"]["unique_id"]]
                                logging.info(
                                    f"正在合块映射文档{file_name}的uid为{unique_id}的ori_id_1字段,其ori_id_1为{uid_sum_list}")
                            else:
                                file_name = doc.meta["metadata"]["file_name"]
                                unique_id = doc.meta["metadata"]["unique_id"]
                                content_filename_uid_list.append(file_name + "_" + str(unique_id))
                                logging.info(
                                    f"正在合块映射文档{file_name}的uid为{unique_id}的ori_id_1字段,其ori_id_1为空列表")
                        else:
                            file_name = doc.meta["metadata"]["file_name"]
                            unique_id = doc.meta["metadata"]["unique_id"]
                            content_filename_uid_list.append(file_name + "_" + str(unique_id))
                    query_filename_uid_list.append(content_filename_uid_list)
                # 将所有元素展开成一维列表
                query_filename_uid_list = [item for sublist in query_filename_uid_list for item in sublist]
                # 去重
                query_filename_uid_list = list(set(query_filename_uid_list))
                filename_query_uid_list_.append(query_filename_uid_list)

        except KeyError as e:
            logging.error(f"召回的chunk中的某些条目拼接'unique_id'和'file_name'字段错误，错误详情: {str(e)}")
            raise ValueError("召回的chunk中的某些条目拼接'unique_id'和'file_name'字段错误")
        # 遍历每个问题的召回结果
        logging.info(f"开始遍历 {len(filename_uid_list)} 个问题的召回结果...")
        for idx, uid_ in enumerate(filename_uid_list):
            # 检查索引是否越界
            if idx >= len(filename_query_uid_list_):
                logging.warning(f"索引 {idx} 超出范围，跳过处理")
                continue

            query_uid = filename_query_uid_list_[idx]

            # 如果 content_ 或 query_contents 为空，则视为错误（不匹配）
            if not uid_ or not query_uid:
                logging.warning(f"问题 {idx} 的filename_uid为空，跳过处理")
                continue
            # 召回率计算，标注uid_列表的元素u，u中任意元素包含在query_uid中，即表示该标注点召回成功
            correct_result = [any(u_item in query_uid for u_item in u) for u in uid_]
            correct = all(correct_result)
            if correct:
                total_correct += 1
                logging.debug(f"问题 {idx} 召回成功")
            else:
                logging.debug(f"问题 {idx} 召回失败")

            # 收集当前问题的详细信息
            answer_details = []
            try:
                for i, doc in enumerate(data_list_all[idx]):
                    try:
                        answer_details.append({
                            "content": doc.content if hasattr(doc, 'content') else '无内容',
                            "meta": doc.meta if hasattr(doc, 'meta') else {},
                            "score": doc.score if hasattr(doc, 'score') else 0.0
                        })
                    except Exception as e:
                        logging.error(f"处理问题 {idx} 的第 {i} 个文档时出错: {str(e)}")
                        continue
            except Exception as e:
                logging.error(f"获取问题 {idx} 的答案详情时出错: {str(e)}")

            # 存储到结果集
            results.append({
                "query": query[idx] if idx < len(query) else '未知问题',
                "is_recalled": "是" if correct else "否",
                "answers": answer_details
            })
            logging.info(f"已处理问题 {idx}/{len(filename_uid_list) - 1}")

        recall = total_correct / len(filename_uid_list)
        logging.info(f"召回率计算完成。总问题数: {len(filename_uid_list)}, 正确召回数: {total_correct}, 召回率: {recall:.2f}")
        return recall, results

    @component.output_types(recall=float, results=List[dict],time_used=List[float])
    def run(self, retrieval_config: str,file_data: dict,logging=None,knowledgeUUID: str = None,parseType=None) -> (float, List[dict],List[float]):
        RM = RetrieverModule()
        RM.init_system(logging=logging)

        data_list_all = []
        total_time = 0  # 初始化总耗时
        time_used = []
        try:
            query= [item["query"] for item in file_data]
        except KeyError as e:
            logging.error(f"字段缺失错误：标注文件中的某些条目缺少'query'字段，错误详情: {str(e)}")
            raise ValueError("标注文件中的某些条目缺少'query'字段")
        # 提取召回中的内容
        for query_ in query:

            data_list= []
            # 记录开始时间
            start_time = time.time()

            docs = RM.run(query=query_, retrieval_config=retrieval_config, logging=logging,knowledgeUUID=knowledgeUUID, parseType=parseType)

            # 计算单次查询耗时
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # 记录单次查询耗时
            logging.info(f"问题: '{query_}' 检索耗时: {elapsed_time:.4f}秒")
            print(f"问题: '{query_}' 检索耗时: {elapsed_time:.4f}秒")
            time_used.append(elapsed_time)
            data_list = docs["documents"].copy()
            data_list_all.append(data_list)

        # 计算平均耗时
        avg_time = total_time / len(query) if query else 0
        logging.info(f"*****检索耗时统计*****")
        logging.info(f"总查询数量: {len(query)}个")
        logging.info(f"总检索耗时: {total_time:.4f}秒")
        logging.info(f"平均每个问题检索耗时: {avg_time:.4f}秒")
        logging.info("*****进入召回率计算模块*****")

        recall,results = self.calculate_recall(query=query, file_data=file_data,
                                       data_list_all=data_list_all)
        logging.info("*****结束召回率计算模块*****")
        return recall,results,time_used


@component
class Knowledge_Base:
    """
     知识库构建模块，提供对外接口
     一个json文件一次性构建四个知识库，分别对应indexing处理的四种情况，备后续检索直接调用。
     输入：json文件地址 file_path: str,构建索引配置文件 retrieval_config: dict
     输出：知识库构建成功的chunk条数 success_length: list[int]
     """

    def __init__(self):
        self.sparse_timeout = None
        self.sparse_embedder_model_url = None
        self.file_name = None
        self.results = None
        self.llm_enrich_model = None
        self.llm_enrich_model_platform=None
        self.llm_enrich_temperature = None
        self.llm_enrich_num_predict = None
        self.llm_enrich_timeout = None
        self.llm_enrich_model_url = None
        self.indexing_method = None
        self.db_path_url = None
        self.milvus_db_name = None
        self.milvus_collection_name = None
        self.embedder_url = None
        self.embedder_timeout = None
        self.embedder_model = None
        self.embedder_model_platform = None
        self.sparse_embedder_model_platform = None
        self.sparse_embedding_model = None
        self.indexing_methods = None
        self.query_handler = None
        self.default_config_path = "configs/retriever_config_v3.json"
        # LLM Wiki config
        self.use_llm_wiki = False
        self.wiki_config_path = None
        self.wiki_llm_config = None
        load_dotenv()

    def change_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def init_system(self,retrieval_config: dict = None,logging=None):
        PROJECT_ROOT = find_project_root()
        os.chdir(PROJECT_ROOT)
        #
        logging.info(f"切换到项目根目录：{PROJECT_ROOT}")
        # 创建日志目录（如果不存在）
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"Knowledge_Base_{timestamp}.log")

        # # 配置日志记录到文件
        # logging.basicConfig(
        #     force=True,
        #     level=logging.INFO,
        #     format='%(asctime)s - %(levelname)s - %(message)s',
        #     handlers=[
        #         logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        #         # logging.StreamHandler()  # 同时保留控制台输出
        #     ]
        # )

        # 没提供则使用默认配置
        if retrieval_config is None:
            retrieval_config = self.read_json_file(self.default_config_path)
            logging.info("未提供检索配置，使用默认配置文件")
        else:
            retrieval_config = self.read_json_file(retrieval_config)
            logging.info("已加载用户提供的检索配置")

        # 加载LLM丰富化模型相关配置路径
        logging.info("加载LLM丰富化模型相关配置路径")
        llm_enrich_model_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                      "1005.llm_enrich", "1001.value"]
        llm_enrich_model_platform_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                               "1005.llm_enrich", "1005.detail", "1006.model_platform",
                                               "1001.value"]
        llm_enrich_temperature_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                            "1005.llm_enrich", "1005.detail", "1000.temperature", "1001.value"]
        llm_enrich_num_predict_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                            "1005.llm_enrich", "1005.detail", "1001.num_predict", "1001.value"]
        llm_enrich_timeout_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                        "1005.llm_enrich", "1005.detail", "1005.timeout", "1001.value"]
        llm_enrich_model_url_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                          "1005.llm_enrich", "1005.detail", "1004.llm_enrich_model_url",
                                          "1001.value"]

        # 索引配置路径
        logging.info("加载索引相关配置路径")
        indexing_method_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                     "1000.indexing", "1001.value"]
        db_path_url_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                            "1005.detail",
                            "1000.indexing", "1005.detail",
                            "1001.db_path_url", "1001.value"]
        file_name_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                               "1005.detail",
                               "1000.indexing", "1005.detail",
                               "1003.milvus_collection_name", "1001.value"]
        milvus_db_name_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                               "1005.detail",
                               "1000.indexing", "1005.detail",
                               "1002.milvus_db_name", "1001.value"]
        milvus_collection_name_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                       "1005.detail",
                                       "1000.indexing", "1005.detail",
                                       "1003.milvus_collection_name", "1001.value"]

        # Embedder相关配置路径
        logging.info("加载Embedder相关配置路径")
        embedder_url_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                             "1002.retrieval",
                             "1005.detail", "1000.embedding_retriever", "1005.detail",
                             "1000.embedding_model", "1005.detail", "1001.embedder_model_url",
                             "1001.value"]
        embedder_timeout_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                 "1002.retrieval",
                                 "1005.detail", "1000.embedding_retriever", "1005.detail",
                                 "1000.embedding_model", "1005.detail", "1003.timeout",
                                 "1001.value"]
        sparse_embedder_model_url_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                               "1002.retrieval",
                                               "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                               "1000.sparse_embedding_model", "1005.detail",
                                               "1000.sparse_embedder_model_url",
                                               "1001.value"]
        sparse_timeout_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                    "1002.retrieval",
                                    "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                    "1000.sparse_embedding_model", "1005.detail", "1002.sparse_timeout",
                                    "1001.value"]
        # 读取LLM丰富化模型配置值
        logging.info("读取LLM丰富化模型配置值")
        self.llm_enrich_model = self.get_value_from_key_path(retrieval_config,
                                                        llm_enrich_model_path_list)
        logging.info(f"llm_enrich_model：{self.llm_enrich_model}")
        self.llm_enrich_model_platform = self.get_value_from_key_path(retrieval_config,
                                                                 llm_enrich_model_platform_path_list)
        logging.info(f"llm_enrich_model_platform：{self.llm_enrich_model_platform}")
        self.llm_enrich_temperature = self.get_value_from_key_path(retrieval_config,
                                                              llm_enrich_temperature_path_list)
        logging.info(f"llm_enrich_temperature：{self.llm_enrich_temperature}")
        self.llm_enrich_num_predict = self.get_value_from_key_path(retrieval_config,
                                                              llm_enrich_num_predict_path_list)
        logging.info(f"llm_enrich_num_predict：{self.llm_enrich_num_predict}")
        self.llm_enrich_timeout = self.get_value_from_key_path(retrieval_config,
                                                          llm_enrich_timeout_path_list)
        logging.info(f"llm_enrich_timeout：{self.llm_enrich_timeout}")
        self.llm_enrich_model_url = self.get_value_from_key_path(retrieval_config,
                                                            llm_enrich_model_url_path_list)
        logging.info(f"llm_enrich_model_url：{self.llm_enrich_model_url}")

        # 读取索引相关配置值
        logging.info("读取索引相关配置值")
        self.indexing_method = self.get_value_from_key_path(retrieval_config,
                                                       indexing_method_path_list)
        logging.info(f"indexing_method：{self.indexing_method}")
        self.db_path_url = self.get_value_from_key_path(retrieval_config,
                                                   db_path_url_list)
        logging.info(f"db_path_url：{self.db_path_url}")

        self.file_name = self.get_value_from_key_path(retrieval_config,
                                                           file_name_list)
        logging.info(f"milvus_db_name：{self.milvus_db_name}")

        self.milvus_db_name = self.get_value_from_key_path(retrieval_config,
                                                      milvus_db_name_list)
        logging.info(f"milvus_db_name：{self.milvus_db_name}")
        self.milvus_collection_name = self.get_value_from_key_path(retrieval_config,
                                                              milvus_collection_name_list)
        logging.info(f"milvus_collection_name：{self.milvus_collection_name}")
        
        # 读取Embedder相关配置值
        logging.info("读取Embedder相关配置值")
        self.embedder_url = self.get_value_from_key_path(retrieval_config,
                                                    embedder_url_list)
        logging.info(f"embedder_url：{self.embedder_url}")
        self.embedder_timeout = self.get_value_from_key_path(retrieval_config,
                                                        embedder_timeout_list)
        logging.info(f"embedder_timeout：{self.embedder_timeout}")
        self.sparse_embedder_model_url = self.get_value_from_key_path(retrieval_config,
                                                                      sparse_embedder_model_url_path_list)
        logging.info(f"sparse_embedder_model_url：{self.sparse_embedder_model_url}")
        self.sparse_timeout = self.get_value_from_key_path(retrieval_config,
                                                           sparse_timeout_path_list)
        logging.info(f"sparse_timeout：{self.sparse_timeout}")

        # 加载嵌入器配置路径
        logging.info("加载嵌入器相关配置路径")
        embedder_model_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                    "1002.retrieval",
                                    "1005.detail", "1000.embedding_retriever", "1005.detail",
                                    "1000.embedding_model",
                                    "1001.value"]
        embedder_model_platform_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                             "1002.retrieval",
                                             "1005.detail", "1000.embedding_retriever", "1005.detail",
                                             "1000.embedding_model", "1005.detail",
                                             "1002.embedder_model_platform",
                                             "1001.value"]
        sparse_embedding_model_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                            "1002.retrieval",
                                            "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                            "1000.sparse_embedding_model",
                                            "1001.value"]
        sparse_embedder_model_platform_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                                    "1002.retrieval",
                                                    "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                                    "1000.sparse_embedding_model", "1005.detail",
                                                    "1001.sparse_embedder_model_platform",
                                                    "1001.value"]
        # 读取嵌入器配置值
        logging.info("读取嵌入器配置值")
        self.embedder_model = self.get_value_from_key_path(retrieval_config,
                                                      embedder_model_path_list)
        logging.info(f"embedder_model：{self.embedder_model}")
        self.embedder_model_platform = self.get_value_from_key_path(retrieval_config,
                                                               embedder_model_platform_path_list)
        logging.info(f"embedder_model_platform：{self.embedder_model_platform}")
        self.sparse_embedding_model = self.get_value_from_key_path(retrieval_config,
                                                              sparse_embedding_model_path_list)
        logging.info(f"sparse_embedding_model：{self.sparse_embedding_model}")
        self.sparse_embedder_model_platform = self.get_value_from_key_path(retrieval_config,
                                                                           sparse_embedder_model_platform_path_list)
        logging.info(f"sparse_embedder_model_platform：{self.sparse_embedder_model_platform}")

        self.query_handler = MilvusQueryHandler(
            uri=self.db_path_url,
            db_name=self.milvus_db_name,
            collection_name=self.milvus_collection_name,
            logging=logging
        )

        self.results=instantiate(self.modular_rag,values={
                "indexing.indexing_method": self.indexing_method,
                "indexing.model": self.llm_enrich_model,
                "indexing.model_platform": self.llm_enrich_model_platform,
                "indexing.llm_enrich_model_url": self.llm_enrich_model_url,
                "indexing.temperature": self.llm_enrich_temperature,
                "indexing.num_predict": self.llm_enrich_num_predict,
                "indexing.timeout": self.llm_enrich_timeout,
                "indexing.db_name": self.file_name,
                "indexing.milvus_db_name": self.milvus_db_name,
                "indexing.db_path_url": self.db_path_url,
                "indexing.embedder_model": self.embedder_model,
                "indexing.embedder_model_platform": self.embedder_model_platform,
                "indexing.embedder_sparse_model": self.sparse_embedding_model,
                "indexing.sparse_embedder_model_platform": self.sparse_embedder_model_platform,
                "indexing.embedder_url": self.embedder_url,
                "indexing.embedder_timeout": self.embedder_timeout,
                "indexing.sparse_embedder_model_url": self.sparse_embedder_model_url,
                "indexing.sparse_timeout": self.sparse_timeout

            })
        # LLM Wiki 配置读取
        try:
            llm_wiki_path_list = ["1007.llm_wiki", "1005.detail"]
            llm_wiki_detail = self.get_value_from_key_path(retrieval_config, llm_wiki_path_list)
            use_llm_wiki_path = ["1001.use_llm_wiki", "1001.value"]
            wiki_config_path_path = ["1002.wiki_config_path", "1001.value"]
            self.use_llm_wiki = self.change_bool(self.get_value_from_key_path(llm_wiki_detail, use_llm_wiki_path))
            self.wiki_config_path = self.get_value_from_key_path(llm_wiki_detail, wiki_config_path_path)
            if self.use_llm_wiki and self.wiki_config_path:
                wiki_cfg = self.read_json_file(self.wiki_config_path)
                llm_cfg = wiki_cfg.get("llm_config", {})
                self.wiki_llm_config = LlmConfig(
                    provider=llm_cfg.get("provider", "openai"),
                    api_key=llm_cfg.get("api_key", ""),
                    model=llm_cfg.get("model", "gpt-4o"),
                    base_url=llm_cfg.get("base_url", ""),
                    max_context_size=llm_cfg.get("max_context_size", 204800),
                    temperature=llm_cfg.get("temperature", 0.1),
                )
            logging.info(f"Knowledge_Base LLM Wiki配置: use_llm_wiki={self.use_llm_wiki}, config_path={self.wiki_config_path}")
        except Exception as e:
            logging.warning(f"Knowledge_Base LLM Wiki配置读取失败: {e}")

        indexing_pipeline = self.results["indexing_pipeline"]
        indexing_pipeline.warm_up()
        logging.info("索引管道已初始化并预热")


    def modular_rag(self,hp: HP):
        from retriever.configs.indexing import indexing_config
        indexing = hp.nest(indexing_config,name="indexing")
        indexing_pipeline = indexing["pipeline"]
        return {"indexing_pipeline": indexing_pipeline}

    def read_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    def get_value_from_key_path(self, data_dict, key_path_list):
        """
        通过键路径列表获取值
        :param data_dict: 要查询的字典
        :param key_path_list: 键路径列表，例如 ["1st_retriever", "1005.detail", "general", "1000.param_meaning"]
        :return: 对应的值
        """
        current_data = data_dict
        for key in key_path_list:
            current_data = current_data[key]
        return current_data
    @component.output_types(creation=bool)
    def run(self, file_path: str, retrieval_config: dict = None,logging=None):
        # 没提供则使用默认配置
        if retrieval_config is None:
            retrieval_config = self.read_json_file(self.default_config_path)
            logging.info("未提供检索配置，使用默认配置文件")
        else:
            retrieval_config = self.read_json_file(retrieval_config)
            logging.info("已加载用户提供的检索配置")
        messages = []  # 用于存储每个文件处理后的消息
        elapsed_times = []  # 用于存储每个文件的向量化耗时
        # 检查file_path是否是json文件
        if os.path.isfile(file_path) and file_path.endswith('.json'):
            file_paths = [file_path]
            logging.info(f"检测到单个JSON文件：{file_path}")
        elif os.path.isdir(file_path):
            file_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.json')]
            logging.info(f"检测到文件夹路径，找到 {len(file_paths)} 个 JSON 文件")
        else:
            error_message = "提供的路径既不是JSON文件也不是文件夹路径!"
            logging.error(error_message)
            raise ValueError(error_message)
        success_length_list = []
        for file_path in file_paths:
            logging.info(f"开始处理JSON文件：{file_path}")
            # 从文件开头读取第一个样本获取file_name，避免加载大文件
            original_name = None
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    # 读取文件开头直到找到第一个样本的file_name
                    content = file.read(4096)  # 读取前4KB，通常足够包含第一个样本
                    # 解析第一个样本的file_name字段
                    import re
                    match = re.search(r'"file_name"\s*:\s*"([^"]+)"', content)
                    if match:
                        original_name = match.group(1)
                        logging.info(f"从文件开头获取file_name：{original_name}")
                    else:
                        raise ValueError("未找到file_name字段")
            except Exception as e:
                # 回退到使用文件名（去除.json后缀）
                file_path_new = os.path.basename(file_path)
                original_name = file_path_new.replace(".json", "")
                logging.warning(f"从文件获取file_name失败({e})，回退使用文件名：{original_name}")

            file_name = self.milvus_collection_name
            logging.info(f"插入向量库的表名：{file_name}")
            is_filename_empty = self.query_handler.is_filename_empty(original_name)
            is_collection_empty = self.query_handler.is_collection_empty()
            logging.info(f"表{file_name}是否为空，{is_collection_empty}")
            logging.info(f"文件{original_name}是否为空，{is_filename_empty}")

            if is_collection_empty or is_filename_empty:
                indexing_pipeline = self.results["indexing_pipeline"]
                # 读取JSON文件内容
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                # 获取最后一个元素的 page_idx 值
                page_idx=None
                if data:  # 确保列表不为空
                    last_element = data[-1]
                    page_idx = last_element.get('page_idx')
                    if page_idx is not None:
                        logging.info(f"成功加载JSON文件：{file_path}，共{page_idx+1}页，共包含 {len(data)} 条数据")
                    else:
                        logging.info(f"成功加载JSON文件：{file_path}，无页码信息，共包含 {len(data)} 条数据")
                else:
                    logging.warning("JSON文件中没有数据")
                # 过滤掉那些没有'text'键或'text'值为空的字典
                data = [item for item in data if item.get('text')]
                logging.info(f"过滤没有'text'键或'text'值为空的字典后剩余有效数据 {len(data)} 条")
                sources = []
                for item in data:
                    source = ByteStream.from_string(json.dumps(item))
                    sources.append(source)
                logging.info("*****进行构建索引管道*****")
                with LoggedTime("文档向量化耗时统计", logging) as timer:
                    documents_written = indexing_pipeline.run({"loader": {"sources": sources}})
                success_length = documents_written["document_writer"]["documents_written"]
                success_length_list.append(success_length)
                elapsed_time = timer.get_elapsed_time()
                elapsed_times.append(elapsed_time)

                if success_length > 0:
                    page_suffix = f"共{page_idx+1}页!" if page_idx and page_idx+1 > 0 else "无页码信息!"
                    message = f"{original_name}文件构建成功!{page_suffix}长度为{success_length}!文档向量化耗时: {elapsed_time:.2f}秒"
                    logging.info(message)
                    # LLM Wiki 构建触发
                    if self.use_llm_wiki and self.wiki_llm_config:
                        try:
                            source_content = json_chunks_to_source_content(data)
                            project_path = normalize_path(os.path.join(find_project_root(), "llm_wiki"))
                            written = auto_ingest(
                                project_path=project_path,
                                source_path=file_path,
                                llm_config=self.wiki_llm_config,
                                source_content=source_content,
                                domain=original_name,
                            )
                            logging.info(f"LLM Wiki 构建完成: {original_name}，写入 {len(written)} 个文件")
                        except Exception as e:
                            logging.error(f"LLM Wiki 构建失败: {original_name}: {e}")
                else:
                    message = f"{original_name}文件构建失败或无新增内容!文档向量化耗时: {elapsed_time:.2f}秒"
                    logging.error(message)

            else:
                message = "{}文件已经存在表{}中!".format(original_name,file_name)
                logging.warning(message)
                elapsed_times.append(None)  # 对于已存在的文件，添加None占位

            messages.append(message)
        messages_ = "\n".join(messages)
        return messages_
@component
class Documents_Insert:
    """
     知识库构建模块，提供对外接口
     一个json文件一次性构建四个知识库，分别对应indexing处理的四种情况，备后续检索直接调用。
     输入：json文件地址 file_path: str,构建索引配置文件 retrieval_config: dict
     输出：知识库构建成功的chunk条数 success_length: list[int]
     """

    def __init__(self):
        self.retrieval_fetchType = None
        self.sparse_timeout = None
        self.sparse_embedder_model_url = None
        self.file_name = None
        self.results = None
        self.llm_enrich_model = None
        self.llm_enrich_model_platform = None
        self.llm_enrich_temperature = None
        self.llm_enrich_num_predict = None
        self.llm_enrich_timeout = None
        self.llm_enrich_model_url = None
        self.indexing_method = None
        self.db_path_url = None
        self.milvus_db_name = None
        self.milvus_collection_name = None
        self.embedder_url = None
        self.embedder_timeout = None
        self.embedder_model = None
        self.embedder_model_platform = None
        self.sparse_embedder_model_platform = None
        self.sparse_embedding_model = None
        self.indexing_methods = None
        self.query_handler = None
        self.default_config_path = "configs/retriever_config_v3.json"
        # LLM Wiki config
        self.use_llm_wiki = False
        self.wiki_config_path = None
        self.wiki_llm_config = None
        load_dotenv()

    def change_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def init_system(self, retrieval_config: dict = None, logging=None):
        PROJECT_ROOT = find_project_root()
        os.chdir(PROJECT_ROOT)
        logging.info(f"切换到项目根目录：{PROJECT_ROOT}")
        # 没提供则使用默认配置
        if retrieval_config is None:
            retrieval_config = self.read_json_file(self.default_config_path)
            logging.info("未提供检索配置，使用默认配置文件")
        else:
            retrieval_config = self.read_json_file(retrieval_config)
            logging.info("已加载用户提供的检索配置")

        # 加载LLM丰富化模型相关配置路径
        logging.info("加载LLM丰富化模型相关配置路径")
        llm_enrich_model_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                      "1005.llm_enrich", "1001.value"]
        llm_enrich_model_platform_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                               "1005.llm_enrich", "1005.detail", "1006.model_platform",
                                               "1001.value"]
        llm_enrich_temperature_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                            "1005.llm_enrich", "1005.detail", "1000.temperature", "1001.value"]
        llm_enrich_num_predict_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                            "1005.llm_enrich", "1005.detail", "1001.num_predict", "1001.value"]
        llm_enrich_timeout_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                        "1005.llm_enrich", "1005.detail", "1005.timeout", "1001.value"]
        llm_enrich_model_url_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                          "1005.llm_enrich", "1005.detail", "1004.llm_enrich_model_url",
                                          "1001.value"]

        # 索引配置路径
        logging.info("加载索引相关配置路径")
        indexing_method_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                     "1000.indexing", "1001.value"]
        db_path_url_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                            "1005.detail",
                            "1000.indexing", "1005.detail",
                            "1001.db_path_url", "1001.value"]
        file_name_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                          "1005.detail",
                          "1000.indexing", "1005.detail",
                          "1003.milvus_collection_name", "1001.value"]
        milvus_db_name_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                               "1005.detail",
                               "1000.indexing", "1005.detail",
                               "1002.milvus_db_name", "1001.value"]
        milvus_collection_name_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                       "1005.detail",
                                       "1000.indexing", "1005.detail",
                                       "1003.milvus_collection_name", "1001.value"]

        # Embedder相关配置路径
        logging.info("加载Embedder相关配置路径")
        embedder_url_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                             "1002.retrieval",
                             "1005.detail", "1000.embedding_retriever", "1005.detail",
                             "1000.embedding_model", "1005.detail", "1001.embedder_model_url",
                             "1001.value"]
        embedder_timeout_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                 "1002.retrieval",
                                 "1005.detail", "1000.embedding_retriever", "1005.detail",
                                 "1000.embedding_model", "1005.detail", "1003.timeout",
                                 "1001.value"]
        sparse_embedder_model_url_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                               "1002.retrieval",
                                               "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                               "1000.sparse_embedding_model", "1005.detail",
                                               "1000.sparse_embedder_model_url",
                                               "1001.value"]
        sparse_timeout_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                    "1002.retrieval",
                                    "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                    "1000.sparse_embedding_model", "1005.detail", "1002.sparse_timeout",
                                    "1001.value"]
        # 读取LLM丰富化模型配置值
        logging.info("读取LLM丰富化模型配置值")
        self.llm_enrich_model = self.get_value_from_key_path(retrieval_config,
                                                             llm_enrich_model_path_list)
        logging.info(f"llm_enrich_model：{self.llm_enrich_model}")
        self.llm_enrich_model_platform = self.get_value_from_key_path(retrieval_config,
                                                                      llm_enrich_model_platform_path_list)
        logging.info(f"llm_enrich_model_platform：{self.llm_enrich_model_platform}")
        self.llm_enrich_temperature = self.get_value_from_key_path(retrieval_config,
                                                                   llm_enrich_temperature_path_list)
        logging.info(f"llm_enrich_temperature：{self.llm_enrich_temperature}")
        self.llm_enrich_num_predict = self.get_value_from_key_path(retrieval_config,
                                                                   llm_enrich_num_predict_path_list)
        logging.info(f"llm_enrich_num_predict：{self.llm_enrich_num_predict}")
        self.llm_enrich_timeout = self.get_value_from_key_path(retrieval_config,
                                                               llm_enrich_timeout_path_list)
        logging.info(f"llm_enrich_timeout：{self.llm_enrich_timeout}")
        self.llm_enrich_model_url = self.get_value_from_key_path(retrieval_config,
                                                                 llm_enrich_model_url_path_list)
        logging.info(f"llm_enrich_model_url：{self.llm_enrich_model_url}")

        # 读取索引相关配置值
        logging.info("读取索引相关配置值")
        self.indexing_method = self.get_value_from_key_path(retrieval_config,
                                                            indexing_method_path_list)
        logging.info(f"indexing_method：{self.indexing_method}")
        self.db_path_url = self.get_value_from_key_path(retrieval_config,
                                                        db_path_url_list)
        logging.info(f"db_path_url：{self.db_path_url}")

        self.file_name = self.get_value_from_key_path(retrieval_config,
                                                      file_name_list)
        logging.info(f"milvus_db_name：{self.milvus_db_name}")

        self.milvus_db_name = self.get_value_from_key_path(retrieval_config,
                                                           milvus_db_name_list)
        logging.info(f"milvus_db_name：{self.milvus_db_name}")
        self.milvus_collection_name = self.get_value_from_key_path(retrieval_config,
                                                                   milvus_collection_name_list)
        logging.info(f"milvus_collection_name：{self.milvus_collection_name}")

        # 读取Embedder相关配置值
        logging.info("读取Embedder相关配置值")
        self.embedder_url = self.get_value_from_key_path(retrieval_config,
                                                         embedder_url_list)
        logging.info(f"embedder_url：{self.embedder_url}")
        self.embedder_timeout = self.get_value_from_key_path(retrieval_config,
                                                             embedder_timeout_list)
        logging.info(f"embedder_timeout：{self.embedder_timeout}")
        self.sparse_embedder_model_url = self.get_value_from_key_path(retrieval_config,
                                                                      sparse_embedder_model_url_path_list)
        logging.info(f"sparse_embedder_model_url：{self.sparse_embedder_model_url}")
        self.sparse_timeout = self.get_value_from_key_path(retrieval_config,
                                                           sparse_timeout_path_list)
        logging.info(f"sparse_timeout：{self.sparse_timeout}")

        # 加载嵌入器配置路径
        logging.info("加载嵌入器相关配置路径")
        embedder_model_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                    "1002.retrieval",
                                    "1005.detail", "1000.embedding_retriever", "1005.detail",
                                    "1000.embedding_model",
                                    "1001.value"]
        embedder_model_platform_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                             "1002.retrieval",
                                             "1005.detail", "1000.embedding_retriever", "1005.detail",
                                             "1000.embedding_model", "1005.detail",
                                             "1002.embedder_model_platform",
                                             "1001.value"]
        sparse_embedding_model_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                            "1002.retrieval",
                                            "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                            "1000.sparse_embedding_model",
                                            "1001.value"]
        sparse_embedder_model_platform_path_list = ["1000.1st_retriever", "1005.detail", "1000.general", "1005.detail",
                                                    "1002.retrieval",
                                                    "1005.detail", "1001.sparse_embedding_retriever", "1005.detail",
                                                    "1000.sparse_embedding_model", "1005.detail",
                                                    "1001.sparse_embedder_model_platform",
                                                    "1001.value"]
        # 读取嵌入器配置值
        logging.info("读取嵌入器配置值")
        self.embedder_model = self.get_value_from_key_path(retrieval_config,
                                                           embedder_model_path_list)
        logging.info(f"embedder_model：{self.embedder_model}")
        self.embedder_model_platform = self.get_value_from_key_path(retrieval_config,
                                                                    embedder_model_platform_path_list)
        logging.info(f"embedder_model_platform：{self.embedder_model_platform}")
        self.sparse_embedding_model = self.get_value_from_key_path(retrieval_config,
                                                                   sparse_embedding_model_path_list)
        logging.info(f"sparse_embedding_model：{self.sparse_embedding_model}")
        self.sparse_embedder_model_platform = self.get_value_from_key_path(retrieval_config,
                                                                           sparse_embedder_model_platform_path_list)
        logging.info(f"sparse_embedder_model_platform：{self.sparse_embedder_model_platform}")
        # 加载混合检索的形式配置路径
        retrieval_fetchType_path_list = ["1000.1st_retriever", "1005.detail", "1000.general",
                                         "1005.detail", "1002.retrieval", "1005.detail",
                                         "1003.hybrid_retriever", "1005.detail", "1009.retrieval_fetchType",
                                         "1001.value"]
        # 读取混合检索的形式配置值
        self.retrieval_fetchType = self.get_value_from_key_path(retrieval_config,
                                                                retrieval_fetchType_path_list)
        logging.info(f"retrieval_fetchType：{self.retrieval_fetchType}")

        # self.query_handler = MilvusQueryHandler(
        #     uri=self.db_path_url,
        #     db_name=self.milvus_db_name,
        #     collection_name=self.milvus_collection_name,
        #     logging=logging
        # )

        self.results = instantiate(self.modular_rag, values={
            "indexing.indexing_method": self.indexing_method,
            "indexing.model": self.llm_enrich_model,
            "indexing.model_platform": self.llm_enrich_model_platform,
            "indexing.llm_enrich_model_url": self.llm_enrich_model_url,
            "indexing.temperature": self.llm_enrich_temperature,
            "indexing.num_predict": self.llm_enrich_num_predict,
            "indexing.timeout": self.llm_enrich_timeout,
            "indexing.db_name": self.file_name,
            "indexing.milvus_db_name": self.milvus_db_name,
            "indexing.db_path_url": self.db_path_url,
            "indexing.embedder_model": self.embedder_model,
            "indexing.embedder_model_platform": self.embedder_model_platform,
            "indexing.embedder_sparse_model": self.sparse_embedding_model,
            "indexing.sparse_embedder_model_platform": self.sparse_embedder_model_platform,
            "indexing.embedder_url": self.embedder_url,
            "indexing.embedder_timeout": self.embedder_timeout,
            "indexing.sparse_embedder_model_url": self.sparse_embedder_model_url,
            "indexing.sparse_timeout": self.sparse_timeout,
            "indexing.retrieval_fetchType": self.retrieval_fetchType,

        })
        # LLM Wiki 配置读取
        try:
            llm_wiki_path_list = ["1007.llm_wiki", "1005.detail"]
            llm_wiki_detail = self.get_value_from_key_path(retrieval_config, llm_wiki_path_list)
            use_llm_wiki_path = ["1001.use_llm_wiki", "1001.value"]
            wiki_config_path_path = ["1002.wiki_config_path", "1001.value"]
            self.use_llm_wiki = self.change_bool(self.get_value_from_key_path(llm_wiki_detail, use_llm_wiki_path))
            self.wiki_config_path = self.get_value_from_key_path(llm_wiki_detail, wiki_config_path_path)
            if self.use_llm_wiki and self.wiki_config_path:
                wiki_cfg = self.read_json_file(self.wiki_config_path)
                llm_cfg = wiki_cfg.get("llm_config", {})
                self.wiki_llm_config = LlmConfig(
                    provider=llm_cfg.get("provider", "openai"),
                    api_key=llm_cfg.get("api_key", ""),
                    model=llm_cfg.get("model", "gpt-4o"),
                    base_url=llm_cfg.get("base_url", ""),
                    max_context_size=llm_cfg.get("max_context_size", 204800),
                    temperature=llm_cfg.get("temperature", 0.1),
                )
            logging.info(f"Documents_Insert LLM Wiki配置: use_llm_wiki={self.use_llm_wiki}, config_path={self.wiki_config_path}")
        except Exception as e:
            logging.warning(f"Documents_Insert LLM Wiki配置读取失败: {e}")

        indexing_pipeline = self.results["indexing_pipeline"]
        indexing_pipeline.warm_up()
        logging.info("索引管道已初始化并预热")


    def modular_rag(self,hp: HP):
        from retriever.configs.indexing import indexing_config
        indexing = hp.nest(indexing_config,name="indexing")
        indexing_pipeline = indexing["pipeline"]
        return {"indexing_pipeline": indexing_pipeline}

    def read_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    def get_value_from_key_path(self, data_dict, key_path_list):
        """
        通过键路径列表获取值
        :param data_dict: 要查询的字典
        :param key_path_list: 键路径列表，例如 ["1st_retriever", "1005.detail", "general", "1000.param_meaning"]
        :return: 对应的值
        """
        current_data = data_dict
        for key in key_path_list:
            current_data = current_data[key]
        return current_data
    def documents_dict_to_json(self, insert_dict):
        """
        将文档列表转换为JSON格式字符串列表。

        Args:
            documents (list): 文档对象列表，每个文档对象包含content、content_type、id、score、embedding、meta属性。

        Returns:
            list: 包含JSON格式字符串的列表。

        """
        try:
            # 检查必要字段是否存在
            required_fields = ['file_name', 'text', 'embedding', 'id', 'sparse_embedding', 'metadata']
            for field in required_fields:
                if field not in insert_dict:
                    error_msg = f"缺少必要字段: {field}"
                    logging.error(error_msg)
                    raise ValueError(error_msg)

            # 检查各列表长度是否一致
            list_fields = ['text', 'embedding', 'id', 'sparse_embedding', 'metadata']
            lengths = {field: len(insert_dict[field]) for field in list_fields}
            if len(set(lengths.values())) > 1:
                error_msg = (f"各列表字段长度不一致: {lengths}。"
                             "text、embedding、id、sparse_embedding和metadata的长度必须相同")
                logging.error(error_msg)
                raise ValueError(error_msg)

            # # 获取文档数量
            # num_docs = len(insert_dict['text'])
            # result_json = {}
            #
            # for i in range(num_docs):
            #     # 直接将文档字段合并到 result_json 中（注意：如果字段名相同，后面的会覆盖前面的）
            #     result_json = {
            #         f"doc_{i}": {
            #             "file_name": insert_dict['file_name'],
            #             "text": insert_dict['text'][i],
            #             "embedding": insert_dict['embedding'][i],
            #             "id": insert_dict['id'][i],
            #             "sparse_embedding": insert_dict['sparse_embedding'][i],
            #             "metadata": insert_dict['metadata'][i],
            #         } for i in range(num_docs)
            #     }
            #
            # # 转换为JSON字符串
            # # final_json_string = json.dumps(result_json)
            # return result_json

            # 获取文档数量
            num_docs = len(insert_dict['text'])
            json_list = []

            for i in range(num_docs):
                # 为每个位置创建单独的JSON对象
                doc_json = {
                    "file_name": insert_dict['file_name'],
                    "text": insert_dict['text'][i],
                    "embedding": insert_dict['embedding'][i],
                    "id": insert_dict['id'][i],
                    "sparse_embedding": insert_dict['sparse_embedding'][i],
                    "metadata": insert_dict['metadata'][i],
                }
                # json_list.append(json.dumps(doc_json))
                json_list.append(doc_json)

            return json_list

        except Exception as e:
            logging.error(f"文档转换JSON失败: {str(e)}")
            raise  # 重新抛出异常以便上层处理
    def validate_and_correct_name(self, name):
        """
        Check if the first character of the name is an underscore or a letter.
        If not, prepend an underscore to the name.

        Args:
        name (str): The name to be checked and possibly corrected.

        Returns:
        str: The corrected name if it was invalid, or the original name if it was valid.
        """
        if not name[0].isalpha() and name[0] != '_':
            # Prepend an underscore if the first character is not a letter or an underscore
            return '_' + name
        else:
            # Return the original name if it's already valid
            return name
    @component.output_types(creation=bool)
    def run(self, file_path: str, logging=None):
        # 检查file_path是否是json文件
        if os.path.isfile(file_path) and file_path.endswith('.json'):
            file_paths = [file_path]
            logging.info(f"检测到单个JSON文件：{file_path}")
        elif os.path.isdir(file_path):
            file_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.json')]
            logging.info(f"检测到文件夹路径，找到 {len(file_paths)} 个 JSON 文件")
        else:
            error_message = "提供的路径既不是JSON文件也不是文件夹路径!"
            logging.error(error_message)
            raise ValueError(error_message)
        insert_json_list = []
        for file_path in file_paths:
            logging.info(f"开始处理JSON文件：{file_path}")
            # 获取文件名
            file_path_new = os.path.basename(file_path)
            original_name = file_path_new.replace(".json", "")
            logging.info(f"JSON文件原始名称：{original_name}")
            file_name = self.milvus_collection_name
            logging.info(f"插入向量库的表名：{file_name}")

            indexing_pipeline = self.results["indexing_pipeline"]
            # 读取JSON文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            # logging.info(f"成功加载JSON文件：{file_path}，共包含 {len(data)} 条数据")
            # 获取最后一个元素的 page_idx 值
            if data:  # 确保列表不为空
                last_element = data[-1]
                page_idx = last_element.get('page_idx')
                if page_idx is not None:
                    logging.info(f"成功加载JSON文件：{file_path}，共{page_idx + 1}页，共包含 {len(data)} 条数据")
                else:
                    logging.info(f"成功加载JSON文件：{file_path}，无页码信息，共包含 {len(data)} 条数据")
            else:
                logging.warning("JSON文件中没有数据")
            # 过滤掉那些没有'text'键或'text'值为空的字典
            data = [item for item in data if item.get('text')]
            logging.info(f"过滤没有'text'键或'text'值为空的字典后剩余有效数据 {len(data)} 条")
            sources = []
            for item in data:
                source = ByteStream.from_string(json.dumps(item))
                sources.append(source)
            logging.info("*****进行构建索引管道*****")

            insert_dict = indexing_pipeline.run({"loader": {"sources": sources}})
            logging.info("数据已通过索引管道处理")
            # 兼容 local/api 两种模式
            if "Data_To_Written" in insert_dict:
                insert_dict = insert_dict['Data_To_Written']
                insert_dict['file_name'] = original_name
                insert_json = self.documents_dict_to_json(insert_dict)
            elif "document_writer" in insert_dict:
                docs_written = insert_dict["document_writer"].get("documents_written", 0)
                logging.info(f"local 模式：直接写入 Milvus {docs_written} 条文档")
                insert_json = json.dumps({
                    "file_name": original_name,
                    "documents": [],
                    "metadata": {},
                    "documents_written": docs_written,
                })
            else:
                raise RuntimeError(f"索引管道返回未知结构: {list(insert_dict.keys())}")
            insert_json_list.append(insert_json)
            logging.info(f"{original_name} 文件的知识库信息构建完成，并成功添加到返回列表中")
            # LLM Wiki 构建触发
            if self.use_llm_wiki and self.wiki_llm_config:
                try:
                    source_content = json_chunks_to_source_content(data)
                    project_path = normalize_path(os.path.join(find_project_root(), "llm_wiki"))
                    written = auto_ingest(
                        project_path=project_path,
                        source_path=file_path,
                        llm_config=self.wiki_llm_config,
                        source_content=source_content,
                        domain=original_name,
                    )
                    logging.info(f"LLM Wiki 构建完成: {original_name}，写入 {len(written)} 个文件")
                except Exception as e:
                    logging.error(f"LLM Wiki 构建失败: {original_name}: {e}")
        logging.info("知识库信息构建流程完成，返回结果汇总")
        # 只返回第一个文件的(平台只会一次存一个)
        return insert_json_list[0]

    def run_json(self, json_str: str, logging=None):
        """直接处理 JSON 字符串的方法"""

        # 解析 JSON 字符串
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            error_message = f"JSON 字符串解析失败: {str(e)}"
            logging.error(error_message)
            raise ValueError(error_message)

        original_name = "from_json_string"
        file_name = self.milvus_collection_name
        logging.info(f"JSON 字符串原始名称：{original_name}")
        logging.info(f"插入向量库的表名：{file_name}")

        indexing_pipeline = self.results["indexing_pipeline"]

        # 获取最后一个元素的 page_idx 值
        if data:  # 确保列表不为空
            last_element = data[-1]
            page_idx = last_element.get('page_idx')
            if page_idx is not None:
                logging.info(f"成功解析 JSON 字符串，共{page_idx + 1}页，共包含 {len(data)} 条数据")
            else:
                logging.info(f"成功解析 JSON 字符串，无页码信息，共包含 {len(data)} 条数据")
        else:
            # 返回空的 JSON 结构，而不是 None
            empty_result = {
                "file_name": original_name,
                "documents": [],
                "metadata": {}
            }
            return json.dumps(empty_result)

        # 过滤掉那些没有'text'键或'text'值为空的字典
        data = [item for item in data if item.get('text')]
        logging.info(f"过滤没有'text'键或'text'值为空的字典后剩余有效数据 {len(data)} 条")

        sources = []
        for item in data:
            source = ByteStream.from_string(json.dumps(item))
            sources.append(source)

        logging.info("*****进行构建索引管道*****")
        insert_dict = indexing_pipeline.run({"loader": {"sources": sources}})
        logging.info("数据已通过索引管道处理")

        # 兼容 local/api 两种模式
        if "Data_To_Written" in insert_dict:
            insert_dict = insert_dict['Data_To_Written']
            insert_dict['file_name'] = original_name
            insert_json = self.documents_dict_to_json(insert_dict)
        elif "document_writer" in insert_dict:
            docs_written = insert_dict["document_writer"].get("documents_written", 0)
            logging.info(f"local 模式：直接写入 Milvus {docs_written} 条文档")
            insert_json = json.dumps({
                "file_name": original_name,
                "documents": [],
                "metadata": {},
                "documents_written": docs_written,
            })
        else:
            raise RuntimeError(f"索引管道返回未知结构: {list(insert_dict.keys())}")
        logging.info("JSON 字符串的知识库信息构建完成")

        # LLM Wiki 构建触发
        if self.use_llm_wiki and self.wiki_llm_config:
            try:
                source_content = json_chunks_to_source_content(data)
                project_path = normalize_path(os.path.join(find_project_root(), "llm_wiki"))
                virtual_source_path = os.path.join(find_project_root(), "upload", f"{original_name}.json")
                written = auto_ingest(
                    project_path=project_path,
                    source_path=virtual_source_path,
                    llm_config=self.wiki_llm_config,
                    source_content=source_content,
                    domain=original_name,
                )
                logging.info(f"LLM Wiki 构建完成: {original_name}，写入 {len(written)} 个文件")
            except Exception as e:
                logging.error(f"LLM Wiki 构建失败: {original_name}: {e}")

        return insert_json


def find_project_root(start_dir=os.getcwd()):  # 寻找项目根路径
    current_dir = start_dir
    if getattr(sys, 'frozen', False):
        # 打包环境：返回可执行文件所在目录
        return Path(sys.executable).parent
    while True:
        if "main.py" in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached the root directory
            raise FileNotFoundError("Could not find main.py in any parent directory.")
        current_dir = parent_dir
