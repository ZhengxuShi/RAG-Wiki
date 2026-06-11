import csv
import json
import requests
# import logging
from logger_local import app_logger as logging
import os
import sys
import time
import requests
from more_itertools import batched
from typing import Any, Dict, List,Optional, Tuple
from haystack import Document, component
from pathlib import Path
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack.utils import Secret, deserialize_secrets_inplace
from retriever.configs.tokenize_config import RerankTokenizer, EmbeddingTokenizer
from openai.types import CreateEmbeddingResponse
from haystack.utils.http_client import init_http_client
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from openai import APIError, AsyncOpenAI, OpenAI
from hypster import HP
import re
from datetime import datetime
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
import fnmatch
from haystack.core.component.types import Variadic
import itertools
from retriever.configs.response import response_config
from transformers import AutoTokenizer
from dotenv import load_dotenv

# import jieba
from collections import Counter
from html import unescape

class LoggedTime:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start = time.perf_counter()
        logging.info(f"Starting {self.task_name}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        logging.info(f"Finished {self.task_name} in {elapsed:.2f} seconds")
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
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
@component
class SimpleDocumentJoiner:
    """
    极简版 DocumentJoiner：
    只负责把多个 Document 列表拼接成一个列表，不去重、不排序、不截断。
    """

    def __init__(self):
        pass

    @component.output_types(documents=List[Document])
    def run(self, documents1: List[Document], documents2: List[Document]):
        """
        处理嵌套的文档结构
        """
        with LoggedTime("多路检索结果合并模块耗时统计"):
            documents = [documents1, documents2]
            output_documents = []
            for item in documents:
                for item_ in item:
                    output_documents.extend(item_["documents"])
            output_documents_ = [{"documents": output_documents}]
            return {"documents": output_documents_}


@component
class Word_Filter:
    """
     创建一个自定义组件来过滤特定词
     """

    @component.output_types(query=str)
    def run(self, query: str):
        stop_words = []  # Configure domain-specific stop words as needed
        # query过滤的方法 待添加
        for word in stop_words:
            query = query.replace(word, '')
        return {"query": query}

@component
class Word_Filter_Muiltiple:
    """
     创建一个自定义组件来过滤特定词
     """

    @component.output_types(query=List[str])
    def run(self, query: str):
        stop_words = []  # Configure domain-specific stop words as needed
        # query过滤的方法 待添加
        for word in stop_words:
            query = query.replace(word, '')
        return {"query": [query]}


@component
class Answer_association:
    """
     创建一个自定义组件提取联想生成的回复中的问题列表
     入参：List[Document]
     返回：List[Document]
     """

    @component.output_types(associated_questions=List[str])
    def run(self, replies: List[str]):
        associated_questions = []
        if len(replies) == 0 or replies[0] == "":
            return {"associated_questions": []}
        else:
            for reply in replies:
                questions = re.findall(r'"Associated Question \d+": "(.*?)"', reply)
                # reply_dict = json.loads(reply)
                # # 提取值并组成列表
                # associated_questions = list(reply_dict.values())

            # Removing empty strings from the list
            cleaned_questions = [question for question in questions if question != ""]
            return {"associated_questions": cleaned_questions}


@component
class PassThroughDocuments:
    """
    A component for normalizing the input and output of the pipeline
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        logging.info("*****进行检索结果整理模块*****")
        with LoggedTime("检索结果整理模块耗时统计"):
            for index, element in enumerate(documents):
                flag = 0
                for document in element['documents']:
                    logging.info("检索完成，文档[%d]内容=%s,得分为：%s", flag + 1, document.content[:50], document.score)
                    flag += 1
            concatenated_documents = []
            for item in documents:
                # 将每个字典中的 "documents" 列表添加到新列表中
                concatenated_documents.extend(item["documents"])
            logging.info("*****结束检索结果整理模块*****")
            return {"documents": concatenated_documents}


@component
class Compressed_Search_Results:
    """
    一个对于检索结果压缩的组件，入参和输出都是文档列表 List[Document]
    功能：
    1. 合并来自同一表格的文档内容
    2. 去除内容完全相同的重复文档
    3. 限制最终返回的文档数量
    """

    def __init__(self, retain_count: int):
        self.retain_count = int(retain_count)

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        logging.info("*****进行检索结果压缩模块*****")
        with LoggedTime("检索结果压缩模块耗时统计"):
            # 第一阶段：表格内容压缩
            # 用于存储每个imgpath第一次出现的文档的索引
            first_occurrence: Dict[str, int] = {}
            # 用于存储每个imgpath的所有文档内容
            content_buffers: Dict[str, List[str]] = {}
            # 用于存储需要删除的文档索引
            to_remove: List[int] = []

            # 遍历文档，分组并记录内容
            for index, doc in enumerate(documents):
                imgpath = doc.meta.get("metadata", {}).get("img_path", None)
                if imgpath :
                    if imgpath not in first_occurrence:
                        first_occurrence[imgpath] = index
                        content_buffers[imgpath] = []
                    content_buffers[imgpath].append(doc.content)
                    if index != first_occurrence[imgpath]:
                        to_remove.append(index)
                    logging.debug(f"文档[%d]已分配到imgpath组: %s", index, imgpath)
                else:
                    logging.warning(f"文档[%d]没有img_path信息，跳过分组", index)

            # 合并内容并修改原始文档
            for imgpath, contents in content_buffers.items():
                combined_content = "\n".join(contents)
                first_index = first_occurrence[imgpath]
                logging.debug(f"合并imgpath组: {imgpath}，共{len(contents)}个文档，首个文档索引: {first_index}")
                documents[first_index].content = combined_content

            # 删除被拼接的文档
            for index in sorted(to_remove, reverse=True):
                logging.debug(f"正在删除重复的文档，索引: %d", index)
                del documents[index]

            # 第二阶段：相同内容压缩
            # 用于存储已出现的内容和第一次出现的索引
            content_seen: Dict[str, int] = {}
            # 重置需要删除的索引列表
            to_remove = []

            for index, doc in enumerate(documents):
                content = doc.content
                if content in content_seen:
                    logging.debug(f"发现重复内容文档[%d]，与文档[%d]内容相同，标记为删除",
                                  index, content_seen[content])
                    to_remove.append(index)
                else:
                    content_seen[content] = index

            # 删除重复内容的文档
            for index in sorted(to_remove, reverse=True):
                logging.debug(f"正在删除内容重复的文档，索引: %d", index)
                del documents[index]

            # 第三阶段：限制结果数量
            if len(documents) > self.retain_count:
                removed_count = len(documents) - self.retain_count
                documents = documents[:self.retain_count]
                logging.info(f"已限制结果数量，移除了{removed_count}个额外文档，保留前{self.retain_count}个")
            else:
                logging.info(f"当前文档总数为{len(documents)}，未超过限制{self.retain_count}，无需裁剪")

            logging.info("*****检索结果压缩模块完成，最终返回%d个文档*****", len(documents))
            return {"documents": documents}


@component
class PassThroughDocuments_Rerank:
    """
    A component for normalizing the input and output of the pipeline
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        logging.info("*****进行rerank结果整理模块*****")
        with LoggedTime("rerank结果整理模块耗时统计"):
            for index, document in enumerate(documents):
                logging.info("rerank完成，文档[%d]内容=%s,得分为：%s", index + 1, document.content[:50], document.score)
            logging.info("*****结束rerank结果整理模块*****")
            return {"documents": documents}


@component
class LLM_Rescore:

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        for doc in documents:
            if doc.meta["metadata"]["score"] < 0.1:
                doc.meta["metadata"]["score"] = 0.1
        return {"documents": documents}


@component
class DocumentToList:
    """
    A component for normalizing the input and output of the pipeline
    """

    @component.output_types(document=Document, documents=List[Document])
    def run(self, document: Document):
        return {"document": document, "documents": [document]}


@component
class PassThroughText:
    """
    A component for normalizing the input and output of the pipeline
    """

    @component.output_types(text=str)
    def run(self, text: str):
        return {"text": text}


@component
class ContentThroughList:
    """
    A component for normalizing the input and output of the pipeline
    """

    @component.output_types(content=List[str])
    def run(self, content: List[str]):
        return {"content": content}


@component
class AddLLMMetadata:
    """
    A component for adding an object to a document's metadata
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], replies: List[str]) -> Dict[str, Any]:
        logging.info("*****进行AddLLMMetadata模块*****")
        # 如果replies比documents短，用''补齐
        if len(replies) < len(documents):
            replies += [''] * (len(documents) - len(replies))
            logging.info("Replies列表长度小于文档数，已用空字符串补齐")

        # 如果replies比documents长，截断replies
        if len(replies) > len(documents):
            replies = replies[:len(documents)]
            logging.info("Replies列表长度大于文档数，已截断")

        # 更新文档的meta信息
        for idx, (doc, reply) in enumerate(zip(documents, replies)):
            logging.debug("处理第 %d 个文档的元数据更新", idx + 1)
            # 覆盖original_content字段并记录原始内容
            doc.meta["metadata"]["original_content"] = doc.content
            logging.debug("文档[%s]已保存原始内容到original_content", doc.id)

            try:
                # 更新文档内容为LLM回复
                doc.content = reply["replies"][0]
                logging.debug("文档[%s]的内容已更新为LLM回复", doc.id)
            except (KeyError, IndexError) as e:
                logging.error("处理文档[%s]时发生错误：%s", doc.id, str(e))
                doc.content = ""

        logging.info("AddLLMMetadata模块执行完成，共处理 %d 个文档", len(documents))
        return {"documents": documents}


@component
class RepliesToQuery:
    """
    一个自定义组件，将llm回复的replies[list]转成query[list]
    """

    @component.output_types(query=List[str])
    def run(self, replies: List[str]):
        query = replies
        return {"query": query}


@component
class Query2query:
    """
    一个自定义组件，将query[list]的所有问题，经过所有问题处理query_methods，
    转成新的query[list]
    """
    def __init__(self, hp: HP, query_methods: str, llm_query):
        self.hp = hp
        self.query_methods = query_methods
        self.llm_query = llm_query
        self.response = self.hp.nest(response_config, name="response")

    @component.output_types(query=List[str])
    def run(self, query: str):
        from jinja2 import Template
        from hypster import HP

        logging.info("*****进行问题改写模块*****")
        # response = self.hp.nest(config_func="retriever/configs/response.py", name="response")

        llm_query = self.llm_query
        # query_methods = query_["query_methods"]
        query_methods = self.query_methods
        answer_query = []
        answer_query_temp = []
        query_methods = query_methods.split("+") if "+" in query_methods else [query_methods]

        for query_method in query_methods:
            if query_method == "query_hyde":
                prompt = self.response["prompt_hyde"]
                template = Template(prompt)

                if len(answer_query) > 1:
                    for a_query in answer_query:
                        prompt_text = template.render(query=a_query)
                        answer = llm_query.run(prompt=prompt_text)
                        answer_query_temp.append(answer["replies"][0])
                    answer_query = answer_query_temp
                    answer_query_temp = []
                    for index, element in enumerate(answer_query):
                        logging.info("经过Hyde处理，问题[%d]=%s", index, element)

                elif len(answer_query) == 1:
                    prompt_text = template.render(query=answer_query[0])
                    answer = llm_query.run(prompt=prompt_text)
                    answer_query = []
                    answer_query.append(answer["replies"][0])
                    for index, element in enumerate(answer_query):
                        logging.info("经过Hyde处理，问题[%d]=%s", index, element)
                else:
                    prompt_text = template.render(query=query)
                    answer = llm_query.run(prompt=prompt_text)
                    answer_query.append(answer["replies"][0])
                    for index, element in enumerate(answer_query):
                        logging.info("经过Hyde处理，问题[%d]=%s", index, element)



            elif query_method == "query_language":
                logging.info("问题准备经过Query_Language处理")
                prompt = self.response["prompt_language"]
                template = Template(prompt)

                if len(answer_query) > 1:
                    for a_query in answer_query:
                        prompt_text = template.render(query=a_query)
                        answer = llm_query.run(prompt=prompt_text)
                        answer_query_temp.append(answer["replies"][0])
                    answer_query = answer_query_temp
                    answer_query_temp = []
                    for index, element in enumerate(answer_query):
                        logging.info("经过Query_Language处理，问题[%d]=%s", index, element)

                elif len(answer_query) == 1:
                    prompt_text = template.render(query=answer_query[0])
                    answer = llm_query.run(prompt=prompt_text)
                    answer_query = []
                    answer_query.append(answer["replies"][0])
                    for index, element in enumerate(answer_query):
                        logging.info("经过Query_Language处理，问题[%d]=%s", index, element)
                else:
                    prompt_text = template.render(query=query)
                    answer = llm_query.run(prompt=prompt_text)
                    answer_query.append(answer["replies"][0])
                    for index, element in enumerate(answer_query):
                        logging.info("经过Query_Language处理，问题[%d]=%s", index, element)



            elif query_method == "query_association":
                logging.info("问题准备经过联想处理")
                answer_association = Answer_association()
                prompt = self.response["prompt_association"]
                template = Template(prompt)

                if len(answer_query) > 1:
                    for a_query in answer_query:
                        prompt_text = template.render(query=a_query)
                        answer = llm_query.run(prompt=prompt_text)
                        cleaned_query = answer_association.run(replies=answer["replies"])
                        a_query_ = cleaned_query["associated_questions"]
                        answer_query_temp.append(a_query_)
                    answer_query = sum(answer_query_temp, [])
                    answer_query_temp = []
                    for index, element in enumerate(answer_query):
                        logging.info("经过问题联想处理，问题[%d]=%s", index, element)

                elif len(answer_query) == 1:
                    prompt_text = template.render(query=answer_query[0])
                    answer = llm_query.run(prompt=prompt_text)
                    cleaned_query = answer_association.run(replies=answer["replies"])
                    answer_query = cleaned_query["associated_questions"]
                    for index, element in enumerate(answer_query):
                        logging.info("经过问题联想处理，问题[%d]=%s", index, element)
                else:
                    prompt_text = template.render(query=query)
                    answer = llm_query.run(prompt=prompt_text)
                    cleaned_query = answer_association.run(replies=answer["replies"])
                    answer_query = cleaned_query["associated_questions"]
                    for index, element in enumerate(answer_query):
                        logging.info("经过问题联想处理，问题[%d]=%s", index, element)

            elif query_method == "none":
                answer_query.append(query)
                logging.info("问题不经过任何处理，问题=%s", query)
        logging.info("*****结束问题改写模块*****")
        return {"query": answer_query}


from typing import Any, Dict, List, Optional
from ollama import Client
from openai import OpenAI


@component
class Query2TextEbd:
    """
    一个自定义组件，将query[list]的所有问题，向量化存储为result，
    转成新的query[list]
    """

    def __init__(
            self,
            embedder,
            retrieval
    ):
        self.retrieval = retrieval
        self.retrieval_method = self.retrieval["retrieval_method"]
        self.embedder = embedder

    @component.output_types(query_embedding=List[Dict[str, Any]])
    def run(self, query: List[str]):
        logging.info("*****进行问题向量化模块*****")
        with LoggedTime("问题向量化模块耗时统计"):
            if self.retrieval_method == "embedding_retriever":
                logging.info("使用稠密向量检索方法进行问题向量化")
                self.model = self.embedder["model"]
                text_embedder = self.embedder["text_embedder"]
                if self.embedder["embedder_model_platform"] == "local_st":
                    text_embedder.warm_up()
                result_ = []
                for idx, query_ in enumerate(query):
                    result = text_embedder.run(text=query_)
                    result["meta"] = {"model": self.model, "query_embedding": query_}
                    # result["metadata"]["LLM_Enrich"] = doc.content
                    result_.append(result)
                    logging.info("问题[%d]: %s 已完成稠密向量化", idx, query_[:30] + "..." if len(query_) > 30 else query_)
                logging.info("稠密向量化处理完成，共处理 %d 个问题", len(query))
                logging.info("*****结束问题向量化模块*****")
                return {"query_embedding": result_}

            elif self.retrieval_method == "sparse_embedding_retriever":
                logging.info("使用稀疏向量检索方法进行问题向量化")
                text_embedder = self.embedder["sparse_text_embedder"]
                if self.embedder["sparse_embedder_model_platform"] == "local_st":
                    text_embedder.warm_up()
                self.model = self.embedder["sparse_model"]
                result_ = []
                for idx, query_ in enumerate(query):
                    result = text_embedder.run(text=query_)
                    result["meta"] = {"model": self.model, "query_embedding": query_}
                    result_.append(result)
                    logging.info("问题[%d]: %s 已完成稀疏向量化", idx, query_[:30] + "..." if len(query_) > 30 else query_)
                logging.info("稀疏向量化处理完成，共处理 %d 个问题", len(query))
                logging.info("*****结束问题向量化模块*****")
                return {"query_embedding": result_}


            elif self.retrieval_method == "hybrid_retriever":
                logging.info("使用混合检索方法（稠密+稀疏）进行问题向量化")
                self.model = self.embedder["model"]
                sparse_embedder = self.embedder["sparse_text_embedder"]
                text_embedder = self.embedder["text_embedder"]
                if self.embedder["embedder_model_platform"] == "local_st":
                    text_embedder.warm_up()
                if self.embedder["sparse_embedder_model_platform"] == "local_st":
                    sparse_embedder.warm_up()
                result_ = []
                for idx, query_ in enumerate(query):
                    with LoggedTime("text_embedder耗时统计"):
                        result = text_embedder.run(text=query_)
                    with LoggedTime("sparse_embedder耗时统计"):
                        result_sparse = sparse_embedder.run(text=query_)
                    result["meta"] = {"model_embedder": self.model, "model_sparse": self.embedder["sparse_model"],
                                      "query_embedding": query_}
                    result["sparse_embedding"] = result_sparse["sparse_embedding"]
                    result_.append(result)
                    logging.info("问题[%d]: %s 已完成混合向量化", idx, query_[:30] + "..." if len(query_) > 30 else query_)
                logging.info("混合向量化处理完成，共处理 %d 个问题", len(query))
                logging.info("*****结束问题向量化模块*****")
                return {"query_embedding": result_}

@component
class OpenAISparseTextEmbedder:
    """
    Embeds strings using OpenAI models.

    You can use it to embed user query and send it to an embedding Retriever.

    ### Usage example

    ```python
    from haystack.components.embedders import OpenAITextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = OpenAITextEmbedder()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    # 'meta': {'model': 'text-embedding-ada-002-v2',
    #          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "text-embedding-ada-002",
        dimensions: Optional[int] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an OpenAITextEmbedder component.

        Before initializing the component, you can set the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES'
        environment variables to override the `timeout` and `max_retries` parameters respectively
        in the OpenAI client.

        :param api_key:
            The OpenAI API key.
            You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
            during initialization.
        :param model:
            The name of the model to use for calculating embeddings.
            The default model is `text-embedding-ada-002`.
        :param dimensions:
            The number of dimensions of the resulting embeddings. Only `text-embedding-3` and
            later models support this parameter.
        :param api_base_url:
            Overrides default base URL for all HTTP requests.
        :param organization:
            Your organization ID. See OpenAI's
            [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
            for more information.
        :param prefix:
            A string to add at the beginning of each text to embed.
        :param suffix:
            A string to add at the end of each text to embed.
        :param timeout:
            Timeout for OpenAI client calls. If not set, it defaults to either the
            `OPENAI_TIMEOUT` environment variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact OpenAI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        self.model = model
        self.dimensions = dimensions
        self.api_base_url = api_base_url
        self.organization = organization
        self.prefix = prefix
        self.suffix = suffix
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.http_client_kwargs = http_client_kwargs

        if timeout is None:
            timeout = float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        if max_retries is None:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key.resolve_value(),
            "organization": organization,
            "base_url": api_base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }

        self.client = OpenAI(http_client=init_http_client(self.http_client_kwargs, async_client=False), **client_kwargs)
        self.async_client = AsyncOpenAI(
            http_client=init_http_client(self.http_client_kwargs, async_client=True), **client_kwargs
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            dimensions=self.dimensions,
            api_base_url=self.api_base_url,
            organization=self.organization,
            prefix=self.prefix,
            suffix=self.suffix,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAITextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_input(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError(
                "OpenAITextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the OpenAIDocumentEmbedder."
            )

        text_to_embed = self.prefix + text + self.suffix

        kwargs: Dict[str, Any] = {"model": self.model, "input": text_to_embed}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        return kwargs

    def _prepare_output(self, result: CreateEmbeddingResponse) -> Dict[str, Any]:
        # 获取原始的稀疏嵌入数据，兼容两种访问方式：
        # 1. model_extra是字典，使用.get()方法
        # 2. model_extra直接包含sparse_embedding属性，直接访问

        sparse_embedding = None
        try:
            if isinstance(result.data[0].model_extra, dict):
                raw_sparse_embedding = result.data[0].model_extra.get('sparse_embedding', {})
            else:
                raw_sparse_embedding = getattr(result.data[0].model_extra, 'sparse_embedding', None)

            # 检查是否成功获取sparse_embedding数据
            if raw_sparse_embedding is None or (
                not isinstance(raw_sparse_embedding, dict) or
                raw_sparse_embedding.get('indices') is None or
                raw_sparse_embedding.get('values') is None
            ):
                # 获取失败，返回空sparse_embedding
                sparse_embedding = None
            else:
                # 转换为 SparseEmbedding 类型
                sparse_embedding = SparseEmbedding(
                    indices=raw_sparse_embedding['indices'],
                    values=raw_sparse_embedding['values']
                )
        except Exception as e:
            # 出现任何异常，记录日志并返回空sparse_embedding
            logging.warning(f"稀疏向量接口返回结果中获取sparse_embedding数据失败: {str(e)}")
            sparse_embedding = None

        return {"sparse_embedding": sparse_embedding, "meta": {"model": result.model, "usage": dict(result.usage)}}

    @component.output_types(sparse_embedding=SparseEmbedding, meta=Dict[str, Any])
    def run(self, text: str):
        """
        Embeds a single string.

        :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
            - `meta`: Information about the usage of the model.
        """
        create_kwargs = self._prepare_input(text=text)
        response = self.client.embeddings.create(**create_kwargs)
        return self._prepare_output(result=response)

    @component.output_types(sparse_embedding=SparseEmbedding, meta=Dict[str, Any])
    async def run_async(self, text: str):
        """
        Asynchronously embed a single string.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in async code.

        :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
            - `meta`: Information about the usage of the model.
        """
        create_kwargs = self._prepare_input(text=text)
        response = await self.async_client.embeddings.create(**create_kwargs)
        return self._prepare_output(result=response)
@component
class OpenAISparseDocumentEmbedder:
    """
    Computes document embeddings using OpenAI models.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.embedders import OpenAIDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = OpenAIDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(  # noqa: PLR0913 (too-many-arguments) # pylint: disable=too-many-positional-arguments
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "text-embedding-ada-002",
        dimensions: Optional[int] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
        *,
        raise_on_failure: bool = False,
    ):
        """
        Creates an OpenAIDocumentEmbedder component.

        Before initializing the component, you can set the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES'
        environment variables to override the `timeout` and `max_retries` parameters respectively
        in the OpenAI client.

        :param api_key:
            The OpenAI API key.
            You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
            during initialization.
        :param model:
            The name of the model to use for calculating embeddings.
            The default model is `text-embedding-ada-002`.
        :param dimensions:
            The number of dimensions of the resulting embeddings. Only `text-embedding-3` and
            later models support this parameter.
        :param api_base_url:
            Overrides the default base URL for all HTTP requests.
        :param organization:
            Your OpenAI organization ID. See OpenAI's
            [Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
            for more information.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        :param batch_size:
            Number of documents to embed at once.
        :param progress_bar:
            If `True`, shows a progress bar when running.
        :param meta_fields_to_embed:
            List of metadata fields to embed along with the document text.
        :param embedding_separator:
            Separator used to concatenate the metadata fields to the document text.
        :param timeout:
            Timeout for OpenAI client calls. If not set, it defaults to either the
            `OPENAI_TIMEOUT` environment variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact OpenAI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or 5 retries.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param raise_on_failure:
            Whether to raise an exception if the embedding request fails. If `False`, the component will log the error
            and continue processing the remaining documents. If `True`, it will raise an exception on failure.
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.api_base_url = api_base_url
        self.organization = organization
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.timeout = timeout
        self.max_retries = max_retries
        self.http_client_kwargs = http_client_kwargs
        self.raise_on_failure = raise_on_failure

        if timeout is None:
            timeout = float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        if max_retries is None:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key.resolve_value(),
            "organization": organization,
            "base_url": api_base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }

        self.client = OpenAI(http_client=init_http_client(self.http_client_kwargs, async_client=False), **client_kwargs)
        self.async_client = AsyncOpenAI(
            http_client=init_http_client(self.http_client_kwargs, async_client=True), **client_kwargs
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            dimensions=self.dimensions,
            api_base_url=self.api_base_url,
            organization=self.organization,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> Dict[str, str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = {}
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]

            texts_to_embed[doc.id] = (
                self.prefix + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""]) + self.suffix
            )

        return texts_to_embed

    def _embed_batch(
        self, texts_to_embed: Dict[str, str], batch_size: int
    ) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """

        doc_ids_to_embeddings: Dict[str, List[float]] = {}
        meta: Dict[str, Any] = {}
        for batch in tqdm(
            batched(texts_to_embed.items(), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            args: Dict[str, Any] = {"model": self.model, "input": [b[1] for b in batch]}

            if self.dimensions is not None:
                args["dimensions"] = self.dimensions

            try:
                response = self.client.embeddings.create(**args)
            except APIError as exc:
                ids = ", ".join(b[0] for b in batch)
                logging.error("Failed embedding of documents %s caused by %s", ids, exc)
                if self.raise_on_failure:
                    raise exc
                continue

            sparse_embedding_list = []
            for el in response.data:
                try:
                    # 兼容两种方式获取sparse_embedding：
                    # 1. el.model_extra是字典，使用.get()方法
                    # 2. el.model_extra直接包含sparse_embedding属性，直接访问
                    if isinstance(el.model_extra, dict):
                        raw_sparse_embedding = el.model_extra.get('sparse_embedding', {})
                    else:
                        raw_sparse_embedding = getattr(el.model_extra, 'sparse_embedding', None)

                    if (raw_sparse_embedding and
                            raw_sparse_embedding.get('indices') is not None and
                            raw_sparse_embedding.get('values') is not None):

                        sparse_embedding = SparseEmbedding(
                            indices=raw_sparse_embedding['indices'],
                            values=raw_sparse_embedding['values']
                        )
                        sparse_embedding_list.append(sparse_embedding)
                    else:
                        sparse_embedding_list.append(None)

                except (AttributeError, KeyError, TypeError):
                    sparse_embedding_list.append(None)

            doc_ids_to_embeddings.update(dict(zip((b[0] for b in batch), sparse_embedding_list)))

            if "model" not in meta:
                meta["model"] = response.model
            if "usage" not in meta:
                # response中有usage数据才计算，否则按照注释的赋值0
                if hasattr(response, 'usage') and response.usage is not None:
                    meta["usage"] = dict(response.usage)
                else:
                    meta["usage"] = {"prompt_tokens": 0, "total_tokens": 0}
            else:
                # 当response中有usage时才累加，否则跳过累加
                try:
                    if hasattr(response, 'usage') and response.usage is not None:
                        meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                        meta["usage"]["total_tokens"] += response.usage.total_tokens
                except Exception:
                    # 出现任何异常，赋值0
                    meta["usage"]["prompt_tokens"] = 0
                    meta["usage"]["total_tokens"] = 0

        return doc_ids_to_embeddings, meta

    async def _embed_batch_async(
        self, texts_to_embed: Dict[str, str], batch_size: int
    ) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        """
        Embed a list of texts in batches asynchronously.
        """

        doc_ids_to_embeddings: Dict[str, List[float]] = {}
        meta: Dict[str, Any] = {}

        batches = list(batched(texts_to_embed.items(), batch_size))
        if self.progress_bar:
            batches = async_tqdm(batches, desc="Calculating embeddings")

        for batch in batches:
            args: Dict[str, Any] = {"model": self.model, "input": [b[1] for b in batch]}

            if self.dimensions is not None:
                args["dimensions"] = self.dimensions

            try:
                response = await self.async_client.embeddings.create(**args)
            except APIError as exc:
                ids = ", ".join(b[0] for b in batch)
                logging.error("Failed embedding of documents %s caused by %s", ids, exc)
                if self.raise_on_failure:
                    raise exc
                continue

            sparse_embedding_list = []
            for el in response.data:
                try:
                    # 兼容两种方式获取sparse_embedding：
                    # 1. el.model_extra是字典，使用.get()方法
                    # 2. el.model_extra直接包含sparse_embedding属性，直接访问
                    if isinstance(el.model_extra, dict):
                        raw_sparse_embedding = el.model_extra.get('sparse_embedding', {})
                    else:
                        raw_sparse_embedding = getattr(el.model_extra, 'sparse_embedding', None)

                    if (raw_sparse_embedding and
                            raw_sparse_embedding.get('indices') is not None and
                            raw_sparse_embedding.get('values') is not None):

                        sparse_embedding = SparseEmbedding(
                            indices=raw_sparse_embedding['indices'],
                            values=raw_sparse_embedding['values']
                        )
                        sparse_embedding_list.append(sparse_embedding)
                    else:
                        sparse_embedding_list.append(None)

                except (AttributeError, KeyError, TypeError):
                    sparse_embedding_list.append(None)

            doc_ids_to_embeddings.update(dict(zip((b[0] for b in batch), sparse_embedding_list)))

            if "model" not in meta:
                meta["model"] = response.model
            if "usage" not in meta:
                meta["usage"] = dict(response.usage)
            else:
                # 当response中有usage时才累加，否则跳过累加
                try:
                    if hasattr(response, 'usage') and response.usage is not None:
                        meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                        meta["usage"]["total_tokens"] += response.usage.total_tokens
                except Exception:
                    # 出现任何异常，赋值0
                    meta["usage"]["prompt_tokens"] = 0
                    meta["usage"]["total_tokens"] = 0


        return doc_ids_to_embeddings, meta

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]):
        """
        Embeds a list of documents.

        :param documents:
            A list of documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "OpenAIDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the OpenAITextEmbedder."
            )

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        doc_ids_to_embeddings, meta = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        doc_id_to_document = {doc.id: doc for doc in documents}
        for doc_id, emb in doc_ids_to_embeddings.items():
            doc_id_to_document[doc_id].sparse_embedding = emb
        documents_tmp=list(doc_id_to_document.values())
        return {"documents": documents_tmp, "meta": meta}

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    async def run_async(self, documents: List[Document]):
        """
        Embeds a list of documents asynchronously.

        :param documents:
            A list of documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "OpenAIDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the OpenAITextEmbedder."
            )

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        doc_ids_to_embeddings, meta = await self._embed_batch_async(
            texts_to_embed=texts_to_embed, batch_size=self.batch_size
        )

        doc_id_to_document = {doc.id: doc for doc in documents}
        for doc_id, emb in doc_ids_to_embeddings.items():
            doc_id_to_document[doc_id].sparse_embedding = emb
        documents_tmp = list(doc_id_to_document.values())
        return {"documents": documents_tmp, "meta": meta}
@component
class TextEbd_EmbeddingRetriever:
    """
    一个自定义组件，对result_[list]的所有embedding，进行检索，输出 documents=List[Document]
    """

    def __init__(self, retrieval):
        self.retrieval = retrieval
        self.retrieval_method = self.retrieval["retrieval_method"]
        self.retrieval_hybrid_retriever_top_k = self.retrieval["retrieval_hybrid_retriever_top_k"]
        self.embedding_retriever = self.retrieval["embedding_retriever"]
    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[Dict[str, Any]], milvus_filename=[], knowledgeUUID=None, query_hybrid_handler=None, parseType=None, logging=logging):
        logging.info("*****进行检索模块*****")
        with LoggedTime("实际检索模块耗时统计"):
            documents_ = []
            if self.retrieval_method == "embedding_retriever":
                logging.info("使用稠密向量检索方法进行检索")
                for query in query_embedding:
                    logging.debug("处理第 %d 个查询的稠密向量检索", query_embedding.index(query) + 1)
                    documents = self.embedding_retriever.run(query_embedding=query['embedding'])
                    for doc in documents["documents"]:
                        doc.meta["metadata"]["query_embedding"] = query["meta"]["query_embedding"]
                        logging.debug("文档[%s]已添加查询向量信息", doc.id)
                    documents_.append(documents)
                documents_ = [sublist for sublist in documents_]
                logging.info("稠密向量检索完成，共处理 %d 个查询", len(query_embedding))
                return {"documents": documents_}

            elif self.retrieval_method == "sparse_embedding_retriever":
                logging.info("使用稀疏向量检索方法进行检索")
                for query in query_embedding:
                    logging.debug("处理第 %d 个查询的稀疏向量检索", query_embedding.index(query) + 1)
                    documents = self.embedding_retriever.run(query_sparse_embedding=query['sparse_embedding'])
                    for doc in documents["documents"]:
                        doc.meta["metadata"]["query_embedding"] = query["meta"]["query_embedding"]
                        logging.debug("文档[%s]已添加查询向量信息", doc.id)
                    documents_.append(documents)
                documents_ = [sublist for sublist in documents_]
                logging.info("稀疏向量检索完成，共处理 %d 个查询", len(query_embedding))
                return {"documents": documents_}

            elif self.retrieval_method == "hybrid_retriever":
                logging.info("使用混合检索方法（稠密+稀疏）进行检索")
                for query in query_embedding:
                    logging.debug("处理第 %d 个查询的混合检索", query_embedding.index(query) + 1)
                    documents = self.embedding_retriever.run(
                        query_sparse_embedding=query['sparse_embedding'],
                        query_embedding=query['embedding'],
                        filename=milvus_filename,
                        knowledgeUUID=knowledgeUUID,
                        query_hybrid_handler=query_hybrid_handler,
                        parseType=parseType,
                        logging=logging
                    )
                    for doc in documents["documents"]:
                        doc.meta["metadata"]["query_embedding"] = query["meta"]["query_embedding"]
                        logging.debug("文档[%s]已添加混合检索元数据", doc.id)
                    documents_.append(documents)
                documents_ = [sublist for sublist in documents_]
                logging.info("混合检索完成，共处理 %d 个查询", len(query_embedding))

            logging.info("检索模块执行结束，返回结果包含 %d 组文档", len(documents_))
            logging.info("*****结束检索模块*****")
            return {"documents": documents_}


@component
class Content_Restore:
    """
    一个自定义组件，对检索结果documents的content内容进行还原，输出 documents=List[Document]
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        logging.info("*****进行检索结果还原模块*****")
        with LoggedTime("检索结果还原模块耗时统计"):
            for idx, documents_ in enumerate(documents):
                logging.debug("处理第 %d 个检索结果的文档还原", idx + 1)
                for doc_idx, doc in enumerate(documents_["documents"]):
                    logging.debug("正在处理文档[%d]的内容还原", doc_idx + 1)
                    if "original_content" in doc.meta["metadata"]:
                        # 记录当前文档内容到LLM_Enrich字段
                        doc.meta["metadata"]["LLM_Enrich"] = doc.content
                        # 将文档内容恢复为原始内容
                        doc.content = doc.meta["metadata"]["original_content"]
                        logging.debug("文档[%s]已成功还原原始内容", doc.id)
                    else:
                        logging.debug("文档[%s]无original_content信息，跳过还原", doc.id)

            logging.info("检索结果还原模块执行完成，共处理 %d 组文档", len(documents))
            logging.info("*****结束检索结果还原模块*****")
            return {"documents": documents}


@component
class Enrich_Doc_LLM:
    """
    一个自定义组件，将输入documents: List[Document]总结或提取,输出 replies=List[str]
    """

    def __init__(self, hp: Any, indexing_method: str, llm_enrich, file_name):
        self.hp = hp
        self.indexing_method = indexing_method
        self.llm_enrich = llm_enrich
        self.save_path = "./saved_replies"
        self.replies = []
        self.processed_count = 0
        self.batch_size = 100  # 每隔100个保存一次
        self.file_name = file_name
        self.response = self.hp.nest(response_config, name="response")
        # 确保存储路径存在
        os.makedirs(self.save_path, exist_ok=True)

        # 尝试从本地恢复已保存的replies

        self._load_saved_replies()

    def _load_saved_replies(self):
        """从本地加载已保存的replies"""
        logging.info("*****开始尝试加载已保存的回复数据*****")
        # 根据file_name和indexing_method确定文件名模式
        file_pattern = f"replies_{self.file_name}_{self.indexing_method}_*.json"
        logging.debug(f"查找匹配模式 {file_pattern} 的保存文件")
        saved_files = [f for f in os.listdir(self.save_path) if fnmatch.fnmatch(f, file_pattern)]

        if not saved_files:
            logging.warning(f"未找到与模式 {file_pattern} 匹配的已保存文件")
            return

        # 按时间排序，取最新的文件
        logging.info(f"发现 {len(saved_files)} 个匹配文件，正在按修改时间排序")
        saved_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.save_path, x)), reverse=True)
        latest_file = os.path.join(self.save_path, saved_files[0])
        logging.info(f"选择最新文件 {latest_file} 进行加载")

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                self.replies = saved_data.get("replies", [])
                self.processed_count = saved_data.get("processed_count", 0)
                logging.info(f"成功加载 {len(self.replies)} 条回复，已处理文档计数为 {self.processed_count}")
                logging.debug(f"加载数据来自文件: {latest_file}")
        except Exception as e:
            logging.error(f"加载已保存回复时发生错误: {e}", exc_info=True)

    def _save_replies(self):
        """将当前的replies保存到本地"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 保存文件名包含file_name
        file_name = f"replies_{self.file_name}_{self.indexing_method}_{timestamp}.json"
        file_path = os.path.join(self.save_path, file_name)

        data_to_save = {
            "replies": self.replies,
            "processed_count": self.processed_count
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {len(self.replies)} replies to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save replies: {e}")

    @component.output_types(replies=List[str])
    def run(self, documents: List[Document]):
        from jinja2 import Template
        # response = self.hp.nest("retriever/configs/response.py", name="response")
        indexing_method = self.indexing_method

        # 继续处理未完成的文档
        start_index = self.processed_count
        total_documents = len(documents)

        for idx, document in tqdm(enumerate(documents[start_index:]), desc="Enrich", unit="doc", initial=start_index,
                                  total=total_documents):
            if indexing_method == "enrich_doc_key":
                logging.info("Enrich Doc By Key")
                prompt = self.response["prompt_enrich_doc_key"]
                template = Template(prompt)
                prompt_text = template.render(document=document)
                reply = self.llm_enrich.run(prompt=prompt_text)
                self.replies.append(reply)
            elif indexing_method == "enrich_doc_summary":
                logging.info("Enrich Doc By Summary")
                # prompt = self.response["prompt_enrich_doc_link_summary"]
                prompt = self.response["prompt_enrich_doc_summary"]
                template = Template(prompt)
                prompt_text = template.render(document=document)
                # 记录开始时间
                start_time = time.time()
                reply = self.llm_enrich.run(prompt=prompt_text)
                # 记录结束时间
                end_time = time.time()
                # 计算并打印时间消耗
                time_consumed = end_time - start_time
                print(f"llm_enrich执行时间: {time_consumed:.4f} 秒")
                self.replies.append(reply)
            elif indexing_method == "enrich_doc_summary_key":
                logging.info("Enrich Doc By Summary and Key")
                prompt_summary = self.response["prompt_enrich_doc_summary"]
                template = Template(prompt_summary)
                prompt_text = template.render(document=document)
                reply_summary = self.llm_enrich.run(prompt=prompt_text)

                prompt_key = self.response["prompt_enrich_doc_summary_key"]
                template = Template(prompt_key)
                prompt_text = template.render(document=reply_summary['replies'])
                reply_key = self.llm_enrich.run(prompt=prompt_text)
                self.replies.append(reply_key)
            elif indexing_method == "none":
                self.replies.append(document.content)

            # 更新已处理的文档计数
            self.processed_count += 1

            # 每隔100个保存一次
            if self.processed_count % self.batch_size == 0:
                self._save_replies()

        # 最后一次保存
        self._save_replies()

        return {"replies": self.replies}
def documents_similarity(reranker,documents):
    def calculate_similarity(doc1, doc2):
        result = reranker.run(
            query=doc1.content,
            documents=[doc2]
        )
        return result['documents'][0].score if result else 0

    def find_related_docs(current_id, visited=None, direction=None):
        if visited is None:
            visited = set()

        if current_id in visited:
            return []

        visited.add(current_id)
        related_ids = [current_id]
        current_doc = doc_dict[current_id]

        # 向前查找（上一个文档）
        if direction != 'next' and current_id > min(unique_ids):
            prev_id = current_id - 1
            if prev_id in doc_dict:
                similarity = calculate_similarity(current_doc, doc_dict[prev_id])
                if similarity > 0.5:
                    related_ids.extend(find_related_docs(prev_id, visited, 'prev'))

        # 向后查找（下一个文档）
        if direction != 'prev' and current_id < max(unique_ids):
            next_id = current_id + 1
            if next_id in doc_dict:
                similarity = calculate_similarity(current_doc, doc_dict[next_id])
                if similarity > 0.5:
                    related_ids.extend(find_related_docs(next_id, visited, 'next'))

        return sorted(related_ids)
    # 按unique_id排序文档
    sorted_docs = sorted(documents, key=lambda x: int(x.meta['metadata']['unique_id']))
    print(f"总块数：{len(sorted_docs)}")
    # 创建unique_id到Document的映射字典，便于快速查找
    doc_dict = {int(doc.meta['metadata']['unique_id']): doc for doc in sorted_docs}
    unique_ids = sorted(doc_dict.keys())
    start_time = time.time()
    for i, current_id in enumerate(tqdm(unique_ids, desc="Processing documents", total=len(unique_ids))):
        related_ids = find_related_docs(current_id)
        # 添加chunk_link字段到metadata
        if 'chunk_link' not in documents[i].meta["metadata"]:
            documents[i].meta["metadata"]["chunk_link"] = []
        documents[i].meta["metadata"]["chunk_link"] = related_ids
    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time

    # 打印耗时统计
    print(f"\n处理完成！总共处理了 {len(unique_ids)} 个chunk")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个chunk耗时: {total_time / len(unique_ids):.4f} 秒")
    return documents

def save_similarity_data_to_csv(documents, similarity_dict, csv_file_path):
    """
    将相似度数据和chunk_link信息保存到CSV文件
    """
    sorted_docs = sorted(documents, key=lambda x: int(x.meta['metadata']['unique_id']))

    csv_data = []

    for doc in sorted_docs:
        metadata = doc.meta['metadata']
        unique_id = int(metadata['unique_id'])

        # 获取前后文档的相似度得分
        prev_similarity = similarity_dict.get((unique_id, unique_id - 1), "N/A")
        next_similarity = similarity_dict.get((unique_id, unique_id + 1), "N/A")

        row = {
            'unique_id': unique_id,
            'content': doc.content[:200] + '...' if len(doc.content) > 200 else doc.content,
            'prev_similarity': prev_similarity,
            'next_similarity': next_similarity,
            'chunk_link': '|'.join(map(str, metadata.get('chunk_link', []))),
            'file_name': metadata.get('file_name', 'unknown'),
            'chunk_size': len(doc.content),
            'content_link': metadata.get('content_link', ''),
            'content_link_size': len(metadata.get('content_link', ''))
        }
        csv_data.append(row)

    # 写入CSV文件
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['unique_id', 'content', 'prev_similarity', 'next_similarity',
                      'chunk_link', 'file_name', 'chunk_size','content_link', 'content_link_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

    print(f"相似度数据已保存到: {csv_file_path}")


def documents_similarity_with_storage(reranker, documents, csv_output_dir="files_chunk_link"):
    def calculate_similarity(doc1, doc2):
        result = reranker.run(
            query=doc1.content,
            documents=[doc2]
        )
        return result['documents'][0].score if result else 0

    def find_related_docs(current_id, visited=None, direction=None):
        if visited is None:
            visited = set()

        if current_id in visited:
            return []

        visited.add(current_id)
        related_ids = [current_id]
        current_doc = doc_dict[current_id]

        # 向前查找（上一个文档）
        if direction != 'next' and current_id > min(unique_ids):
            prev_id = current_id - 1
            if prev_id in doc_dict:
                similarity = calculate_similarity(current_doc, doc_dict[prev_id])
                # 存储相似度得分
                similarity_dict[(current_id, prev_id)] = similarity
                if similarity > 0.5:
                    related_ids.extend(find_related_docs(prev_id, visited, 'prev'))

        # 向后查找（下一个文档）
        if direction != 'prev' and current_id < max(unique_ids):
            next_id = current_id + 1
            if next_id in doc_dict:
                similarity = calculate_similarity(current_doc, doc_dict[next_id])
                # 存储相似度得分
                similarity_dict[(current_id, next_id)] = similarity
                if similarity > 0.5:
                    related_ids.extend(find_related_docs(next_id, visited, 'next'))

        return sorted(related_ids)

    # 存储相似度得分的字典
    similarity_dict = {}

    # 按unique_id排序文档
    sorted_docs = sorted(documents, key=lambda x: int(x.meta['metadata']['unique_id']))
    print(f"总块数：{len(sorted_docs)}")

    # 创建unique_id到Document的映射字典，便于快速查找
    doc_dict = {int(doc.meta['metadata']['unique_id']): doc for doc in sorted_docs}
    unique_ids = sorted(doc_dict.keys())

    # 创建一个映射，从unique_id到原始documents中的索引位置
    id_to_index = {}
    for i, doc in enumerate(documents):
        unique_id = int(doc.meta['metadata']['unique_id'])
        id_to_index[unique_id] = i

    start_time = time.time()
    for i, current_id in enumerate(tqdm(unique_ids, desc="Processing documents", total=len(unique_ids))):
        related_ids = find_related_docs(current_id)
        # 合并所有相关文档的内容
        merged_content = []
        for rid in related_ids:
            if rid in doc_dict:
                merged_content.append(doc_dict[rid].content)

        # 将合并后的内容连接成一个字符串
        merged_content_str = "\n".join(merged_content)

        # 获取当前文档在原始documents列表中的索引
        original_index = id_to_index[current_id]
        current_doc = documents[original_index]

        # 添加chunk_link字段到metadata（保持为简单的ID列表）
        if 'chunk_link' not in current_doc.meta["metadata"]:
            current_doc.meta["metadata"]["chunk_link"] = []
        current_doc.meta["metadata"]["chunk_link"] = related_ids

        # 添加content_link字段到metadata（包含合并后的内容）
        if 'content_link' not in current_doc.meta["metadata"]:
            current_doc.meta["metadata"]["content_link"] = ""
        current_doc.meta["metadata"]["content_link"] = merged_content_str[:12000]

        # 更新原始documents列表中的文档
        documents[original_index] = current_doc

    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time

    # 打印耗时统计
    print(f"\n处理完成！总共处理了 {len(unique_ids)} 个chunk")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个chunk耗时: {total_time / len(unique_ids):.4f} 秒")

    # 确保输出目录存在
    os.makedirs(csv_output_dir, exist_ok=True)

    # 从原始文件名中提取基本名称（去掉路径和扩展名）
    if documents:
        base_name = documents[0].meta["metadata"]["file_name"]
    else:
        base_name = "unknown"
    # 生成基于原始文件名的CSV文件名
    csv_filename = f"{base_name}_chunk_link.csv"
    csv_output_path = os.path.join(csv_output_dir, csv_filename)

    # 保存到CSV
    save_similarity_data_to_csv(documents, similarity_dict, csv_output_path)

    print(f"结果已保存到: {csv_output_path}")

    return documents

def documents_similarity_with_storage_load(reranker, documents, csv_output_dir="files_chunk_link"):
    def calculate_similarity(doc1, doc2):
        # 检查是否已经计算过这对文档的相似度
        pair_key1 = (int(doc1.meta['metadata']['unique_id']), int(doc2.meta['metadata']['unique_id']))
        pair_key2 = (pair_key1[1], pair_key1[0])  # 反向键，因为相似度是对称的

        # 首先检查当前计算会话中是否已经计算过
        if pair_key1 in similarity_dict:
            return similarity_dict[pair_key1]
        if pair_key2 in similarity_dict:
            return similarity_dict[pair_key2]

        # 然后检查已存储的数据中是否有这个相似度
        doc1_id = int(doc1.meta['metadata']['unique_id'])
        doc2_id = int(doc2.meta['metadata']['unique_id'])

        # 检查是否是相邻文档（prev/next关系）
        if abs(doc1_id - doc2_id) == 1:
            smaller_id = min(doc1_id, doc2_id)
            larger_id = max(doc1_id, doc2_id)

            # 检查是否在已加载的数据中
            if smaller_id in loaded_chunk_data:
                chunk_info = loaded_chunk_data[smaller_id]
                if larger_id == smaller_id + 1:  # next关系
                    if chunk_info['next_similarity'] is not None:
                        similarity_dict[pair_key1] = chunk_info['next_similarity']
                        return chunk_info['next_similarity']
                elif larger_id == smaller_id - 1:  # prev关系（理论上不会出现，因为smaller_id是较小的）
                    pass

            if larger_id in loaded_chunk_data:
                chunk_info = loaded_chunk_data[larger_id]
                if smaller_id == larger_id - 1:  # prev关系
                    if chunk_info['prev_similarity'] is not None:
                        similarity_dict[pair_key1] = chunk_info['prev_similarity']
                        return chunk_info['prev_similarity']

        # 如果没有找到已存储的相似度，则进行计算
        result = reranker.run(
            query=doc1.content,
            documents=[doc2]
        )
        similarity_score = result['documents'][0].score if result else 0
        similarity_dict[pair_key1] = similarity_score
        return similarity_score

    def find_related_docs(current_id, visited=None, direction=None):
        if visited is None:
            visited = set()

        if current_id in visited:
            return []

        visited.add(current_id)
        related_ids = [current_id]
        current_doc = doc_dict[current_id]

        # 首先检查已存储的数据中是否有chunk_link信息
        if current_id in loaded_chunk_data and not force_recalculate:
            stored_links = loaded_chunk_data[current_id]['chunk_link']
            if stored_links and len(stored_links) > 1:  # 如果有存储的链接且不止自己
                # 使用存储的链接，但需要验证主要链接的相似度
                for linked_id in stored_links:
                    if linked_id != current_id and linked_id in doc_dict and linked_id not in visited:
                        # 对于直接相邻的文档，检查相似度是否仍然有效
                        if abs(linked_id - current_id) == 1:
                            similarity = calculate_similarity(current_doc, doc_dict[linked_id])
                            if similarity > 0.5:
                                related_ids.extend(find_related_docs(linked_id, visited,
                                                                     'prev' if linked_id < current_id else 'next'))
                        else:
                            # 对于非直接相邻的文档，直接使用存储的链接
                            related_ids.extend(find_related_docs(linked_id, visited, None))
                return sorted(related_ids)

        # 如果没有存储的数据或强制重新计算，则进行原始计算
        # 向前查找（上一个文档）
        if direction != 'next' and current_id > min(unique_ids):
            prev_id = current_id - 1
            if prev_id in doc_dict:
                similarity = calculate_similarity(current_doc, doc_dict[prev_id])
                if similarity > 0.5:
                    related_ids.extend(find_related_docs(prev_id, visited, 'prev'))

        # 向后查找（下一个文档）
        if direction != 'prev' and current_id < max(unique_ids):
            next_id = current_id + 1
            if next_id in doc_dict:
                similarity = calculate_similarity(current_doc, doc_dict[next_id])
                if similarity > 0.5:
                    related_ids.extend(find_related_docs(next_id, visited, 'next'))

        return sorted(related_ids)

    # 存储相似度得分的字典
    similarity_dict = {}

    # 按unique_id排序文档
    sorted_docs = sorted(documents, key=lambda x: int(x.meta['metadata']['unique_id']))
    print(f"总块数：{len(sorted_docs)}")

    # 创建unique_id到Document的映射字典，便于快速查找
    doc_dict = {int(doc.meta['metadata']['unique_id']): doc for doc in sorted_docs}
    unique_ids = sorted(doc_dict.keys())

    # 创建一个映射，从unique_id到原始documents中的索引位置
    id_to_index = {}
    for i, doc in enumerate(documents):
        unique_id = int(doc.meta['metadata']['unique_id'])
        id_to_index[unique_id] = i

    # 从原始文件名中提取基本名称（去掉路径和扩展名）
    if documents:
        base_name = documents[0].meta["metadata"]["file_name"]
    else:
        base_name = "unknown"

    # 生成基于原始文件名的CSV文件名
    csv_filename = f"{base_name}_chunk_link.csv"
    csv_output_path = os.path.join(csv_output_dir, csv_filename)

    # 加载已存储的chunk数据
    loaded_chunk_data = {}
    if os.path.exists(csv_output_path):
        loaded_chunk_data = load_chunk_data_from_csv(csv_output_path)
        print(f"已加载 {len(loaded_chunk_data)} 个已存储的chunk数据")

    # 是否强制重新计算所有相似度（如果文件已存在但想要重新计算，可以设置此参数）
    force_recalculate = False

    start_time = time.time()
    processed_count = 0
    skipped_count = 0

    for i, current_id in enumerate(tqdm(unique_ids, desc="Processing documents", total=len(unique_ids))):
        # 检查是否已经有完整的存储数据且不需要重新计算
        if (current_id in loaded_chunk_data and not force_recalculate and
                loaded_chunk_data[current_id]['chunk_link'] and
                len(loaded_chunk_data[current_id]['chunk_link']) > 1):

            # 使用存储的chunk_link
            related_ids = loaded_chunk_data[current_id]['chunk_link']
            skipped_count += 1
        else:
            # 需要重新计算
            related_ids = find_related_docs(current_id)
            processed_count += 1

        # 合并所有相关文档的内容
        merged_content = []
        for rid in related_ids:
            if rid in doc_dict:
                merged_content.append(doc_dict[rid].content)

        # 将合并后的内容连接成一个字符串
        merged_content_str = "\n".join(merged_content)

        # 获取当前文档在原始documents列表中的索引
        original_index = id_to_index[current_id]
        current_doc = documents[original_index]

        # 添加chunk_link字段到metadata（保持为简单的ID列表）
        if 'chunk_link' not in current_doc.meta["metadata"]:
            current_doc.meta["metadata"]["chunk_link"] = []
        current_doc.meta["metadata"]["chunk_link"] = related_ids

        # 添加content_link字段到metadata（包含合并后的内容）
        if 'content_link' not in current_doc.meta["metadata"]:
            current_doc.meta["metadata"]["content_link"] = ""
        current_doc.meta["metadata"]["content_link"] = merged_content_str[:12000]

        # 更新原始documents列表中的文档
        documents[original_index] = current_doc

    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time

    # 打印耗时统计
    print(f"\n处理完成！总共处理了 {len(unique_ids)} 个chunk")
    print(f"重新计算了 {processed_count} 个chunk，跳过了 {skipped_count} 个已存储的chunk")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个chunk耗时: {total_time / len(unique_ids):.4f} 秒")

    # 确保输出目录存在
    os.makedirs(csv_output_dir, exist_ok=True)

    # 保存到CSV
    save_similarity_data_to_csv(documents, similarity_dict, csv_output_path)

    print(f"结果已保存到: {csv_output_path}")

    return documents
def load_chunk_data_from_csv(csv_file_path):
    """
    从CSV文件加载chunk数据
    """
    chunk_data = {}

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                unique_id = int(row['unique_id'])
                chunk_data[unique_id] = {
                    'prev_similarity': float(row['prev_similarity']) if row['prev_similarity'] != 'N/A' else None,
                    'next_similarity': float(row['next_similarity']) if row['next_similarity'] != 'N/A' else None,
                    'chunk_link': [int(x) for x in row['chunk_link'].split('|') if x],
                    'file_name': row['file_name']
                }
        print(f"已从 {csv_file_path} 加载 {len(chunk_data)} 个chunk的数据")
    except FileNotFoundError:
        print(f"文件 {csv_file_path} 不存在")
        return {}

    return chunk_data
@component
class Updated_Meta_Info_old:
    """
    一个自定义组件，将输入的documents: List[Document]元数据信息修改为存入metadata,输出 documents=List[Document]
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document],len_limit: int):
        logging.info("*****进行Updated_Meta_Info组件，构建metadata*****")
        for doc in documents:
            # 检查content并限制长度
            if doc.content:
                doc.content = doc.content[:len_limit]
            # 检查并迁移page_idx元数据
            if "page_idx" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["page_idx"] = doc.meta["page_idx"]
                del doc.meta["page_idx"]
                logging.debug("成功迁移文档[%s]的page_idx到metadata", doc.id)
            else:
                logging.debug("文档[%s]无page_idx信息，跳过迁移", doc.id)
            # 检查并迁移img_path元数据
            if "img_path" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["img_path"] = doc.meta["img_path"]
                del doc.meta["img_path"]
                logging.debug("成功迁移文档[%s]的img_path到metadata", doc.id)
            else:
                logging.debug("文档[%s]无img_path信息，跳过迁移", doc.id)
            # 检查并迁移unique_id元数据
            if "unique_id" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["unique_id"] = doc.meta["unique_id"]
                del doc.meta["unique_id"]
                logging.debug("成功迁移文档[%s]的unique_id到metadata", doc.id)
            else:
                logging.debug("文档[%s]无unique_id信息，跳过迁移", doc.id)
            # 检查并迁移next_text_level元数据
            if "next_text_level" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["next_text_level"] = doc.meta["next_text_level"]
                del doc.meta["next_text_level"]
                logging.debug("成功迁移文档[%s]的next_text_level到metadata", doc.id)
            else:
                logging.debug("文档[%s]无next_text_level信息，跳过迁移", doc.id)
            # 检查并迁移previous_text_level元数据
            if "previous_text_level" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["previous_text_level"] = doc.meta["previous_text_level"]
                del doc.meta["previous_text_level"]
                logging.debug("成功迁移文档[%s]的previous_text_level到metadata", doc.id)
            else:
                logging.debug("文档[%s]无previous_text_level信息，跳过迁移", doc.id)
            # 检查并迁移parallel_text_level元数据
            if "parallel_text_level" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["parallel_text_level"] = doc.meta["parallel_text_level"]
                del doc.meta["parallel_text_level"]
                logging.debug("成功迁移文档[%s]的parallel_text_level到metadata", doc.id)
            else:
                logging.debug("文档[%s]无parallel_text_level信息，跳过迁移", doc.id)
            # 检查并迁移file_name元数据
            if "file_name" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["file_name"] = doc.meta["file_name"]
                del doc.meta["file_name"]
                logging.debug("成功迁移文档[%s]的file_name到metadata", doc.id)
            else:
                logging.debug("文档[%s]无file_name信息，跳过迁移", doc.id)
            # 检查并迁移type元数据
            if "type" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["type"] = doc.meta["type"]
                del doc.meta["type"]
                logging.debug("成功迁移文档[%s]的type到metadata", doc.id)
            else:
                logging.debug("文档[%s]无type信息，跳过迁移", doc.id)
            # 检查并迁移text_level元数据
            if "text_level" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["text_level"] = doc.meta["text_level"]
                del doc.meta["text_level"]
                logging.debug("成功迁移文档[%s]的text_level到metadata", doc.id)
            else:
                logging.debug("文档[%s]无text_level信息，跳过迁移", doc.id)
            # 检查并迁移table_caption元数据
            if "table_caption" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                if doc.meta["table_caption"]:
                    doc.meta["metadata"]["table_caption"] = doc.meta["table_caption"][:len_limit]
                else:
                    doc.meta["metadata"]["table_caption"] = doc.meta["table_caption"]
                del doc.meta["table_caption"]
                logging.debug("成功迁移文档[%s]的table_caption到metadata", doc.id)
            else:
                logging.debug("文档[%s]无table_caption信息，跳过迁移", doc.id)
            # 检查并迁移table_footnote元数据
            if "table_footnote" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                if doc.meta["table_footnote"]:
                    doc.meta["metadata"]["table_footnote"] = doc.meta["table_footnote"][:len_limit]
                else:
                    doc.meta["metadata"]["table_footnote"] = doc.meta["table_footnote"]
                del doc.meta["table_footnote"]
                logging.debug("成功迁移文档[%s]的table_footnote到metadata", doc.id)
            else:
                logging.debug("文档[%s]无table_footnote信息，跳过迁移", doc.id)

            # 检查并迁移sheet元数据
            if "sheet" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["sheet"] = doc.meta["sheet"]
                del doc.meta["sheet"]
                logging.debug("成功迁移文档[%s]的sheet到metadata", doc.id)
            else:
                logging.debug("文档[%s]无sheet信息，跳过迁移", doc.id)

            # 检查并迁移version元数据
            if "version" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["version"] = doc.meta["version"]
                del doc.meta["version"]
                logging.debug("成功迁移文档[%s]的version到metadata", doc.id)
            else:
                logging.debug("文档[%s]无version信息，跳过迁移", doc.id)
            # 检查并迁移url元数据
            if "url" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                if doc.meta["url"]:
                    doc.meta["metadata"]["url"] = doc.meta["url"][:len_limit]
                else:
                    doc.meta["metadata"]["url"] = doc.meta["url"]
                del doc.meta["url"]
                logging.debug("成功迁移文档[%s]的url到metadata", doc.id)
            else:
                logging.debug("文档[%s]无url信息，跳过迁移", doc.id)
            # 检查并迁移version元数据
            if "ori_id_1" in doc.meta:
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["ori_id_1"] = doc.meta["ori_id_1"]
                del doc.meta["ori_id_1"]
                logging.debug("成功迁移文档[%s]的ori_id_1到metadata", doc.id)
            else:
                logging.debug("文档[%s]无ori_id_1信息，跳过迁移", doc.id)
        # # 添加chunks_link
        # reranker = FastAPIRanker(url="http://0.0.0.0:11500/v1/rerank", top_k=10)
        # # documents = documents_similarity(reranker, documents)
        # documents=documents_similarity_with_storage_load(reranker, documents,csv_output_dir="YOUR_OUTPUT_DIR")
        logging.info("Updated_Meta_Info组件执行完成，共处理 %d 个文档", len(documents))
        return {"documents": documents}


@component
class Updated_Meta_Info:
    """
    一个自定义组件，将输入的documents: List[Document]元数据信息修改为存入metadata,输出 documents=List[Document]
    使用EmbeddingTokenizer进行token级别的截断，而不是字符串长度限制
    """

    def __init__(self, retrieval_config: dict = None):
        """
        初始化Updated_Meta_Info组件

        Args:
            retrieval_config: 检索配置，用于获取EmbeddingTokenizer配置参数
        """
        # 初始化EmbeddingTokenizer用于token级别截断
        self.embedding_tokenizer = EmbeddingTokenizer(retrieval_config=retrieval_config)

    def _limit_string_by_tokens(self, value: Any) -> Any:
        """
        基于token数量限制字符串长度的方法
        注意：max_length已在EmbeddingTokenizer初始化时从retrieval_config中设置

        Args:
            value: 要处理的值，如果是字符串则进行截断

        Returns:
            处理后的值
        """
        if not isinstance(value, str) or value is None or value == "":
            return value

        try:
            # 使用EmbeddingTokenizer计算token数量（使用初始化时配置的max_length）
            result = self.embedding_tokenizer.tokenize(
                input_text=value,
                logging=logging
            )
            # 获取截断后的文本（已移除<s>和</s>标签）
            truncated_texts = result.get("truncated_texts", [])
            if truncated_texts:
                return truncated_texts[0]
            return value
        except Exception as e:
            logging.warning(f"Token截断失败，使用原值: {str(e)}")
            return value

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        logging.info("*****进行Updated_Meta_Info组件，构建metadata*****")
        # 从环境变量获取入库用的 knowledge_uuid
        import os
        knowledge_uuid = os.getenv("KB_BUILD_KNOWLEDGE_UUID")
        if not knowledge_uuid:
            raise ValueError("环境变量 KB_BUILD_KNOWLEDGE_UUID 未设置，请先设置该环境变量")
        print(f"使用 knowledge_uuid: {knowledge_uuid}")

        meta_fields_to_migrate = {
            "page_idx": False,
            "img_path": False,
            "unique_id": False,
            "next_text_level": False,
            "previous_text_level": False,
            "parallel_text_level": False,
            "file_name": False,
            "type": False,
            "text_level": False,
            "table_caption": True,
            "table_footnote": True,
            "sheet": False,
            "version": True,
            "url": True,
            "ori_id_1": False
        }
        with LoggedTime("Updated_Meta_Info组件中全文embedding模型tokens计算耗时统计"):
            for doc in documents:

                # 检查content并限制长度（使用token级别截断）
                if doc.content:
                    doc.content = self._limit_string_by_tokens(doc.content)

                # 确保metadata字典存在
                if "metadata" not in doc.meta:
                    doc.meta["metadata"] = {}
                doc.meta["metadata"]["knowledge_uuid"] = knowledge_uuid
                # 遍历所有需要迁移的字段
                for field_name, needs_length_limit in meta_fields_to_migrate.items():
                    if field_name in doc.meta:
                        value = doc.meta[field_name]

                        # 根据字段类型决定是否进行长度限制
                        if needs_length_limit:
                            value = self._limit_string_by_tokens(value)

                        doc.meta["metadata"][field_name] = value
                        del doc.meta[field_name]
                        logging.debug("成功迁移文档[%s]的%s到metadata", doc.id, field_name)
                    else:
                        logging.debug("文档[%s]无%s信息，跳过迁移", doc.id, field_name)

            # # 添加chunks_link
            # reranker = FastAPIRanker(url="http://0.0.0.0:11500/v1/rerank", top_k=10)
            # # documents = documents_similarity(reranker, documents)
            # documents=documents_similarity_with_storage_load(reranker, documents,csv_output_dir="YOUR_OUTPUT_DIR")
            logging.info("Updated_Meta_Info组件执行完成，共处理 %d 个文档", len(documents))
            return {"documents": documents}


@component
class LLM_Evaluation:
    """
    一个自定义组件，利用llm评估首页信息能否回答问题,输出 replies=str
    """

    def __init__(self, model_platform, model, llm_enrich_model_url, temperature, num_predict, timeout):
        self.model_platform = model_platform
        self.model = model
        self.llm_enrich_model_url = llm_enrich_model_url
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

    @component.output_types(reply=str)
    def run(self, documents: str, query: str,flag:str,logging):
        from jinja2 import Template
        from retriever.configs.response import prompt_is_homepage,prompt_is_relevant
        prompt=""
        if flag=="homepage":
            logging.info(f"开始使用LLM评估首页信息是否能回答问题")
            prompt = prompt_is_homepage
        elif flag=="similarity":
            logging.info(f"开始使用LLM评估文档问题相关性")
            prompt = prompt_is_relevant
        template = Template(prompt)
        prompt_text = template.render(documents=documents, query=query)

        if self.model_platform == "openai":
            from haystack.components.generators import OpenAIGenerator

            llm_enrich = OpenAIGenerator(model=self.model, api_base_url=self.llm_enrich_model_url,
                                         generation_kwargs={"temperature": self.temperature,
                                                            "max_tokens": self.num_predict,
                                                            "timeout": self.timeout})
        elif self.model_platform == "ollama":
            from haystack_integrations.components.generators.ollama.generator import OllamaGenerator

            llm_enrich = OllamaGenerator(model=self.model, url=self.llm_enrich_model_url,
                                         generation_kwargs={"temperature": self.temperature,
                                                            "num_predict": self.num_predict,
                                                            "timeout": self.timeout})
        reply = llm_enrich.run(prompt=prompt_text)
        logging.info(f"LLM评估信息完成")

        return {"reply": reply}


@component
class Data_To_Written:
    """
    将文档列表转换为适用于Milvus向量数据库的插入字典。

    Args:
        documents (List[Document]): 文档对象列表，每个文档对象应包含content、embedding、sparse_embedding和meta属性。

    Returns:
        Dict[str, List]: 一个字典，包含要插入Milvus向量数据库的字段及其对应的值列表。
            - text: 文档内容的文本字段列表。
            - embedding: 文档嵌入向量的列表。
            - id: 文档的唯一标识符列表。
            - sparse_embedding: 文档的稀疏嵌入字典列表，每个字典的键为索引，值为对应的稀疏向量值。
            - metadata中的其他字段（如存在）：这些字段及其对应的值列表也会被包含在返回的字典中。

    Raises:
        ValueError: 如果documents参数中的对象不是Document类型的实例列表，则引发此异常。

    """
    from haystack.dataclasses.sparse_embedding import SparseEmbedding

    def _convert_sparse_to_dict(self, sparse_embedding: SparseEmbedding) -> Dict:
        return dict(zip(sparse_embedding.indices, sparse_embedding.values))

    @component.output_types(insert_dict=dict)
    def run(self, documents: List[Document]):
        from milvus_haystack import MilvusDocumentStore
        from copy import deepcopy
        from haystack.dataclasses.sparse_embedding import SparseEmbedding
        logging.info("*****进行Data_To_Written模块，将文档列表转换为适用于Milvus向量数据库的插入字典*****")
        _dummy_value = 999.0
        text_field: str = "text"
        vector_field: str = "embedding"
        primary_field: str = "id"
        sparse_vector_field = "sparse_embedding"
        # 向量库中该有的字段
        fields: List[str] = ['metadata', 'text', 'id', 'embedding', 'sparse_embedding']
        documents_cp = [MilvusDocumentStore._discard_invalid_meta(doc) for doc in deepcopy(documents)]
        if len(documents_cp) > 0 and not isinstance(documents_cp[0], Document):
            err_msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(err_msg)
        # Check embeddings
        embedding_dim = 128
        for doc in documents_cp:
            if doc.embedding:
                embedding_dim = len(doc.embedding)
                break

        for doc_idx, doc in enumerate(documents_cp):
            if doc.embedding is None:
                dummy_vector = [_dummy_value] * embedding_dim
                doc.embedding = dummy_vector
                logging.debug(f"文档[{doc.id}]的embedding为空，已填充默认向量长度{embedding_dim}")
            if doc.sparse_embedding is None:
                dummy_sparse_vector = SparseEmbedding(
                    indices=[0],
                    values=[_dummy_value],
                )
                doc.sparse_embedding = dummy_sparse_vector
                logging.debug(f"文档[{doc.id}]的sparse_embedding为空，已填充默认稀疏向量")
            if doc.content is None:
                doc.content = ""
                logging.warning(f"文档[{doc.id}]的内容(content)为空，已填充空字符串")
        embeddings = [doc.embedding for doc in documents_cp]
        sparse_embeddings = [self._convert_sparse_to_dict(doc.sparse_embedding) for doc in documents_cp]
        metas = [doc.meta for doc in documents_cp]
        texts = [doc.content for doc in documents_cp]
        ids = [doc.id for doc in documents_cp]

        # Dict to hold all insert columns
        insert_dict: Dict[str, List] = {text_field: texts, vector_field: embeddings, primary_field: ids,
                                        sparse_vector_field: sparse_embeddings}
        # Collect the meta into the insert dict.
        if metas is not None:
            for d_idx, d in enumerate(metas):
                for key, value in d.items():
                    if key in fields:
                        insert_dict.setdefault(key, []).append(value)
                        logging.debug(f"文档[{documents_cp[d_idx].id}]的元数据字段'{key}'已添加到插入字典")

        logging.info(f"成功构建插入字典，包含{len(documents_cp)}条数据，字段：{list(insert_dict.keys())}")
        return insert_dict


@component
class Homepage_Info:
    """
    一个自定义组件，将输入的documents: List[Document]的首页信息存储为本地json文件,输出 documents=List[Document]
    """

    def __init__(self, file_name, file_path):
        self.file_name = file_name
        self.file_path = file_path

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        import io
        homepage_documents = []
        file_path = self.file_path
        for doc in documents:
            # 安全地获取page_idx，如果不存在则返回None
            page_idx = doc.meta.get("metadata", {}).get("page_idx")
            # 检查page_idx是否存在且等于0
            if page_idx == 0:
                homepage_documents.append({"content": [doc.content[:1000]]})
        # 尝试读取现有数据
        try:
            with io.open(file_path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = {}
        # 检查self.file_name是否已存在
        if self.file_name in existing_data:
            print(f"Warning: '{self.file_name}' already exists in the file. Skipping write operation.")
        else:
            # 添加新数据
            existing_data[self.file_name] = homepage_documents
            # 写入文件
            with io.open(file_path, 'w', encoding='utf-8') as file:
                json.dump(existing_data, file, indent=4, ensure_ascii=False)
        return {"documents": documents}


class TableRecoveryProcessor:
    def __init__(self):
        self.current_data_img_paths = []

    def process_documents(self, documents, query_handler,retrieval_fetchType,knowledgeUUID_list,parseType,logging=logging, skip_knowledge_uuids=None):
        """
        处理文档进行表格复原

        Args:
            skip_knowledge_uuids: 需要跳过表格复原的知识库UUID集合
        """
        if skip_knowledge_uuids is None:
            skip_knowledge_uuids = set()

        processed_docs = []
        if not documents:  # 会检测 None、空列表、空元组、空字典等
            logging.warning("documents为空，无需表格还原")
            return documents

        for doc in documents:
            # 获取文档的knowledge_uuid
            doc_knowledge_uuid = doc.meta.get("metadata", {}).get("knowledge_uuid", "")

            # 检查是否需要跳过该文档的表格复原
            if doc_knowledge_uuid in skip_knowledge_uuids:
                logging.info(f"文档knowledge_uuid为{doc_knowledge_uuid}，跳过表格复原")
                processed_docs.append(doc)
                continue

            # 获取metadata字典，确保安全访问
            doc_metadata = doc.meta.get("metadata", {})

            logging.info(f"表格还原-处理文档详细信息:\n"
                        f"  id={doc.id}\n"
                        f"  knowledge_uuid={doc_knowledge_uuid}\n"
                        f"  file_name={doc_metadata.get('file_name', '')}\n"
                        f"  type={doc_metadata.get('type', '')}\n"
                        f"  content_length={doc.content and len(doc.content) or 0}\n"
                        f"  content_preview={doc.content[:200] if doc.content else 'None'}...\n"
                        f"  meta={doc.meta}")

            doc_file_name = doc_metadata.get("file_name")
            if not doc_file_name:
                logging.warning(f"文档id:{doc.id}缺少file_name字段，跳过表格还原")
                processed_docs.append(doc)
                continue

            # 安全检查：判断是否为excel文件，检查sheet键是否存在且为True
            if doc_metadata.get("sheet", False):
                logging.warning(f"{doc_file_name}为excel文件，无需表格还原")
                processed_docs.append(doc)
                continue

            if doc_metadata.get("type") == "table":
                with LoggedTime("进行表格复原耗时统计"):
                    full_table,full_uid = self.recover_table(query_handler,retrieval_fetchType, doc.id,doc_file_name,knowledgeUUID_list,parseType,logging)
                if full_table is not None and len(full_table) > 0:  # 如果有表格内容则替换
                    doc.content = "\n".join(full_table)
                    if "metadata" not in doc.meta:
                        doc.meta["metadata"] = {}
                    doc.meta["metadata"]["unique_id"] = full_uid
                processed_docs.append(doc)  # 其他情况都保留原始文档
            else:
                logging.warning(f"id:{doc.id}的类型不是table，无需表格还原")
                processed_docs.append(doc)
                continue
        return processed_docs
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
    def recover_table_filters(self, knowledgeUUID_list, file_name, doc_id):
        try:
            # 为每个 knowledgeUUID 构建条件
            expr_list = []
            for knowledgeUUID in knowledgeUUID_list:
                # 修正语法错误，添加引号
                expr = f'metadata["knowledge_uuid"] == "{knowledgeUUID}" && metadata["file_name"] == "{file_name}" && doc_id == "{doc_id}"'
                expr_list.append(expr)

            expr = " || ".join(expr_list)
            logging.info(f"过滤检索接口-表格还原-过滤符：{expr}")
            return expr
        except Exception as e:
            logging.error(f"过滤检索接口-表格还原-构建过滤符错误: {e}")
            return ""
    def recover_table_filters2(self, knowledgeUUID_list, img_path):
        try:
            # 为每个 knowledgeUUID 构建条件
            expr_list = []
            for knowledgeUUID in knowledgeUUID_list:
                # 修正语法错误，添加引号
                expr = f'metadata["knowledge_uuid"] == "{knowledgeUUID}" && metadata["img_path"] == "{img_path}"'
                expr_list.append(expr)

            expr = " || ".join(expr_list)
            logging.info(f"过滤检索接口-表格还原img_path-过滤符：{expr}")
            return expr
        except Exception as e:
            logging.error(f"过滤检索接口-表格还原img_path-构建过滤符错误: {e}")
            return ""
    def recover_table(self, query_handler, retrieval_fetchType, id, file_name, knowledgeUUID_list, parseType, logging):
        full_table = []
        full_uid = []

        # 获取 table_docs 的逻辑保持不变
        if retrieval_fetchType == "local":
            table_docs = query_handler.query_documents(
                filters={
                    "operator": "AND",
                    "conditions": [
                        {
                            "field": "metadata['file_name']",
                            "operator": "==",
                            "value": file_name
                        },
                        {
                            "field": "doc_id",
                            "operator": "==",
                            "value": id
                        }
                    ]
                }
            )
        else:
            expr = self.recover_table_filters(knowledgeUUID_list, file_name, id)
            table_docs = query_handler.call_milvus_filtered_retrieval(expr=expr,knowledgeUUID_list=knowledgeUUID_list)

        if not table_docs:
            logging.warning(f"{id}：在milvus的text中未找到相同信息，无法进行表格还原")
            return full_table, full_uid

        current_data_img_path = table_docs[0].meta["metadata"]["img_path"]
        if not current_data_img_path:
            unique_id = table_docs[0].meta["metadata"]["unique_id"]
            logging.warning(f"uid:{unique_id}，的图片路径为空，无法进行表格还原")
            return full_table, full_uid

        # 修改去重逻辑：考虑 knowledgeUUID + img_path
        if retrieval_fetchType == "api":
            # 对于 API 模式，需要结合 knowledgeUUID 来去重
            knowledge_uuid = table_docs[0].meta["metadata"].get("knowledge_uuid")
            combined_key = f"{knowledge_uuid}_{current_data_img_path}" if knowledge_uuid else current_data_img_path
        else:
            combined_key = current_data_img_path

        if combined_key in self.current_data_img_paths:
            logging.info(f"图片路径 {current_data_img_path} (key: {combined_key}) 已处理过，跳过此文档")
            return None, None

        # 查询特定图片路径的内容（根据模式添加过滤条件）
        if retrieval_fetchType == "local":
            img_path_filters = {
                "field": "metadata['img_path']",
                "operator": "==",
                "value": current_data_img_path
            }
            img_path_docs = query_handler.query_documents(filters=img_path_filters)
        else:
            # API 模式：需要同时过滤 img_path 和 knowledgeUUID
            knowledge_uuid = table_docs[0].meta["metadata"]["knowledge_uuid"]
            knowledgeUUID_list = self.split_knowledge_uuid(knowledge_uuid_str=knowledge_uuid)
            img_path_filters = {
                "operator": "AND",
                "conditions": [
                    {
                        "field": "metadata['img_path']",
                        "operator": "==",
                        "value": current_data_img_path
                    },
                    {
                        "field": "metadata['knowledge_uuid']",
                        "operator": "==",
                        "value": knowledge_uuid
                    }
                ]
            }
            expr = self.recover_table_filters2(knowledgeUUID_list, current_data_img_path)
            img_path_docs = query_handler.call_milvus_filtered_retrieval(expr=expr, knowledgeUUID_list=knowledgeUUID_list)

        if not img_path_docs:
            logging.warning(f"在milvus中未找到图片路径 {current_data_img_path} 对应的文档")
        else:
            logging.info(f"找到图片路径 {current_data_img_path} 对应的文档，共 {len(img_path_docs)} 条")
            for doc in img_path_docs:
                full_table.append(doc.content)
                ori_id = doc.meta["metadata"].get("ori_id_1", [])
                uid_to_append = ori_id if ori_id else [doc.meta["metadata"]["unique_id"]]
                full_uid.append(uid_to_append)
                logging.info(f"添加文档内容到表格: {doc.content[:10]}...")

        logging.info(f"添加复原的表格对应的uid: {full_uid}...")
        self.current_data_img_paths.append(combined_key)  # 使用组合键
        logging.info(f"图片路径 {current_data_img_path} 已添加到已处理列表")

        return full_table, full_uid
@component
class FastAPIRanker(TransformersSimilarityRanker):
    def __init__(
            self,
        top_k: int = 10,
        query_prefix: str = "",
        document_prefix: str = "",
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        scale_score: bool = True,
        calibration_factor: Optional[float] = 1.0,
        score_threshold: Optional[float] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 16,
            url:str=''):
        """
        初始化FastAPIRanker

        Args:
            url: FastAPI服务的URL
            **kwargs: 其他传递给父类的参数
        """
        self.url = url
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.tokenizer = None
        self.device = None
        self.top_k = top_k
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.scale_score = scale_score
        self.calibration_factor = calibration_factor
        self.score_threshold = score_threshold
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.batch_size = batch_size

    def warm_up(self):
        pass
    @component.output_types(documents=List[Document])
    def run(
            self,
            query: str,
            documents: List[Document],
            top_k: Optional[int] = None,
            scale_score: Optional[bool] = None,
            calibration_factor: Optional[float] = None,
            score_threshold: Optional[float] = None,
    ):
        import requests
        if not documents:
            return {"documents": []}

        top_k = self.top_k
        scale_score = scale_score or self.scale_score
        calibration_factor = calibration_factor or self.calibration_factor
        score_threshold = score_threshold or self.score_threshold

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        if scale_score and calibration_factor is None:
            raise ValueError(
                f"scale_score is True so calibration_factor must be provided, but got {calibration_factor}"
            )

        # Prepare the documents for the API call
        doc_contents = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            text_to_embed = self.embedding_separator.join(meta_values_to_embed + [doc.content or ""])
            # text_to_embed = self.embedding_separator.join(meta_values_to_embed + [doc.meta['metadata']['link_summary'] or ""])
            doc_contents.append(text_to_embed)

        # Call the rerank API
        headers = {"Content-Type": "application/json"}

        data = {
            "query": query,
            "documents": doc_contents
        }

        try:
            with LoggedTime("reranker模块耗时统计"):
                results = requests.post(self.url, json=data, headers=headers)
            results.raise_for_status()
            results = results.json()

            ranked_docs = []
            for sorted_index in results["results"]:
                i = sorted_index['indices']
                documents[i].score = sorted_index['score']
                ranked_docs.append(documents[i])

            if score_threshold is not None:
                ranked_docs = [doc for doc in ranked_docs if doc.score >= score_threshold]

            return {"documents": ranked_docs[:top_k]}

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to call rerank API: {str(e)}")


@component
class XinferenceReranker(TransformersSimilarityRanker):
    def __init__(
            self,
            top_k: int = 10,
            query_prefix: str = "",
            document_prefix: str = "",
            meta_fields_to_embed: Optional[List[str]] = None,
            embedding_separator: str = "\n",
            scale_score: bool = True,
            calibration_factor: Optional[float] = 1.0,
            score_threshold: Optional[float] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            batch_size: int = 32,
            host: str = "127.0.0.1",
            port: str = "9997",
            model_uid: str = "bge-reranker-large",
            api_key: str = ""
    ):
        """
        初始化XinferenceReranker

        Args:
            host: Xinference服务器地址
            port: Xinference服务器端口
            model_uid: 模型UID
            api_key: API密钥
            **kwargs: 其他传递给父类的参数
        """
        self.host = host
        self.port = port
        self.model_uid = model_uid
        self.api_key = api_key
        self.url = f"http://{host}:{port}/v1/rerank"
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.tokenizer = None
        self.device = None
        self.top_k = top_k
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.scale_score = scale_score
        self.calibration_factor = calibration_factor
        self.score_threshold = score_threshold
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.batch_size = batch_size

    def warm_up(self):
        pass

    @component.output_types(documents=List[Document])
    def run(
            self,
            query: str,
            documents: List[Document],
            top_k: Optional[int] = None,
            scale_score: Optional[bool] = None,
            calibration_factor: Optional[float] = None,
            score_threshold: Optional[float] = None,
    ):
        import requests
        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        scale_score = scale_score or self.scale_score
        calibration_factor = calibration_factor or self.calibration_factor
        score_threshold = score_threshold or self.score_threshold

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        if scale_score and calibration_factor is None:
            raise ValueError(
                f"scale_score is True so calibration_factor must be provided, but got {calibration_factor}"
            )

        # Prepare the documents for the API call
        doc_contents = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            text_to_embed = self.embedding_separator.join(meta_values_to_embed + [doc.content or ""])
            doc_contents.append(text_to_embed)

        # Prepare the request data
        data = {
            "model": self.model_uid,
            "query": query,
            "documents": doc_contents
        }

        try:
            with LoggedTime("reranker模块耗时统计"):
                response = requests.post(self.url, headers=self.headers, json=data)
            response.raise_for_status()
            results = response.json()

            # Rank documents based on the response
            ranked_docs = []
            for result in results["results"]:
                doc_index = result["index"]
                score = result["relevance_score"]
                documents[doc_index].score = score
                ranked_docs.append(documents[doc_index])

            # Sort documents by score in descending order
            ranked_docs.sort(key=lambda x: x.score, reverse=True)

            if score_threshold is not None:
                ranked_docs = [doc for doc in ranked_docs if doc.score >= score_threshold]

            return {"documents": ranked_docs[:top_k]}

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to call rerank API: {str(e)}")
@component
class FastAPIDocumentEmbedder(FastembedSparseDocumentEmbedder):
    def __init__(
        self,
        model: str = "prithivida/Splade_PP_en_v1",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        parallel: Optional[int] = None,
        local_files_only: bool = False,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model
        self.cache_dir = cache_dir
        self.threads = threads
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.parallel = parallel
        self.local_files_only = local_files_only
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.model_kwargs = model_kwargs


    def warm_up(self):
        pass
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embeds a list of Documents.

        :param documents: List of Documents to embed.
        :returns: A dictionary with the following keys:
            - `documents`: List of Documents with each Document's `sparse_embedding`
                            field set to the computed embeddings.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            msg = (
                "FastembedSparseDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a list of strings, please use the FastembedTextEmbedder."
            )
            raise TypeError(msg)
        if not hasattr(self, "embedding_backend"):
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        embeddings = self.embedding_backend.embed(
            texts_to_embed,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            parallel=self.parallel,
        )

        for doc, emb in zip(documents, embeddings):
            doc.sparse_embedding = emb
        return {"documents": documents}

@component
class OpenAIReranker():
    def __init__(
            self,
            top_k: int = 10,
            meta_fields_to_embed: Optional[List[str]] = None,
            embedding_separator: str = "\n",
            scale_score: bool = True,
            calibration_factor: Optional[float] = 1.0,
            score_threshold: Optional[float] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            batch_size: int = 16,
            base_url: str = "",
            api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
            model: str = "",
            timeout: int = 300,
            default_config_path=""
    ):
        """
        初始化OpenAIReranker

        Args:
            base_url: OpenAI API base URL
            api_key: OpenAI API key
            model: Model name for reranking
            max_length: 最大token长度
            **kwargs: 其他传递给父类的参数
        """
        self.base_url = base_url
        # 在初始化方法中解析 Secret
        self.api_key = api_key.resolve_value() if api_key else None
        self.model = model
        self.device = None
        self.top_k = top_k
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.scale_score = scale_score
        self.calibration_factor = calibration_factor
        self.score_threshold = score_threshold
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.batch_size = batch_size
        self.timeout=timeout
        self.default_config_path=default_config_path
        PROJECT_ROOT = find_project_root()
        os.chdir(PROJECT_ROOT)
        retrieval_config = read_json_file(self.default_config_path)
        self.rerank_tokenizer = RerankTokenizer(retrieval_config=retrieval_config)
        # 检查 API key 是否有效
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Please set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

    def _prepare_document_text(self, document: Document) -> str:
        """准备文档文本"""
        meta_values_to_embed = [
            str(document.meta[key]) for key in self.meta_fields_to_embed
            if key in document.meta and document.meta[key]
        ]
        return self.embedding_separator.join(meta_values_to_embed + [document.content or ""])

    def _truncate_query_documents(self, query: str, documents: List[Document],logging) -> tuple[str, List[str]]:
        """
        使用tokenize计算器对查询和文档进行整体截断

        Args:
            query: 原始查询
            documents: 文档列表

        Returns:
            tuple: (截断后的查询, 截断后的文档列表)
        """
        # Nas 不使用tokenize计算器
        # try:
        #     # 准备文档文本
        #     document_texts = [self._prepare_document_text(doc) for doc in documents]
        # except Exception as e:
        #     logging.error(f"tokenize计算准备文档文本失败: {str(e)}")
        #     raise RuntimeError(f"tokenize计算准备文档文本失败: {str(e)}")
        # return query, document_texts
        try:
            document_texts = [self._prepare_document_text(doc) for doc in documents]
            # 使用tokenize计算器进行精确的token级别截断
            result = self.rerank_tokenizer.tokenize(
                query=query,
                documents=document_texts,
                logging=logging
            )

            # 返回截断后的文本
            return result['truncated_texts']['query'], result['truncated_texts']['documents']

        except ImportError:
            logging.error("无法导入 tokenize 计算器，请确保 retriever.configs.tokenize 可用")
            raise ImportError("无法导入 tokenize 计算器，请确保 retriever.configs.tokenize 可用")
        except Exception as e:
            logging.error(f"tokenize计算器调用失败: {str(e)}")
            raise RuntimeError(f"tokenize计算器调用失败: {str(e)}")

    def warm_up(self):
        pass
    @component.output_types(documents=List[Document])
    def run(
            self,
            query: str,
            documents: List[Document],
            top_k: Optional[int] = None,
            scale_score: Optional[bool] = None,
            calibration_factor: Optional[float] = None,
            score_threshold: Optional[float] = None,
            logging= logging
    ):
        if not documents:
            return {"documents": []}
        # 过滤掉 content 为空的文档，并建立索引映射
        # 需要在过滤前保存原始索引，以便后续 rerank 结果能正确映射回原始文档
        # valid_doc_indices = []
        valid_documents = []
        for idx, doc in enumerate(documents):
            # 检查 content 是否为空或仅包含空白字符
            if doc.content and doc.content.strip():
                # valid_doc_indices.append(idx)
                valid_documents.append(doc)

        # 如果所有文档都被过滤掉了，返回空列表
        if not valid_documents:
            logging.warning("所有文档的 content 都为空，跳过 rerank")
            return {"documents": []}

        # 记录被过滤的空内容文档数量
        filtered_count = len(documents) - len(valid_documents)
        if filtered_count > 0:
            logging.info(f"过滤掉 {filtered_count} 个空内容文档，剩余 {len(valid_documents)} 个有效文档")

        # 使用过滤后的文档进行后续处理
        documents = valid_documents
        # original_to_valid_idx = valid_doc_indices  # 原始索引 -> 有效文档列表中的索引

        top_k = top_k or self.top_k
        scale_score = scale_score or self.scale_score
        calibration_factor = calibration_factor or self.calibration_factor
        score_threshold = score_threshold or self.score_threshold

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        if scale_score and calibration_factor is None:
            raise ValueError(
                f"scale_score is True so calibration_factor must be provided, but got {calibration_factor}"
            )
        with LoggedTime("rerank_tokenize计算器模块耗时统计"):
            # 使用tokenize计算器进行精确的截断
            truncated_query, truncated_docs = self._truncate_query_documents(query, documents,logging)

        # ============ 调试日志：记录发送给rerank API的数据 ============
        logging.info(f"========== Rerank API 调试信息 ==========")
        logging.info(f"Rerank URL: {self.base_url}/rerank")
        logging.info(f"Model: {self.model}")
        logging.info(f"Query (前100字符): {truncated_query[:100]}")
        logging.info(f"Documents数量: {len(truncated_docs)}")
        logging.info(f"Top_k: {top_k}")

        # 检查每个文档的内容
        for idx, doc in enumerate(truncated_docs):
            doc_preview = str(doc)[:200] if len(str(doc)) > 200 else str(doc)
            logging.info(f"Document[{idx}] 预览: {doc_preview}")

        # 检查原始文档的meta字段
        if documents:
            logging.info(f"原始Document[0]的meta字段: {list(documents[0].meta.keys()) if documents[0].meta else []}")
            # 检查是否包含图片相关字段
            if documents[0].meta:
                for key in documents[0].meta.keys():
                    if 'image' in key.lower() or 'img' in key.lower() or 'picture' in key.lower() or 'photo' in key.lower():
                        logging.warning(f"发现图片相关字段: {key} = {documents[0].meta[key]}")
        logging.info(f"==========================================")

        data = {
            "model": self.model,
            "query": truncated_query,
            "documents": truncated_docs,
            "top_n": top_k,
            "max_length": 512
        }

        # 记录请求体的结构（不记录完整内容，避免日志过大）
        logging.info(f"请求体结构: model={data['model']}, query长度={len(data['query'])}, documents数量={len(data['documents'])}, top_n={data['top_n']}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 重试机制配置
        max_retries = 3
        retry_delay = 2  # 秒
        response = None

        for retry_count in range(max_retries):
            try:
                with LoggedTime("reranker模块耗时统计"):
                    # Make the request to rerank endpoint
                    response = requests.post(
                        f"{self.base_url}/rerank",
                        headers=headers,
                        json=data,
                        timeout=self.timeout
                    )

                    if response.status_code != 200:
                        error_msg = f"Rerank API 请求失败! 状态码: {response.status_code}, 响应: {response.text}"

                        # 如果是最后一次重试,则抛出异常
                        if retry_count == max_retries - 1:
                            logging.error(f"{error_msg} (已重试{max_retries}次)")
                            logging.error(f"请求体: model={data['model']}, query长度={len(data['query'])}, documents数量={len(data['documents'])}")
                            raise RuntimeError(f"API request failed with status {response.status_code}: {response.text}")
                        else:
                            logging.warning(f"{error_msg} (第{retry_count + 1}次重试失败,准备进行第{retry_count + 2}次重试...)")
                            time.sleep(retry_delay * (retry_count + 1))  # 递增延迟
                            continue

                    # 成功获取响应,跳出重试循环
                    break

            except requests.exceptions.RequestException as e:
                error_msg = f"Rerank API 网络请求异常: {str(e)}"

                # 如果是最后一次重试,则抛出异常
                if retry_count == max_retries - 1:
                    logging.error(f"{error_msg} (已重试{max_retries}次)")
                    raise RuntimeError(f"Failed to call rerank API after {max_retries} retries: {str(e)}")
                else:
                    logging.warning(f"{error_msg} (第{retry_count + 1}次重试失败,准备进行第{retry_count + 2}次重试...)")
                    time.sleep(retry_delay * (retry_count + 1))  # 递增延迟
                    continue

        # 处理API响应
        result = response.json()

        # 处理API响应
        if "results" not in result:
            if "data" in result:
                results_data = result["data"]
            else:
                print(f"Debug - API response: {result}")
                raise RuntimeError(f"Unexpected API response format: {result}")
        else:
            results_data = result["results"]

        # 修正索引处理逻辑
        ranked_docs = []
        for item in results_data:
            # 根据FastAPI返回的格式调整
            if "indices" in item:
                original_index = item["indices"]
            elif "index" in item:
                original_index = item["index"]
            else:
                continue

            # 获取分数
            score = item.get('relevance_score', 0.0)
            # # 添加sigmoid激活
            # score = 1.0 / (1.0 + np.exp(-raw_score * 1.0))

            # 应用校准
            if scale_score and calibration_factor:
                score = score * calibration_factor

            # 确保索引在有效范围内
            if 0 <= original_index < len(documents):
                ranked_doc = Document(
                    content=documents[original_index].content,
                    meta=documents[original_index].meta.copy() if documents[original_index].meta else {},
                    score=float(score),
                    id=documents[original_index].id,
                    embedding=documents[original_index].embedding,
                    sparse_embedding=documents[original_index].sparse_embedding
                )
                ranked_docs.append(ranked_doc)

        # 按分数降序排序（确保顺序正确）
        ranked_docs.sort(key=lambda x: x.score, reverse=True)

        # 应用分数阈值
        if score_threshold is not None:
            ranked_docs = [doc for doc in ranked_docs if doc.score >= score_threshold]

        return {"documents": ranked_docs[:top_k]}

@component
class SentenceTransformersEmbedderClient:
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:11440",
        model: str = "thenlper/gte-base",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        truncate_dim: Optional[int] = None,
        precision: str = "float32"
    ):
        """
        Initialize the SentenceTransformersEmbedderClient.

        :param api_url: URL of the FastAPI service hosting the embedding model
        :param model: Name of the Sentence Transformers model to use
        :param prefix: Optional prefix to add to each text before embedding
        :param suffix: Optional suffix to add to each text before embedding
        :param batch_size: Number of documents to embed at once
        :param progress_bar: Whether to show a progress bar while embedding
        :param normalize_embeddings: Whether to normalize the embeddings
        :param meta_fields_to_embed: List of metadata fields to embed along with the document content
        :param embedding_separator: Separator used to join metadata fields and document content
        :param truncate_dim: Optional dimension to truncate embeddings to
        :param precision: Precision of the embeddings (e.g., "float32")
        """
        self.api_url = api_url
        self.embed_endpoint = f"{api_url}/embed_documents"
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.truncate_dim = truncate_dim
        self.precision = precision

    def _document_to_dict(self, doc: Document) -> Dict:
        """将 Haystack Document 对象转换为字典"""
        doc_dict = {
            "content": doc.content,
            "meta": doc.meta,
            "id": doc.id,
            "embedding": doc.embedding if doc.embedding is not None else None
        }
        # 添加 score 字段（如果存在）
        if hasattr(doc, "score"):
            doc_dict["score"] = doc.score
        return doc_dict

    def warm_up(self):
        pass

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embed a list of documents using the FastAPI service.

        :param documents: List of Documents to embed
        :return: Dictionary with key "documents" containing the embedded Documents
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "SentenceTransformersEmbedderClient expects a list of Documents as input."
            )

        # Prepare request data
        request_data = {
            "documents": [self._document_to_dict(doc) for doc in documents],
            "model": self.model,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "batch_size": self.batch_size,
            "progress_bar": self.progress_bar,
            "normalize_embeddings": self.normalize_embeddings,
            "meta_fields_to_embed": self.meta_fields_to_embed,
            "embedding_separator": self.embedding_separator,
            "truncate_dim": self.truncate_dim,
            "precision": self.precision
        }
        with LoggedTime("向量模型调用api耗时统计"):
            # Send request
            response = requests.post(self.embed_endpoint, json=request_data)

        if response.status_code != 200:
            raise Exception(f"Embedding request failed with status {response.status_code}: {response.text}")

        # Process response
        result = response.json()
        embedded_docs = []
        # 过滤掉 Haystack Document 不支持的字段
        VALID_FIELDS = {
            'content', 'meta', 'id', 'embedding',
            'score', 'blob', 'mime_type', 'sparse_embedding'
        }
        for idx, doc_data in enumerate(result["documents"]):
            # 使用字典推导式过滤数据
            filtered_data = {
                key: value
                for key, value in doc_data.items()
                if key in VALID_FIELDS
            }
            # embedded_docs.append(Document(**filtered_data))
            # 从原始文档中获取 sparse_embedding 并赋值
            original_doc = documents[idx]
            if hasattr(original_doc, 'sparse_embedding') and original_doc.sparse_embedding is not None:
                filtered_data['sparse_embedding'] = original_doc.sparse_embedding
            elif hasattr(original_doc, 'meta') and original_doc.meta and 'sparse_embedding' in original_doc.meta:
                filtered_data['sparse_embedding'] = original_doc.meta.get('sparse_embedding')

            embedded_docs.append(Document(**filtered_data))

        return {"documents": embedded_docs}

@component
class SentenceTransformersTextEmbedderClient:
    """
    Embeds strings using a remote Sentence Transformers embedding service.

    You can use it to embed user query and send it to an embedding retriever.

    Usage example:
    ```python
    from haystack.components.embedders import SentenceTransformersTextEmbedderClient

    text_to_embed = "I love pizza!"

    text_embedder = SentenceTransformersTextEmbedderClient(api_url="http://localhost:11440")
    text_embedder.warm_up()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [-0.07804739475250244, 0.1498992145061493,, ...]}
    ```
    """

    def __init__(self, api_url: str = "http://127.0.0.1:11440"):
        """
        Initialize the SentenceTransformersTextEmbedderClient.

        :param api_url: The URL of the remote embedding service.
        """
        self.api_url = api_url
        self.embed_endpoint = f"{api_url}/embed_text"


    def warm_up(self):
        """
        No warmup needed for the client as it connects to a remote service.
        """
        pass

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """
        Embed the given text using the remote embedding service.

        :param text: The text to embed.
        :param kwargs: Additional parameters to pass to the embedding service.
        :return: A dictionary containing the embedding.
        :raises RuntimeError: If the API call fails.
        :raises TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(
                "SentenceTransformersTextEmbedderClient expects a string as input. "
                "In case you want to embed a list of Documents, please use the appropriate document embedder."
            )

        payload = {"text": text}

        try:
            response = requests.post(self.embed_endpoint, json=payload)
            response.raise_for_status()
            return {"embedding": response.json()["embedding"]}
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to call embedding API: {str(e)}")