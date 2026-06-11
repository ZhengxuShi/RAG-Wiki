"""
@File        : generator.py
@Date        : 2025-03-03
"""

import os
from datetime import datetime

from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.utils import Secret

from llm.src.utils.file_handler import load_config, save_to_csv
from llm.src.utils.logger import setup_logger


class Generator:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(self.current_dir, "..", "config", "llm_config.json")
        self.config = load_config(config_path)

        # 选择模型
        self.port = self.config.get("PORT")
        self.prompt_template = self.config.get("PROMPT_TEMPLATE")
        self.prompt_builder = PromptBuilder(template=self.prompt_template)

        # 初始化大模型
        self.generator ,self.model_config = self._initialize_generator()

        # 构建流水线
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.generator)
        self.pipeline.connect("prompt_builder", "llm")

    def _initialize_generator(self):
        """ 初始化 OpenAI 或 Ollama 生成器 """
        if self.port.lower() == "ollama":
            ollama_config = self.config["Ollama"]
            if ollama_config["MODEL_NAME"]=="qwen2":
                qwen2_config = ollama_config['qwen2']
                return OllamaGenerator(
                    model=qwen2_config.get("MODEL"),
                    url=qwen2_config.get("URL"),
                    generation_kwargs={
                        "num_predict": qwen2_config.get("NUM_PREDICT"),
                        "temperature": None
                    }),qwen2_config
            else:
                raise ValueError("不支持的模型类型")
        elif self.port.lower() == "openai":
            openai_config = self.config["OPENAI"]
            if openai_config["MODEL_NAME"]=="gpt":
                gpt_config = openai_config['gpt']
                return OpenAIGenerator(model=gpt_config.get("MODEL"), api_base_url=gpt_config.get("BASE_URL"),
                                       api_key=Secret.from_token(gpt_config.get("API_KEY")),
                                       generation_kwargs={"temperature": None}),gpt_config

            elif openai_config["MODEL_NAME"]=="qwen2.5":
                qwen2_5_config = openai_config['qwen2.5']
                return OpenAIGenerator(model=qwen2_5_config.get("MODEL"), api_base_url=qwen2_5_config.get("BASE_URL"),
                                       api_key=Secret.from_token(openai_config['qwen2.5'].get("API_KEY"), ),
                                       timeout=qwen2_5_config.get("TIMEOUT")),qwen2_5_config

            else:
                raise ValueError("不支持的模型类型")
        else:
            raise ValueError("不支持的接口类型")

    def generate(self, query, retrieved_text=None):
        """ 运行生成逻辑 """
        try:
            # model_config = self.config.get("GPT" if self.model_name.lower() == "gpt" else "OLLAMA", {})
            model_identifier = self.model_config.get("MODEL")
            temperature = self.model_config.get("RAG_TEMPERATURE") if retrieved_text else self.model_config.get(
                "GENERATION_ONLY_TEMPERATURE")
            rag_status = "RAG" if retrieved_text else "GEN"
            model_identifier = model_identifier.replace(":", "_")

            log_filename = os.path.join(self.current_dir, "..", "logs",
                                        f"generator_{model_identifier}_temp{temperature}_{rag_status}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log")

            self.logger = setup_logger(log_filename)

            original_question = query

            if retrieved_text:
                query = f"{retrieved_text}\n{query}"

            self.logger.info(
                f"模型: {model_identifier}, 温度: {temperature}, RAG模式: {'是' if retrieved_text else '否'}, 输入: {query}")
            results = self.pipeline.run({"prompt_builder": {"question": query}})
            answer = results["llm"].get("replies", ["No answer generated"])[0]

            # output_path = os.path.join(self.current_dir, "..", "outputs", "results.csv")
            # os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # save_to_csv(output_path, [[original_question, answer, retrieved_text]], ["问题", "生成答案", "检索文本"])

            self.logger.info(f"生成结果: {answer}")
            return answer
        except Exception as e:
            self.logger.error(f"生成失败: {str(e)}")
            return "Error in generation process"
