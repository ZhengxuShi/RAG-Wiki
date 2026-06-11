from typing import Any, Dict, List
from haystack import Document, Pipeline, component
from hypster import HP, config

@config
def cfg_llm_module(hp: HP):
    """
    在这里将prompt builder和generator封装成一个pipeline
    """
    cfg_llm = hp.nest("llm/config/cfg_llm.py")
    cfg_prompt = hp.nest("llm/config/cfg_prompt.py")
    from haystack import Pipeline
    llm_module_pipeline = Pipeline()

    llm_module_pipeline.add_component(name="llm", instance=cfg_llm['llm'])
    llm_module_pipeline.add_component(name="prompt_builder", instance=cfg_prompt['prompt_builder'])

    llm_module_pipeline.connect(sender="prompt_builder", receiver="llm")


    

@component
class LlmModule:
    """
    大语言模型模块，封装提示词及模型选型等组件，提供对外接口

    输入：查询问题
    输出：检索到的文档列表
    """
    def __init__(self):
        self.cfg_llm_module = cfg_llm_module()
        self.llm_pipeline = self.cfg_llm_module['llm_module_pipeline']

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self,
        query: str,
        documents: List[Document]
    ):
        result = self.llm_pipeline.run({"prompt_builder": {"query": query, "documents": documents}})
        return {"replies": result["llm"]["replies"], "meta": result["llm"]["meta"]}
