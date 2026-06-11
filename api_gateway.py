import asyncio
import base64
import time
import os
import json
from typing import List, Dict, Any
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.staticfiles import StaticFiles as StarletteStaticFiles
from pydantic import BaseModel
import argparse
from dotenv import load_dotenv
from logger_local import app_logger as logger

try:
    from vision_core import VisionProcessor
except ImportError:
    VisionProcessor = None

def parse_args():
    parser = argparse.ArgumentParser(description='RAG Gateway API - 统一检索网关')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    return parser.parse_args()

def load_environment_variables():
    """加载环境变量"""
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded .env from: {env_path}")
        return True
    logger.warning("Warning: .env file not found")
    return False

# 服务配置
RAG_URL = os.getenv("RAG_URL", "http://127.0.0.1:17101")  # 远程 RAG 服务
WIKI_URL = os.getenv("WIKI_URL", "http://127.0.0.1:17101")  # 远程 Wiki 增强检索服务（同一地址）
DEFAULT_TOP_K = 30  # 兼容旧请求字段；生成上下文不再按该值截断
INTERNAL_HTTP_CLIENT_KWARGS = {"trust_env": False}

# LLM配置（DashScope 商业 API）
LLM_API_URL = os.getenv("LLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if not LLM_API_KEY:
    logger.warning("DASHSCOPE_API_KEY 未设置，LLM 生成将不可用")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-max")
LLM_TIMEOUT = 300

# LLM 生成模板（6-rule 详细 RAG 模板）
RAG_GENERATION_TEMPLATE = """你是一个有用的 RAG 生成助手，回答问题时必须严格遵循以下规则：

1. **基于参考依据推理**
   - 所有答案必须严格基于提供的参考依据。
   - 在生成答案前，先判断问题所需信息是否能从参考依据中找到，并先在心里完成证据筛选。
   - 如果答案需要综合多个段落、多个来源或多个信息点，请仅使用来源中可确认的内容进行归纳和推理。
   - 可以基于参考依据进行必要的分析、归纳和整理，但分析内容必须来自参考依据本身。
   - 禁止生成参考依据中未出现的内容、推测或假设。

2. **证据筛选与冲突处理**
   - 先识别问题的核心意图、实体、工具名、版本号、平台、动作对象、指标或条件。
   - 对问题中的最小对象要保持敏感：同一缩写、同一关键词或同一主题可能对应不同工具、规范、平台、版本、流程或实现位置，必须先确认对象一致再使用证据。
   - 将参考依据区分为四类，但不要在最终答案中展示分类过程：
     1. 直接相关证据：明确回答当前 query 的核心对象和核心问题，必须优先使用。
     2. 一致的补充证据：与直接相关证据不冲突，能补充背景、步骤、原因、限制或注意事项，可以适度使用。
     3. 主题相关但对象不同的内容：领域、关键词或缩写相似，但工具、版本、平台、对象、问题意图不同，不能用于回答。
     4. 冲突或可疑证据：与直接相关证据、参考答案方向或问题对象冲突时，不能用于生成结论。
   - 对缩写、工具名、版本号、平台名和接口名要特别严格。相同缩写可能代表不同概念，不能因为主题相关就混用。
   - 多个来源出现冲突时，以最直接回答 query 且对象一致的证据为准；不要把冲突来源里的细节混入答案。
   - 如果直接相关证据中包含命令、路径、工具名、版本号、寄存器、bit 位、协议名、函数名、表名、字段名、步骤顺序、条件分支、阈值或适用场景，应尽量保留这些可验证细节；不要用泛泛解释替代关键操作信息。
   - 如果某段内容看起来专业但无法回答当前 query 的核心对象或关键动作，只能作为背景理解，不能作为答案主干。
   - 如果只能找到主题相关但对象不同的内容，应说明参考依据中未提供足够信息，而不是借相近内容扩写。

3. **回答要求**
   - 回答要清晰、完整、自然，不要过度简洁，不要只输出单个短语、单个数值或一句过于压缩的结论。
   - 在不脱离参考依据的前提下，应适当展开答案，让用户能够直接理解结论、相关内容和必要细节。
   - 如果参考依据中包含多个类别、多个条件、多个步骤、多个原因、多个结果或多个相关信息点，应尽量完整覆盖，并按逻辑顺序组织。
   - 如果问题较简单，可以简洁回答；如果问题涉及对比、流程、分类、原因、配置、方法、规范或多个信息点，应适当展开说明。
   - 回答应保持智能、亲切、易读，避免机械罗列；但不能加入参考依据之外的背景知识。
   - 回答必须保留参考依据中的关键术语、单位、数值和量词（如 "g/L"、"mg/m³"）。
   - 对可执行或可复核的技术回答，应优先保留必要的原始名称和精确信息，例如命令、路径、版本、寄存器、bit、函数、表、协议、工具、状态名、进入/退出条件和关键顺序。
   - 回答中应包含必要的描述性关键词，使答案具备可读性和可验证性，例如"方式"、"要求"、"原因"、"步骤"、"配置"、"结果"、"限值"、"标准"等。
   - 即使问题中出现未在参考依据中提及的词汇，只要核心概念存在，也应回答参考依据中相关的信息。
   - 允许补充直接相关的背景和注意事项，但补充内容必须能被一致的补充证据支持，不能把旁支文档当作主答案。

4. **答案格式**
   - 最终只输出"答案："后的正文内容，不要再次输出"答案："标题。
   - 不要输出"支持来源""参考来源""引用来源""来源信息"等内容。
   - 不要在答案中出现任何来源编号、文件编号、引用标记或文件名。
   - 不要显式说明信息来自哪一段、哪一个文件或哪一个参考来源。
   - 优先使用参考依据中的原始表述、术语和数值。
   - 可以在不引入外部信息的前提下，对参考依据内容进行归纳、整理、合并和适度改写。
   - 如果答案包含多个信息点，应使用分段、分点或有序编号组织，提升可读性。
   - 如果答案只涉及一个明确结论，也应尽量给出完整句子，而不是过度简短的片段。

5. **回答策略**
   - 重点匹配问题中的核心概念和指标，忽略参考依据中的无关内容。
   - 回答必须基于参考依据中的内容，但不要在最终答案中展示来源标签。
   - 当参考依据信息不足以完整回答问题时，应明确说明"参考依据中未提供足够信息"，不要补充外部知识。
   - 当参考依据能够部分回答问题时，应先回答可确认的部分，再说明缺失的信息。
   - 禁止生成参考依据中未出现的内容、推测或假设。

6. **输出限制**
   - 最终回答中禁止出现以下形式的内容：
     - "[参考来源x]"
     - "[参考文件x]"
     - "参考来源x"
     - "参考文件x"
     - "文件名：xxx"
     - "来源：xxx"
     - "支持来源：xxx"
     - "引用来源：xxx"
     - "根据参考来源"
     - "根据参考文件"
     - "根据参考依据"
   - 最终回答应像直接回答用户问题一样自然表达，而不是展示检索来源。
   - 可以进行必要的分析、归纳和整理，但分析过程不能暴露来源编号、文件编号、引用标记或文件名。

参考依据：
{context}

问题：
{query}

答案："""

load_environment_variables()
args = parse_args()
app = FastAPI(title="RAG Gateway - 统一检索网关")

# 自定义无缓存静态文件
class NoCacheStaticFiles(StarletteStaticFiles):
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

# 挂载静态文件（强制无缓存）
app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic模型
class QueryRequest(BaseModel):
    query: str
    parseType: str = "firmware"
    knowledgeUUID: str = ""
    image_base64: str = ""
    image_name: str = ""
    top_k: int = DEFAULT_TOP_K  # 兼容字段，当前不再限制参与生成的检索结果数量
    use_llm: bool = True  # 是否使用LLM生成答案
    retrieval_mode: str = "rag"  # 检索模式: "rag"(纯RAG), "wiki"(RAG+Wiki增强)
    wiki_min_relevance: float = 2.0  # Wiki 关联度阈值


class QueryResponse(BaseModel):
    answer: str = ""
    sources: List[Dict[str, Any]]
    rag_results: List[Dict[str, Any]] = []
    wiki_results: List[Dict[str, Any]] = []
    time_cost: Dict[str, float] = {}
    image_description: str = ""
    combined_query: str = ""
    original_query: str = ""
    has_image: bool = False


# Vision Processor 单例
_gateway_vision_processor = None

def get_gateway_vision_processor():
    global _gateway_vision_processor
    if _gateway_vision_processor is None and VisionProcessor is not None:
        _gateway_vision_processor = VisionProcessor(
            api_key=os.getenv("MINICPM_API_KEY"),
            api_base_url=os.getenv("MINICPM_API_URL", "http://localhost:8008/v1"),
        )
    return _gateway_vision_processor


def _clone_request_with_query(request: QueryRequest, query: str) -> QueryRequest:
    if hasattr(request, "model_copy"):
        return request.model_copy(update={"query": query})
    return request.copy(update={"query": query})


async def prepare_multimodal_request(request: QueryRequest) -> tuple[QueryRequest, Dict[str, str]]:
    image_context = {
        "image_description": "",
        "combined_query": "",
        "original_query": request.query,
        "has_image": False,
    }

    image_base64 = (request.image_base64 or "").strip()
    if not image_base64:
        return request, image_context

    try:
        if image_base64.startswith("data:") and "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        image_data = base64.b64decode(image_base64)
        processor = get_gateway_vision_processor()
        if processor is None:
            logger.warning("VisionProcessor 未初始化，跳过图片处理")
            return request, image_context

        image_description = processor.describe_image_with_context(
            image_data,
            request.query,
        )

        if image_description:
            combined_query = processor.combine_query_with_description(
                user_query=request.query,
                image_description=image_description,
            )
            image_context.update(
                {
                    "image_description": image_description,
                    "combined_query": combined_query,
                    "has_image": True,
                }
            )
            return _clone_request_with_query(request, combined_query), image_context

        logger.warning("图片描述获取失败，使用原始查询")
    except Exception as e:
        logger.error(f"处理图片失败: {e}")

    return request, image_context


async def fetch_rag_results(query: str, parse_type: str = "firmware", knowledge_uuid: str = "") -> tuple[list[dict], dict[str, float]]:
    """从RAG服务获取检索结果，返回 (documents, time_cost)"""
    try:
        async with httpx.AsyncClient(timeout=30.0, **INTERNAL_HTTP_CLIENT_KWARGS) as client:
            payload = {
                "query": query,
                "parseType": parse_type
            }
            if knowledge_uuid:
                payload["knowledgeUUID"] = knowledge_uuid

            response = await client.post(
                f"{RAG_URL}/search",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") == 0:
                resp_data = data.get("data", {})
                if isinstance(resp_data, dict):
                    documents = resp_data.get("documents", [])
                    time_cost = resp_data.get("time_cost", {})
                else:
                    documents = resp_data if isinstance(resp_data, list) else []
                    time_cost = {}
                return documents, time_cost
            return [], {}
    except Exception as e:
        logger.error(f"RAG检索失败: {e}")
        return [], {}


async def fetch_wiki_results(query: str, min_relevance: float = 2.0) -> tuple[list[dict], dict[str, float]]:
    """从Wiki服务获取检索结果（使用Wiki增强接口），返回 (documents, time_cost)"""
    try:
        async with httpx.AsyncClient(timeout=120.0, **INTERNAL_HTTP_CLIENT_KWARGS) as client:
            payload = {
                "query": query,
                "parseType": "firmware",
                "wiki_expansion_min_relevance": min_relevance,
            }

            response = await client.post(
                f"{WIKI_URL}/search_with_wiki",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") == 0:
                resp_data = data.get("data", {})
                documents = resp_data.get("documents", []) if isinstance(resp_data, dict) else []
                time_cost = resp_data.get("time_cost", {}) if isinstance(resp_data, dict) else {}
                logger.info(f"Wiki检索成功，返回 {len(documents)} 条结果")
                return documents, time_cost
            else:
                logger.warning(f"Wiki检索返回错误: {data.get('msg', 'Unknown error')}")
                return [], {}
    except Exception as e:
        logger.error(f"Wiki检索失败: {e}")
        return [], {}


async def fetch_retrieval_results(request: QueryRequest) -> tuple[list[dict], list[dict], dict[str, float]]:
    """根据检索模式并发获取RAG和/或Wiki结果，返回 (rag_results, wiki_results, backend_time_costs)"""
    rag_results: list[dict] = []
    wiki_results: list[dict] = []
    backend_time_costs: dict[str, float] = {}
    retrieval_mode = request.retrieval_mode

    if retrieval_mode not in {"rag", "wiki", "all"}:
        logger.warning(f"未知检索模式: {retrieval_mode}，按 rag 处理")
        retrieval_mode = "rag"

    tasks = []
    task_names = []
    if retrieval_mode == "rag":
        tasks.append(fetch_rag_results(request.query, request.parseType, request.knowledgeUUID))
        task_names.append("rag")
    elif retrieval_mode == "wiki":
        tasks.append(fetch_wiki_results(request.query, request.wiki_min_relevance))
        task_names.append("wiki")
    elif retrieval_mode == "all":
        # /search_with_wiki 内部已包含 RAG + Wiki + RRF + rerank，无需再调 /search
        tasks.append(fetch_wiki_results(request.query, request.wiki_min_relevance))
        task_names.append("wiki")

    if not tasks:
        return rag_results, wiki_results, backend_time_costs

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for task_name, result in zip(task_names, results):
        if isinstance(result, Exception):
            logger.error(f"{task_name.upper()}检索异常: {result}")
            continue
        docs, tc = result
        if task_name == "rag":
            rag_results = docs
        elif task_name == "wiki":
            wiki_results = docs
        if isinstance(tc, dict):
            backend_time_costs.update(tc)

    return rag_results, wiki_results, backend_time_costs


def merge_and_rank_results(rag_results: list[dict], wiki_results: list[dict], top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """合并和排序检索结果 - RAG在前，Wiki结果拼接在后。
    注意：top_k 仅为兼容字段，不在此处截断，由后端 reranker 控制返回数量。"""
    import json
    all_results = []

    # 处理RAG结果
    for item in rag_results:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception:
                logger.warning(f"跳过非预期的RAG结果类型: {type(item)}")
                continue
        if not isinstance(item, dict):
            logger.warning(f"跳过非预期的RAG结果类型: {type(item)}")
            continue

        metadata = item.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        all_results.append({
            "content": item.get("content", ""),
            "source": metadata.get("file_name", metadata.get("knowledge_uuid", "unknown")),
            "score": item.get("score", 0.0),
            "metadata": metadata,
            "origin": "rag"
        })

    # 处理Wiki结果
    for item in wiki_results:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception:
                logger.warning(f"跳过非预期的Wiki结果类型: {type(item)}")
                continue
        if not isinstance(item, dict):
            logger.warning(f"跳过非预期的Wiki结果类型: {type(item)}")
            continue

        metadata = item.get("metadata", {})
        source_type = metadata.get("source", "")

        if source_type == "llm_wiki_graph":
            wiki_title = metadata.get("wiki_node_title", "Wiki节点")
            source = f"Wiki-{wiki_title}"
            score = item.get("score", 0.5)
            origin = "wiki"
        else:
            file_name = metadata.get("file_name", "unknown")
            source = f"RAG-{file_name}"
            score = item.get("score", 0.0)
            origin = "rag"

        all_results.append({
            "content": item.get("content", ""),
            "source": source,
            "score": score if score is not None else 0.0,
            "metadata": metadata,
            "origin": origin
        })

    return all_results


async def call_llm_generation(query: str, documents: List[Dict], image_base64: str = "") -> str:
    """调用LLM生成答案（非流式），支持多模态图片输入。"""
    try:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            context_parts.append(f"文档{i}：\n{content}\n")

        context = "\n".join(context_parts)
        prompt = RAG_GENERATION_TEMPLATE.format(
            context=context,
            query=query
        )

        # 构造 messages，支持多模态
        messages_payload = {"role": "user", "content": prompt}
        image_b64_clean = (image_base64 or "").strip()
        if image_b64_clean:
            if not image_b64_clean.startswith("data:"):
                image_b64_clean = f"data:image/jpeg;base64,{image_b64_clean}"
            messages_payload = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_b64_clean}},
                ],
            }

        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(
                f"{LLM_API_URL}/chat/completions",
                headers=headers,
                json={
                    "model": LLM_MODEL,
                    "messages": [messages_payload],
                    "temperature": 0
                }
            )
            response.raise_for_status()
            result = response.json()

            if result.get("choices") and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"]
                if answer.startswith("答案："):
                    answer = answer[3:].strip()
                if answer.startswith("答案:"):
                    answer = answer[3:].strip()
                return answer
            return "生成答案失败"

    except Exception as e:
        logger.error(f"LLM生成失败: {e}")
        return f"生成答案时发生错误: {str(e)}"


async def call_llm_generation_stream(query: str, documents: List[Dict], image_base64: str = ""):
    """调用LLM生成答案（流式），支持多模态图片输入。"""
    try:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            context_parts.append(f"文档{i}：\n{content}\n")

        context = "\n".join(context_parts)
        prompt = RAG_GENERATION_TEMPLATE.format(
            context=context,
            query=query
        )

        # 构造 messages，支持多模态
        messages_payload = {"role": "user", "content": prompt}
        image_b64_clean = (image_base64 or "").strip()
        if image_b64_clean:
            # 补回 data URL 前缀（前端已提取纯 base64）
            if not image_b64_clean.startswith("data:"):
                image_b64_clean = f"data:image/jpeg;base64,{image_b64_clean}"
            messages_payload = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_b64_clean}},
                ],
            }

        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{LLM_API_URL}/chat/completions",
                headers=headers,
                json={
                    "model": LLM_MODEL,
                    "messages": [messages_payload],
                    "temperature": 0,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()
                full_answer = ""
                prefix_removed = False

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]

                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if data.get("choices") and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")

                                if content:
                                    if not prefix_removed:
                                        if full_answer == "" and content.startswith("答案："):
                                            content = content[3:]
                                        elif full_answer == "" and content.startswith("答案:"):
                                            content = content[3:]
                                        prefix_removed = True

                                    full_answer += content
                                    yield content

                        except json.JSONDecodeError:
                            continue

    except Exception as e:
        logger.error(f"LLM流式生成失败: {e}")
        yield f"\n\n[错误: 生成答案时发生错误 - {str(e)}]"


@app.get("/", response_class=HTMLResponse)
async def chat_interface():
    """聊天界面"""
    try:
        static_html_path = Path(__file__).parent / "static" / "index.html"
        with open(static_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(
            content=html_content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        logger.error(f"读取前端页面失败: {e}")
        return HTMLResponse(
            content="<html><body><h1>前端页面加载失败</h1></body></html>",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "message": "聚合网关正常运行",
        "services": {
            "rag": RAG_URL,
            "wiki": WIKI_URL,
            "llm": LLM_API_URL
        }
    }


@app.get("/wiki_graph")
async def proxy_wiki_graph():
    """代理后端 /wiki_graph 接口"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{WIKI_URL}/wiki_graph")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Wiki图谱代理失败: {e}")
        return {"code": 1, "msg": str(e)}


@app.post("/wiki_node_related")
async def proxy_wiki_node_related(request: Request):
    """代理后端 /wiki_node_related 接口"""
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{WIKI_URL}/wiki_node_related", json=body)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Wiki节点相关查询代理失败: {e}")
        return {"code": 1, "msg": str(e)}


@app.post("/query", response_model=QueryResponse)
async def unified_query(request: QueryRequest):
    """
    统一查询接口

    根据retrieval_mode选择检索方式：
    - "rag": 仅使用本地RAG服务
    - "wiki": 仅使用Wiki增强服务
    - "all": 同时使用两个服务
    """
    start_time = time.time()
    time_costs = {}

    request, image_context = await prepare_multimodal_request(request)

    # 第一步：并发检索
    retrieval_start = time.time()
    rag_results, wiki_results, backend_time_costs = await fetch_retrieval_results(request)
    time_costs["retrieval"] = time.time() - retrieval_start
    time_costs.update(backend_time_costs)

    # 第二步：合并结果
    merge_start = time.time()
    all_sources = merge_and_rank_results(rag_results, wiki_results, request.top_k)
    time_costs["merge"] = time.time() - merge_start

    logger.info(f"检索完成 - 模式:{request.retrieval_mode}, RAG: {len(rag_results)}条, Wiki: {len(wiki_results)}条, 合并后: {len(all_sources)}条")

    # 第三步：调用LLM生成答案
    answer = ""
    if request.use_llm and all_sources:
        llm_start = time.time()
        try:
            answer = await call_llm_generation(request.query, all_sources, request.image_base64)
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            answer = f"生成答案时发生错误: {str(e)}"
        time_costs["llm_generation"] = time.time() - llm_start
    else:
        answer = "未启用LLM生成或无检索结果"

    time_costs["total"] = time.time() - start_time

    logger.info(f"查询完成 - 总耗时: {time_costs['total']:.2f}s, 检索: {time_costs['retrieval']:.2f}s, LLM: {time_costs.get('llm_generation', 0):.2f}s")

    return QueryResponse(
        answer=answer,
        sources=all_sources,
        rag_results=rag_results,
        wiki_results=wiki_results,
        time_cost=time_costs,
        image_description=image_context["image_description"],
        combined_query=image_context["combined_query"],
        original_query=image_context["original_query"],
        has_image=image_context["has_image"],
    )


@app.post("/search")
async def search_only(request: QueryRequest):
    """
    仅检索接口，不调用LLM

    返回合并后的检索结果
    """
    start_time = time.time()

    request, image_context = await prepare_multimodal_request(request)

    rag_results, wiki_results, backend_time_costs = await fetch_retrieval_results(request)
    all_sources = merge_and_rank_results(rag_results, wiki_results, request.top_k)

    return {
        "code": 0,
        "data": {
            "documents": all_sources,
            "time_cost": backend_time_costs,
        },
        "time_cost": time.time() - start_time,
        "rag_count": len(rag_results),
        "wiki_count": len(wiki_results),
        "image_description": image_context["image_description"],
        "combined_query": image_context["combined_query"],
        "original_query": image_context["original_query"],
        "has_image": image_context["has_image"],
    }


@app.post("/query_stream")
async def unified_query_stream(request: QueryRequest):
    """
    统一查询接口（流式输出）

    根据retrieval_mode选择检索方式：
    - "rag": 仅使用本地RAG服务
    - "wiki": 仅使用Wiki增强服务
    - "all": 同时使用两个服务
    """
    start_time = time.time()

    request, image_context = await prepare_multimodal_request(request)

    # 第一步：并发检索
    retrieval_start = time.time()
    rag_results, wiki_results, backend_time_costs = await fetch_retrieval_results(request)
    retrieval_time = time.time() - retrieval_start
    logger.info(f"检索完成 - 模式:{request.retrieval_mode}, RAG: {len(rag_results)}条, Wiki: {len(wiki_results)}条, 耗时: {retrieval_time:.2f}s")

    # 第二步：合并结果
    merge_start = time.time()
    all_sources = merge_and_rank_results(rag_results, wiki_results, request.top_k)
    merge_time = time.time() - merge_start

    # 第三步：流式返回LLM生成结果
    if request.use_llm and all_sources:
        async def stream_generator():
            llm_time = 0.0
            try:
                sources_data = json.dumps({
                    "type": "sources",
                    "data": all_sources,
                    "rag_count": len(rag_results),
                    "wiki_count": len(wiki_results),
                    "image_context": image_context,
                    "time_cost": {
                        "retrieval": retrieval_time,
                        "merge": merge_time,
                        **backend_time_costs
                    }
                }, ensure_ascii=False)
                yield f"data: {sources_data}\n\n"

                llm_start = time.time()
                async for chunk in call_llm_generation_stream(request.query, all_sources, request.image_base64):
                    chunk_data = json.dumps({
                        "type": "content",
                        "content": chunk
                    }, ensure_ascii=False)
                    yield f"data: {chunk_data}\n\n"
                llm_time = time.time() - llm_start

                end_data = json.dumps({
                    "type": "done",
                    "time_cost": {
                        "retrieval": retrieval_time,
                        "merge": merge_time,
                        "llm_generation": llm_time,
                        **backend_time_costs
                    },
                    "total_time": time.time() - start_time
                }, ensure_ascii=False)
                yield f"data: {end_data}\n\n"

            except Exception as e:
                error_data = json.dumps({
                    "type": "error",
                    "error": str(e)
                }, ensure_ascii=False)
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        async def error_generator():
            error_data = json.dumps({
                "type": "error",
                "error": "未启用LLM生成或无检索结果"
            }, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream"
        )


if __name__ == "__main__":
    import uvicorn

    logger.info(f"启动聚合网关服务，监听 {args.host}:{args.port}")
    logger.info(f"RAG服务地址: {RAG_URL}")
    logger.info(f"Wiki服务地址: {WIKI_URL}")
    logger.info(f"LLM服务地址: {LLM_API_URL}")

    uvicorn.run(
        "api_gateway:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
        reload=False
    )
