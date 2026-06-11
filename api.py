import time
from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import uuid
import shutil
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
# from data_loader.run_data_loader import run_data_loader_single_file
from retriever.retriever import RetrieverModule
from logger_local import app_logger as logger
import sys
import argparse
from dotenv import load_dotenv
# 配置loguru日志记录器
def parse_args():
    parser = argparse.ArgumentParser(description='RAG Retrieval API')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=17101, help='Port number')
    return parser.parse_args()

def load_environment_variables():
    """加载环境变量，支持打包后的情况"""
    # 尝试从多个可能的位置加载 .env 文件
    possible_env_paths = [
        Path('.env'),  # 当前目录
        Path(sys.argv[0]).parent / '.env',  # 可执行文件所在目录
        Path(__file__).parent / '.env',  # 脚本所在目录
    ]

    for env_path in possible_env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded .env from: {env_path}")
            return True

    logger.error("Warning: .env file not found")
    return False

load_environment_variables()
args = parse_args()
app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 初始化检索模块
RM = RetrieverModule()
RM.init_system(logging= logger)

def get_unique_path(base_dir: str, extension: str = "") -> str:
    """生成唯一的文件路径"""
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{timestamp}_{unique_id}{extension}")

# @app.get("/search")
# async def search_endpoint(query: str, knowledgeUUID: str = None):
#     """搜索"""
#     try:
#         logger.info(f"开始搜索请求，查询: {query}")
#         # RM.init_system()
#         start_time = time.time()
#         result = RM.run(query=query,
#                         retrieval_config=None, logging=logger, knowledgeUUID=knowledgeUUID)
#         # 记录结束时间
#         end_time = time.time()
#         # 计算并打印时间消耗
#         time_consumed = end_time - start_time
#         logger.info(f"总检索方法执行时间: {time_consumed:.4f} 秒")
#         logger.info(f"搜索完成，找到 {len(result)} 条结果")
#         return {'code': 0, 'data': result}
#     except Exception as e:
#         logger.info(f"搜索失败: {str(e)}", exc_info=True)
#         return {'code': 1, 'msg': str(e)}
#

@app.post("/search")
async def search_endpoint(request: Request):
    """搜索"""
    try:
        # 从请求体中获取JSON数据
        body = await request.json()
        query = body.get("query")
        knowledgeUUID = body.get("knowledgeUUID")
        parseType = body.get("parseType")
        retrieval_config = body.get("retrieval_config")  # 支持传入自定义配置文件路径

        logger.info(f"开始搜索请求，查询: {query}, parseType={parseType}, config={retrieval_config}")
        if retrieval_config:
            logger.info(f"检测到自定义检索配置，重新初始化 RetrieverModule: {retrieval_config}")
            RM.init_system(retrieval_config=retrieval_config, logging=logger)
        start_time = time.time()
        result = RM.run(query=query,
                        retrieval_config=retrieval_config, logging=logger, knowledgeUUID=knowledgeUUID, parseType=parseType)
        # 记录结束时间
        end_time = time.time()
        # 计算并打印时间消耗
        time_consumed = end_time - start_time
        logger.info(f"总检索方法执行时间: {time_consumed:.4f} 秒")
        documents = result.get("documents", [])
        time_costs = result.get("time_costs", {})
        logger.info(f"搜索完成，找到 {len(documents)} 条结果，耗时明细: {time_costs}")
        return {'code': 0, 'data': {"documents": documents, "time_costs": time_costs}}
    except Exception as e:
        logger.info(f"搜索失败: {str(e)}", exc_info=True)
        return {'code': 1, 'msg': str(e)}

@app.post("/search_with_wiki")
async def search_with_wiki_endpoint(request: Request):
    """搜索（带 LLM Wiki 图谱增强）"""
    try:
        body = await request.json()
        query = body.get("query")
        knowledgeUUID = body.get("knowledgeUUID")
        parseType = body.get("parseType")
        retrieval_config = body.get("retrieval_config")  # 支持传入自定义配置文件路径
        wiki_min_relevance = body.get("wiki_expansion_min_relevance", 2.0)  # 新增：关联度阈值

        logger.info(f"开始Wiki增强搜索请求，查询: {query}, parseType={parseType}, config={retrieval_config}, wiki_min_relevance={wiki_min_relevance}")
        if retrieval_config:
            logger.info(f"检测到自定义检索配置，重新初始化 RetrieverModule: {retrieval_config}")
            RM.init_system(retrieval_config=retrieval_config, logging=logger)
        start_time = time.time()
        result = RM.run(query=query,
                        retrieval_config=retrieval_config, logging=logger, knowledgeUUID=knowledgeUUID, parseType=parseType, use_wiki=True,
                        wiki_expansion_min_relevance=wiki_min_relevance)
        end_time = time.time()
        time_consumed = end_time - start_time
        documents = result.get("documents", [])
        time_costs = result.get("time_costs", {})
        logger.info(f"总Wiki增强检索方法执行时间: {time_consumed:.4f} 秒，耗时明细: {time_costs}")
        logger.info(f"Wiki增强搜索完成，找到 {len(documents)} 条结果")
        return {'code': 0, 'data': {"documents": documents, "time_costs": time_costs}}
    except Exception as e:
        logger.info(f"Wiki增强搜索失败: {str(e)}", exc_info=True)
        return {'code': 1, 'msg': str(e)}


@app.get("/wiki_graph")
async def wiki_graph_endpoint():
    """返回完整 Wiki 知识图谱的节点和边数据，供前端可视化使用"""
    try:
        from llm_wiki.graph_relevance import build_retrieval_graph, calculate_relevance
        from llm_wiki.wiki_utils import normalize_path
        from retriever.src.haystack_utils import find_project_root
        import os

        project_path = normalize_path(os.path.join(find_project_root(), "llm_wiki"))
        graph = build_retrieval_graph(project_path, data_version=0)

        nodes = []
        for nid, node in graph.nodes.items():
            nodes.append({
                "id": nid,
                "title": node.title,
                "type": node.type,
                "path": node.path,
                "sources": node.sources,
                "linkCount": len(node.out_links) + len(node.in_links),
            })

        edges = []
        seen = set()
        for nid, node in graph.nodes.items():
            for target_id in node.out_links:
                key = tuple(sorted([nid, target_id]))
                if key in seen:
                    continue
                seen.add(key)
                target = graph.nodes.get(target_id)
                if target:
                    rel = calculate_relevance(node, target, graph)
                    edges.append({
                        "source": nid,
                        "target": target_id,
                        "weight": round(rel, 2),
                    })

        return {"code": 0, "data": {"nodes": nodes, "edges": edges}}
    except Exception as e:
        logger.error(f"Wiki图谱构建失败: {e}")
        return {"code": 1, "msg": str(e)}


@app.post("/wiki_node_related")
async def wiki_node_related_endpoint(request: Request):
    """返回指定 Wiki 节点的相关节点，支持多跳衰减"""
    try:
        body = await request.json()
        node_id = body.get("node_id")
        hops = int(body.get("hops", 2))
        min_relevance = float(body.get("min_relevance", 2.0))
        decay = float(body.get("decay", 0.7))
        limit = int(body.get("limit", 10))

        if not node_id:
            return {"code": 1, "msg": "node_id 不能为空"}

        from llm_wiki.graph_relevance import build_retrieval_graph, get_related_nodes
        from llm_wiki.wiki_utils import normalize_path, read_file_utf8, extract_frontmatter
        from retriever.src.haystack_utils import find_project_root

        project_path = normalize_path(os.path.join(find_project_root(), "llm_wiki"))
        graph = build_retrieval_graph(project_path, data_version=0)

        source_node = graph.nodes.get(node_id)
        if not source_node:
            return {"code": 1, "msg": f"节点 {node_id} 不存在"}

        # Read node content and frontmatter
        node_content = ""
        node_frontmatter = {}
        try:
            raw_content = read_file_utf8(source_node.path)
            fm, body = extract_frontmatter(raw_content)
            node_frontmatter = fm
            node_content = body
        except Exception:
            node_content = ""

        visited = {node_id}
        results = []
        frontier = [(node_id, 1.0)]  # (node_id, current_decay_factor)

        for hop in range(hops):
            next_frontier = []
            for current_id, decay_factor in frontier:
                related = get_related_nodes(current_id, graph, limit=limit)
                for item in related:
                    rel_node = item["node"]
                    relevance = float(item["relevance"])
                    effective_relevance = relevance * decay_factor

                    if effective_relevance < min_relevance:
                        continue

                    if rel_node.id not in visited:
                        visited.add(rel_node.id)
                        results.append({
                            "id": rel_node.id,
                            "title": rel_node.title,
                            "relevance": round(relevance, 2),
                            "effective_relevance": round(effective_relevance, 2),
                            "path": rel_node.path,
                            "type": rel_node.type,
                            "hop": hop + 1
                        })
                        next_frontier.append((rel_node.id, decay_factor * decay))
            frontier = next_frontier
            if not frontier:
                break

        results.sort(key=lambda x: -x["effective_relevance"])

        return {
            "code": 0,
            "data": {
                "node_id": node_id,
                "node_title": source_node.title,
                "node_type": source_node.type,
                "node_path": source_node.path,
                "node_content": node_content,
                "node_frontmatter": node_frontmatter,
                "related_nodes": results
            }
        }
    except Exception as e:
        logger.error(f"Wiki节点相关查询失败: {e}", exc_info=True)
        return {"code": 1, "msg": str(e)}


@app.get("/health", summary="健康检查")
async def root():
    logger.info("<UNK>")
    """
    根路径健康检查
    """
    return {"status": "healthy", "message": "检索服务正常运行"}


@app.post("/test")
def upload_file_endpoint():
    return {"message": "Hello, World!"}




if __name__ == "__main__":
    import uvicorn

    logger.info(f"启动FastAPI服务，监听 {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        reload=False
    )
