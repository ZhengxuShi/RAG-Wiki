#!/usr/bin/env python3
"""
Milvus 初始化脚本
创建项目默认使用的 database 和 collection。

使用前请确保：
1. Milvus 已启动（默认连接 localhost:19530）
2. 已安装 pymilvus: pip install pymilvus
"""

import os
import sys

from pymilvus import connections, db, Collection, CollectionSchema, FieldSchema, DataType

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DB_NAME = os.getenv("MILVUS_DB_NAME", "Firmware")
COLLECTION_NAME = "bge_m3_wiki_demo100_v1"
DENSE_DIM = 1024  # bge-m3 dense vector dimension


def parse_uri(uri: str):
    """从 http://host:port 解析 host 和 port"""
    uri = uri.replace("http://", "").replace("https://", "")
    if ":" in uri:
        host, port = uri.split(":")
        return host, int(port)
    return uri, 19530


def init_milvus():
    host, port = parse_uri(MILVUS_URI)
    connections.connect(alias="default", host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")

    # 1. 创建 database（若不存在则创建）
    existing_dbs = db.list_database()
    if DB_NAME not in existing_dbs:
        db.create_database(DB_NAME)
        print(f"Created database: {DB_NAME}")
    else:
        print(f"Database already exists: {DB_NAME}")

    # 切换到目标 database
    db.using_database(DB_NAME)

    # 2. 创建 collection（若不存在则创建）
    existing_collections = Collection.list_collections()
    if COLLECTION_NAME in existing_collections:
        print(f"Collection already exists: {COLLECTION_NAME}")
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DENSE_DIM),
        FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="knowledgeUUID", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="meta", dtype=DataType.JSON),
    ]

    schema = CollectionSchema(fields, description="RAG dense + sparse vector collection")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"Created collection: {COLLECTION_NAME}")

    # 3. 创建索引（可选，建议首次写入数据后创建）
    # 若 collection 为空，部分 Milvus 版本不允许建索引，故此处仅打印提示
    print("\nNote: Please create indexes after the first batch of documents is inserted.")
    print("Example index for dense vector (IVF_FLAT):")
    print(f'  collection.create_index("embedding", {{"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {{"nlist": 128}}}})')
    print("Example index for sparse vector:")
    print(f'  collection.create_index("sparse_embedding", {{"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}})')

    connections.disconnect("default")
    print("\nDone.")


if __name__ == "__main__":
    init_milvus()
