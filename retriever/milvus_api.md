# Hybrid Retrieval API 规范

## 接口功能
提供一个中间层接口，用于构造和执行 Milvus 向量数据库的混合查询。

## 接口名称
`hybrid_retrieval`

## 请求方法
POST

## 请求地址
`/api/retrieval/hybrid_retrieval`

## 请求参数
```json
{
  "query_sparse_embedding": "object",  // 必填，查询的稀疏向量表示
  "query_embedding": "Array<number>",  // 必填，查询的稠密向量表示
  "options": {  // 检索选项配置
    "filters": "object",  // 可选，过滤条件，用于限定检索范围
    "top_k": "number",  // 可选，返回最相关的k个结果，控制返回结果的数量
    "reranker": "object"  // 可选，重排序器配置，用于对初步检索结果进行重新排序，可能包含模型参数或排序策略
  }
}
```
##  响应格式
```json
{
  "code": "number",  
  "message": "string",  
  "data": {  // 返回的检索结果数据
    "documents": [  // 匹配的文档列表，按相关性排序
      {
        "id": "string",  // 文档唯一标识符
        "content": "string",  // 文档内容
        "content_type": "string",  // 内容类型（如 "text"、"table"等）
        "meta": "object",  // 文档元数据
        "embedding": "Array<number>",  // 文档的稠密向量表示
        "query_sparse_embedding": "object",  // 文档的稀疏向量表示
        "score": "float"  // 相关性分数
      }
    ]
  }
}
```

## 调用代码

```python
elif self.retrieval_method == "hybrid_retriever":
    logging.info("使用混合检索方法（稠密+稀疏）进行检索")
    for query in query_embedding:
        logging.debug("处理第 %d 个查询的混合检索", query_embedding.index(query) + 1)
        doc_filter={
            "operator": "OR",
            "conditions": [
                {
                    "field": "metadata['file_name']",
                    "operator": "==",
                    "value": filename
                }
                for filename in self.filename
            ]
        }
        # 构造请求参数
        request_data = {
            "query_sparse_embedding": query['sparse_embedding'],
            "query_embedding": query['embedding'],
            "options": {
                "filters": doc_filter, 
                "top_k": self.retrieval.get('filter_top_k'), 
                "reranker": self.retrieval.get('reranker') 
            }
        }
        
        try:
            # 调用检索接口
            response = requests.post(
                "http://api-server/api/retrieval/hybrid",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=300  # 设置超时时间
            )
            response.raise_for_status()  # 检查HTTP错误
            
            result = response.json()
            if result['code'] != 200:
                logging.error(f"检索接口返回错误: {result['message']}")
                continue
                
            documents = result['data']['documents']
            logging.debug(f"检索成功，获取到 {len(documents)} 个文档")
            
        except requests.exceptions.RequestException as e:
            logging.error(f"调用检索接口失败: {str(e)}")
            continue
```
# Milvus 过滤检索接口规范
## 接口功能
提供一个中间层接口，用于构造和执行 Milvus 向量数据库的过滤查询。
## 接口名称
`filtered_retrieval`

## 请求方法
POST

## 请求地址
`/api/retrieval/filtered_retrieval`
### 输入规范
```json
{
  "db_name": "string",  // 必填，要查询的向量库名称
  "collection_name": "string",  // 必填，要查询的集合名称
  "filters": "object",         // 可选，过滤条件
  "top_k": "number"          // 可选，返回结果数量限制
}
```
### 输出规范
```json
{
  "status": "string",          
  "message": "string",
  "data": {  // 返回的检索结果数据
    "documents": [  // 匹配的文档列表，按相关性排序
      {
        "id": "string",  // 文档唯一标识符
        "content": "string",  // 文档内容
        "content_type": "string",  // 内容类型（如 "text"、"table"等）
        "meta": "object",  // 文档元数据
        "embedding": "Array<number>",  // 文档的稠密向量表示
        "query_sparse_embedding": "object"  // 文档的稀疏向量表示
      }
    ]
  }
}
```

## 调用代码
```python
class FilteredRetrievalClient:
    def __init__(self, base_url: str = "http://api-server"):
        """
        初始化过滤检索API客户端
        
        参数:
            base_url: API服务器的基地址（需去除终端路径）
        """
        # 修正：确保基础URL不包含终端路径
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/api/retrieval/filtered_retrieval"  # 固定接口路径
        self.timeout = 300

    def retrieve(
        self,
        db_name: str,
        collection_name: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        # 请求数据准备（不变）
        request_data = {
            "db_name": db_name,
            "collection_name": collection_name
        }
        if filters is not None:
            request_data["filters"] = filters
        if top_k is not None:
            request_data["top_k"] = top_k
        
        try:
            logging.info("正在执行过滤检索请求")
            response = requests.post(
                self.endpoint,  # 使用预构建的正确端点
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            # 首先检查HTTP状态码
            response.raise_for_status()
            
            # 然后检查业务状态
            result = response.json()
            if result.get('status') != "success":
                error_msg = f"API业务错误: {result.get('message')}"
                logging.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "data": None
                }
                
            # 修正日志级别为INFO
            doc_count = len(result.get('data', {}).get('documents', []))
            logging.info(f"过滤检索成功，获取到 {doc_count} 个文档")
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"请求失败: {str(e)}"
            logging.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "data": None
            }
```

