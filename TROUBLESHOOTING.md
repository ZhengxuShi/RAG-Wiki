# Troubleshooting / 常见问题排查

## 1. ModuleNotFoundError after `pip install`

**Symptom**: `ModuleNotFoundError: No module named 'loguru'` (or `httpx`, `pydantic`, etc.)

**Cause**: `requirements-retriever.txt` may be missing some transitive dependencies.

**Fix**:
```bash
pip install loguru httpx pydantic more_itertools tqdm jinja2 networkx python-louvain
```

---

## 2. `FileNotFoundError: Could not find main.py`

**Symptom**: Crash on startup with `FileNotFoundError: Could not find main.py in any parent directory.`

**Cause**: `find_project_root()` searches upward for `main.py` to determine the project root.

**Fix**: Ensure you run services from the repo root (`FW_RAG_OpenSource/`), or create an empty `main.py` in the root.

---

## 3. Milvus connection errors

**Symptom**: `pymilvus.exceptions.MilvusException: code=StatusCode.UNAVAILABLE`

**Cause**: Milvus is not running, or the database/collection does not exist.

**Fix**:
1. Start Milvus: `docker start milvus-standalone` (or recreate it)
2. Run `scripts/init_milvus.py` to create the database and collection
3. Check `configs/retriever_config_v3.json` for correct `milvus_uri`

---

## 4. `KB_BUILD_KNOWLEDGE_UUID` not set

**Symptom**: `ValueError: 环境变量 KB_BUILD_KNOWLEDGE_UUID 未设置`

**Cause**: Running `ingest.py` without setting this required environment variable.

**Fix**:
```bash
export KB_BUILD_KNOWLEDGE_UUID="your-uuid-here"
```

---

## 5. Frontend shows 404 or missing CSS/JS

**Symptom**: `http://localhost:8000/` shows plain HTML without styling, or returns 404.

**Cause**: `api_gateway.py` serves `static/index.html` relative to its working directory.

**Fix**: Always start `api_gateway.py` from the repo root:
```bash
cd /path/to/FW_RAG_OpenSource
python api_gateway.py
```

---

## 6. `api_gateway.py` returns "Connection refused"

**Symptom**: Queries through the gateway fail with connection errors.

**Cause**: `api.py` (the RAG backend) is not running, or `RAG_URL` / `WIKI_URL` in `.env` point to the wrong address.

**Fix**:
1. Start `api.py` first: `python api.py`
2. Verify `.env` contains:
   ```
   RAG_URL=http://127.0.0.1:17101
   WIKI_URL=http://127.0.0.1:17101
   ```
3. Then start `api_gateway.py`

---

## 7. Embedding / Reranker API errors

**Symptom**: `openai.APIConnectionError` or timeout during retrieval.

**Cause**: The external embedding/reranker/LLM services are not running or the URLs in `configs/retriever_config_v3.json` are incorrect.

**Fix**:
1. Verify your services are up (e.g., `curl http://localhost:8001/v1/models`)
2. Update `configs/retriever_config_v3.json` with the correct endpoints
3. Increase timeouts in `.env` if services are slow:
   ```
   OPENAI_TIMEOUT=60.0
   ```

---

## 8. Wiki pages not found in search

**Symptom**: Wiki path returns no results even though pages exist in `llm_wiki/wiki/`.

**Cause**: The wiki graph cache has not been built or refreshed since the pages were added.

**Fix**:
1. Restart `api.py` to trigger graph loading
2. Or call the graph refresh endpoint if exposed by your API
