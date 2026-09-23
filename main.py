import asyncio
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import Any

import duckdb
import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/buckets/Apimaking/icrm-hitek-full-db-mixed-bucket-new/resolve",
).rstrip("/")

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2
RATE_LIMIT_PER_MIN = int(os.environ.get("ICMR_RATE_LIMIT", "20"))

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

REMOTE_INDEXES = {
    "phone":  [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet"  for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── In-memory cache ─────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
CACHE_MAX = 5000


def _cache_get(key: str):
    with _cache_lock:
        return _cache.get(key)


def _cache_set(key: str, value: dict):
    with _cache_lock:
        if len(_cache) >= CACHE_MAX:
            for k in list(_cache.keys())[: CACHE_MAX // 10]:
                _cache.pop(k, None)
        _cache[key] = value


# ── Rate limiter ────────────────────────────────────────────────────────────
_hits: dict[str, list[float]] = defaultdict(list)
_hits_lock = threading.Lock()


def _rate_ok(ip: str) -> bool:
    now = time()
    with _hits_lock:
        _hits[ip] = [t for t in _hits[ip] if now - t < 60]
        if len(_hits[ip]) >= RATE_LIMIT_PER_MIN:
            return False
        _hits[ip].append(now)
        return True


# ── DuckDB pool ─────────────────────────────────────────────────────────────
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    try:
        con.execute("INSTALL parquet; LOAD parquet;")
        con.execute("INSTALL httpfs;  LOAD httpfs;")
    except Exception:
        con.execute("LOAD parquet;")
        con.execute("LOAD httpfs;")

    con.execute("SET enable_http_metadata_cache=true")
    con.execute("SET enable_object_cache=true")
    con.execute("SET prefetch_all_parquet_files=true")

    if HF_TOKEN:
        try:
            con.execute(
                f"CREATE OR REPLACE SECRET hf_secret "
                f"(TYPE huggingface, PROVIDER config, TOKEN '{HF_TOKEN}')"
            )
        except Exception as e:
            print(f"[HF token secret failed] {e}")

    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        lst = ", ".join(f"'{u}'" for u in urls)
        con.execute(
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT * FROM read_parquet([{lst}])"
        )
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con


def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid


def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


# ── Dedup ───────────────────────────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()


def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected


def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out


# ── Search ──────────────────────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")

    if mode == "exact":
        if field == "phoneNumber" and _idx_ready("phone"):
            view = "people_phone"
        elif field == "aadharNumber" and _idx_ready("aadhar"):
            view = "people_aadhar"
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        sql = f"SELECT * FROM {view} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    elif mode == "contains":
        if field != "name":
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = f"SELECT * FROM people_phone WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}


def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    cache_key = f"{q}|{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if not (q.isdigit() and len(q) >= 8):
        result = {"query": q, "searched_fields": [], "count": 0, "results": []}
        _cache_set(cache_key, result)
        return result

    all_rows, searched = [], []
    if _idx_ready("phone"):
        try:
            r = _run_field_search("phoneNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("phoneNumber")
        except Exception as e:
            print(f"[phone error] {e}")

    if not all_rows and _idx_ready("aadhar"):
        try:
            r = _run_field_search("aadharNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("aadharNumber")
        except Exception as e:
            print(f"[aadhar error] {e}")

    all_rows = _cap_duplicates(all_rows)[:limit]
    result = {
        "query": q, "searched_fields": searched,
        "count": len(all_rows), "results": all_rows,
    }
    _cache_set(cache_key, result)
    return result


# ── FastAPI ─────────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="ICMR + HITEK Search API")


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "records": 2_504_793_870,
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
        "docs": "/docs",
        "developer": "@kzr0x | channel @api_wallah",
    }


@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "cache_size": len(_cache),
        "hf_token": bool(HF_TOKEN),
    }


@fastapi_app.get("/search")
async def search(
    request: Request,
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=100),
    pretty: bool = Query(True),
):
    ip = request.client.host if request.client else "unknown"
    if not _rate_ok(ip):
        raise HTTPException(429, f"Rate limit: {RATE_LIMIT_PER_MIN} requests/minute per IP")

    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile")

    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(pool, _run_field_search, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)

    result = {"success": bool(data["count"]), **data, "number": q_val, "total": data["count"]}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@fastapi_app.post("/search/parallel")
async def search_parallel(request: Request, req: BatchRequest):
    ip = request.client.host if request.client else "unknown"
    if not _rate_ok(ip):
        raise HTTPException(429, f"Rate limit: {RATE_LIMIT_PER_MIN} requests/minute per IP")
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 20:
        raise HTTPException(400, "max 20 queries per batch")

    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            pool, _run_field_search,
            item.get("field", "phoneNumber"),
            item.get("value", ""),
            item.get("mode", "exact"),
            int(item.get("limit", req.limit)),
        )
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return Response(
        content=json.dumps({"searches": len(req.queries), "results": list(results)},
                           indent=2, ensure_ascii=False),
        media_type="application/json",
    )


# ── Auto-pinger ─────────────────────────────────────────────────────────────
async def _pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=15) as client:
        while True:
            await asyncio.sleep(120)
            for _ in range(3):
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        break
                except Exception:
                    await asyncio.sleep(10)


@fastapi_app.on_event("startup")
async def _startup():
    asyncio.create_task(_pinger())


# ── Gradio UI ───────────────────────────────────────────────────────────────
def _format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val:
            lines.append(f"**{field}:** {val}")
    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")
    return "\n\n".join(lines)


def _search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Kuch toh search karo — phone, aadhar, ya name daalo."
    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"

    count = data["count"]
    results = data["results"]
    searched = ", ".join(data.get("searched_fields", []))

    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n❌ **No data found**."

    header = f"🔍 **Query:** `{q}`  |  **Found:** {count}  |  **Searched:** {searched}\n\n---\n\n"
    parts = [f"### Result {i}\n{_format_result(r)}" for i, r in enumerate(results, 1)]
    return header + "\n\n---\n\n".join(parts)


def _build_ui():
    with gr.Blocks(title="ICMR Search API", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API")
        gr.Markdown("Search **2.5 billion records** — phone, Aadhaar, name & more")
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Search Query", placeholder="Phone or Aadhaar...", lines=1)
            with gr.Column(scale=1):
                limit_slider = gr.Slider(1, 50, 10, step=1, label="Max Results")
        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")
        search_btn.click(fn=_search_ui, inputs=[query_input, limit_slider], outputs=output)
        query_input.submit(fn=_search_ui, inputs=[query_input, limit_slider], outputs=output)
        gr.Markdown("---\n<div style='text-align:center'>👨‍💻 @kzr0x | 📢 @api_wallah</div>")
    return demo


demo = _build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
