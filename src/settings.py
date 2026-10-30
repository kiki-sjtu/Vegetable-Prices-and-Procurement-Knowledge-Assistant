"""集中配置：默认只读本项目 .env，也支持显式指定已有模型配置。"""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import dotenv_values

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
DOCUMENT_DIRS = (ROOT / 'data' / 'knowledge', ROOT / 'data' / 'uploads')
INDEX_FILE = ROOT / 'storage' / 'index.npz'
METHODS = {'bm25': 'BM25 关键词', 'vector': '向量检索', 'hybrid': '混合检索 · RRF'}
CHUNK_SIZE, OVERLAP, CANDIDATE_K, RRF_K = 450, 70, 10, 60


def settings():
    local = dotenv_values(ROOT / '.env')
    inherited_path = os.getenv('MODEL_ENV_FILE') or local.get('MODEL_ENV_FILE')
    inherited = dotenv_values(inherited_path) if inherited_path else {}
    values = {**inherited, **{k: v for k, v in local.items() if v}}
    for key in ('LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL', 'LLM_EXTRA_JSON',
                'EMBEDDING_API_KEY', 'EMBEDDING_BASE_URL', 'EMBEDDING_MODEL'):
        if os.getenv(key):
            values[key] = os.environ[key]
    return values


def model_config(kind):
    values = settings()
    if kind == 'embedding':
        key = values.get('EMBEDDING_API_KEY') or values.get('LLM_API_KEY')
        base = values.get('EMBEDDING_BASE_URL') or values.get('LLM_BASE_URL')
        model = values.get('EMBEDDING_MODEL')
    else:
        key, base, model = (values.get(k) for k in ('LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL'))
    if not all((key, base, model)) or '填写' in model or 'replace_with' in str(base):
        raise ValueError(f'请先在 .env 配置 {kind} 的服务地址、模型名和密钥。')
    parsed = urlsplit(base)
    if parsed.username or parsed.password or not parsed.hostname:
        raise ValueError('服务地址不能包含账号或密码。')
    if parsed.scheme != 'https' and not (
        parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1')
    ):
        raise ValueError('远程模型服务须使用 HTTPS；本机服务允许 HTTP。')
    return {'key': key, 'base': base.rstrip('/'), 'model': model}


def llm_extra():
    extra = json.loads(settings().get('LLM_EXTRA_JSON') or '{}')
    if not isinstance(extra, dict) or set(extra) & {'model', 'messages', 'stream'}:
        raise ValueError('LLM_EXTRA_JSON 需要是字典，且不能覆盖 model/messages/stream。')
    return extra
