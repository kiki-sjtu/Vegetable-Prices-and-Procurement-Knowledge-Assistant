"""真实远程 Embedding + 手写 BM25 + RRF；小语料直接用 NumPy 精确检索。"""
from collections import Counter
import hashlib
import json
import math
import os
import re
import time
import httpx
import numpy as np
from documents import document_paths, fingerprint, read_document, split_documents
from settings import CANDIDATE_K, CHUNK_SIZE, INDEX_FILE, METHODS, OVERLAP, RRF_K, model_config


def post_model(config, path, payload):
    try:
        with httpx.Client(timeout=45) as client:
            response = client.post(config['base'] + path, json=payload,
                                   headers={'Authorization': 'Bearer ' + config['key']})
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as error:
        raise RuntimeError(f'模型接口 HTTP {error.response.status_code}；请核对地址、模型和额度。') from None
    except (httpx.RequestError, ValueError):
        raise RuntimeError('模型接口连接超时或返回非 JSON，请检查服务与网络。') from None


class Embedder:
    def __init__(self):
        self.config = model_config('embedding')
        self.identity = {'model': self.config['model'], 'service_hash': hashlib.sha256(
            self.config['base'].encode()).hexdigest()[:12]}

    def encode(self, texts):
        result = post_model(self.config, '/embeddings',
                            {'model': self.config['model'], 'input': texts, 'encoding_format': 'float'})
        rows = sorted(result.get('data', []), key=lambda row: row['index'])
        if [row['index'] for row in rows] != list(range(len(texts))):
            raise RuntimeError('Embedding 返回条数或索引不一致。')
        matrix = np.asarray([row['embedding'] for row in rows], dtype=np.float32)
        if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
            raise RuntimeError('Embedding 返回了无效向量。')
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError('Embedding 返回了零向量。')
        return matrix / norms


def tokens(text):
    text = text.lower()
    words = re.findall(r'[a-z0-9_]+', text)
    for part in re.findall(r'[\u4e00-\u9fff]+', text):
        words.extend(part[i:i + 2] for i in range(len(part) - 1))
        if len(part) == 1:
            words.append(part)
    return words


class BM25:
    def __init__(self, texts):
        self.counts = [Counter(tokens(t)) for t in texts]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg = (sum(self.lengths) / len(self.lengths) if self.lengths else 0) or 1
        df = Counter(word for count in self.counts for word in count)
        self.idf = {word: math.log1p((len(texts) - n + .5) / (n + .5)) for word, n in df.items()}

    def score(self, query):
        scores = []
        for count, length in zip(self.counts, self.lengths):
            normalizer = 1.2 * (.25 + .75 * length / self.avg)
            scores.append(sum(self.idf.get(w, 0) * count[w] * 2.2 / (count[w] + normalizer)
                              for w in set(tokens(query)) if count[w]))
        return np.asarray(scores)


def rrf(rankings, k=RRF_K):
    scores = {}
    for ranking in rankings:
        for rank, index in enumerate(dict.fromkeys(ranking), 1):
            scores[index] = scores.get(index, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda index: (-scores[index], index)), scores


def create_index(embedder=None, paths=None):
    started = time.perf_counter()
    paths = document_paths() if paths is None else paths
    if not paths:
        raise ValueError('请先放入文档。')
    before = fingerprint(paths)
    chunks = split_documents([section for path in paths for section in read_document(path)])
    embedder = embedder or Embedder()
    batches = [embedder.encode([f"{c['section']}\n{c['text']}" for c in chunks[i:i + 8]])
               for i in range(0, len(chunks), 8)]
    vectors = np.vstack(batches)
    if fingerprint(paths) != before:
        raise ValueError('建库时资料发生变化，请重新建库。')
    manifest = {**embedder.identity, 'corpus_hash': before, 'chunk_size': CHUNK_SIZE,
                'overlap': OVERLAP, 'documents': len(paths), 'chunks': len(chunks),
                'dimensions': vectors.shape[1], 'build_seconds': round(time.perf_counter() - started, 3)}
    return chunks, vectors, manifest


def build_index(embedder=None, paths=None, output=INDEX_FILE):
    chunks, vectors, manifest = create_index(embedder, paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(file, vectors=vectors, chunks=json.dumps(chunks, ensure_ascii=False),
                            manifest=json.dumps(manifest))
    os.replace(temporary, output)
    return manifest


class Retriever:
    def __init__(self, path=INDEX_FILE, embedder=None, check_documents=True):
        if not path.exists():
            raise ValueError('尚未建库，请先点击“建立 / 更新索引”。')
        with np.load(path, allow_pickle=False) as data:
            self.vectors = data['vectors']
            self.chunks = json.loads(str(data['chunks']))
            self.manifest = json.loads(str(data['manifest']))
        self.embedder = embedder
        self.bm25 = BM25([c['section'] + '\n' + c['text'] for c in self.chunks])
        if len(self.chunks) != len(self.vectors):
            raise ValueError('索引中的块数与向量数不一致，请重新建库。')
        if check_documents and fingerprint(document_paths()) != self.manifest['corpus_hash']:
            raise ValueError('资料已经变化，索引已过期，请重新建库。')
        if (self.manifest['chunk_size'], self.manifest['overlap']) != (CHUNK_SIZE, OVERLAP):
            raise ValueError('切分参数已变化，请重新建库。')

    def search(self, query, method='hybrid', top_k=3):
        if method not in METHODS or not query.strip() or len(query) > 1000 or not 1 <= top_k <= 10:
            raise ValueError('请选择合法方法，问题长度 1～1000 字，top_k 为 1～10。')
        started = time.perf_counter()
        lexical = self.bm25.score(query)
        sparse = [int(i) for i in np.argsort(-lexical, kind='stable') if lexical[i] > 0][:CANDIDATE_K]
        dense_scores = None
        if method in ('vector', 'hybrid'):
            self.embedder = self.embedder or Embedder()
            if any(self.embedder.identity.get(k) != self.manifest.get(k) for k in ('model', 'service_hash')):
                raise ValueError('Embedding 模型或服务已变化，请重新建库，不能混用向量空间。')
            vector = self.embedder.encode([query])[0]
            if len(vector) != self.vectors.shape[1]:
                raise ValueError('查询向量维度与索引不一致，请重新建库。')
            dense_scores = self.vectors @ vector
            dense = [int(i) for i in np.argsort(-dense_scores, kind='stable')[:CANDIDATE_K]]
        if method == 'bm25':
            ordered, scores = sparse, dict(enumerate(lexical))
        elif method == 'vector':
            ordered, scores = dense, dict(enumerate(dense_scores))
        else:
            ordered, scores = rrf([sparse, dense])
        found = [{**self.chunks[i], 'score': round(float(scores[i]), 6),
                  'bm25_score': round(float(lexical[i]), 6),
                  'vector_score': round(float(dense_scores[i]), 6) if dense_scores is not None else None}
                 for i in ordered[:top_k]]
        return {'method': method, 'contexts': found, 'seconds': time.perf_counter() - started}


if __name__ == '__main__':
    print(json.dumps(build_index(), ensure_ascii=False, indent=2))
