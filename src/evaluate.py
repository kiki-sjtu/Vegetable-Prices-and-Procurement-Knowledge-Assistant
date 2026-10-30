"""三方法共享语料与题集，默认只评检索；--generate 才生成回答供人工评分。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import numpy as np
from qa import generate
from retrieval import Retriever
from settings import METHODS, ROOT


def retrieval_metrics(contexts, evidence):
    relevant = {(e['source'], e['section']) for e in evidence}
    if not relevant:
        return {'hit_at_3': None, 'recall_at_3': None, 'mrr_at_3': None}
    positions = [(c['source'], c['section']) for c in contexts[:3]]
    return {'hit_at_3': int(any(p in relevant for p in positions)),
            'recall_at_3': len(set(positions) & relevant) / len(relevant),
            'mrr_at_3': next((1 / (i + 1) for i, p in enumerate(positions) if p in relevant), 0)}


def evaluate(split='all', repeats=1, with_generation=False, output=None):
    questions = [json.loads(line) for line in (ROOT / 'data' / 'benchmark.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    questions = [q for q in questions if split == 'all' or q['split'] == split]
    retriever = Retriever()
    retriever.search('这是一条连接预热用的教学文本', 'vector', 1)
    runs = []
    for repeat in range(repeats):
        for position, question in enumerate(questions):
            # 轮换顺序，降低固定先后顺序对网络延迟的影响；不缓存查询向量。
            methods = list(METHODS)
            offset = (position + repeat) % len(methods)
            for method in methods[offset:] + methods[:offset]:
                result = retriever.search(question['question'], method)
                row = {'id': question['id'], 'split': question['split'], 'repeat': repeat,
                       'question': question['question'], 'answerable': question['answerable'],
                       'method': method, 'seconds': result['seconds'],
                       **retrieval_metrics(result['contexts'], question['evidence']),
                       'retrieved': [{'id': c['id'], 'source': c['source'], 'section': c['section'],
                                      'score': c['score']} for c in result['contexts']]}
                if with_generation:
                    row['generation'] = generate(question['question'], result['contexts'])
                runs.append(row)
    summary = []
    for group in sorted({q['split'] for q in questions}):
        for method in METHODS:
            subset = [r for r in runs if r['method'] == method and r['split'] == group]
            positives = [r for r in subset if r['answerable']]
            summary.append({'split': group, 'method': method, 'questions': len(subset) // repeats,
                'answerable_questions': len(positives) // repeats,
                **{name: round(float(np.mean([r[name] for r in positives])), 4) if positives else None
                   for name in ('hit_at_3', 'recall_at_3', 'mrr_at_3')},
                'p50_ms': round(float(np.percentile([r['seconds'] * 1000 for r in subset], 50)), 2),
                'p95_ms': round(float(np.percentile([r['seconds'] * 1000 for r in subset], 95)), 2)})
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
              'index': retriever.manifest, 'repeats': repeats, 'summary': summary, 'runs': runs,
              'scope': '自编小样本检索诊断；不含答案准确率或拒答率，不能证明真实业务泛化。',
              'latency_scope': '包含查询Embedding网络耗时；不含首次建库、加载索引与连接预热。',
              'generation_enabled': with_generation}
    if output == '-':
        print(json.dumps(report, ensure_ascii=False))
    else:
        output = Path(output) if output else ROOT / 'reports' / 'retrieval_report.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['all', 'dev', 'test'], default='all')
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--generate', action='store_true', help='额外调用模型，结果保留供人工评分')
    parser.add_argument('--output', default=None, help='输出路径；- 表示只输出 JSON，不写文件')
    args = parser.parse_args()
    evaluate(args.split, args.repeats, args.generate, args.output)
