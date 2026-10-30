"""离线单元回归：FakeEmbedder 仅用于隔离外部服务，不用于项目实测报告。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from documents import read_document, save_upload, split_documents
from evaluate import retrieval_metrics
from prices import lookup_prices
from qa import answer_question, generate
from retrieval import BM25, Embedder, Retriever, build_index, rrf
from settings import ROOT


class FakeEmbedder:
    def __init__(self):
        self.identity = {'model': 'unit-test-only', 'service_hash': 'unit-test-only'}

    def encode(self, texts):
        rows = np.asarray([[1 + text.count('净重'), 1 + text.count('报价'), 1] for text in texts], dtype=np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'storage')
        self.directory = Path(self.temp.name)
        self.source = self.directory / 'sample.md'
        self.source.write_text('# 验收\n\n## 净重\n净重不包含周转筐。\n\n## 报价\n报价要对齐单位。', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_read_markdown_and_text(self):
        self.assertEqual([s['section'] for s in read_document(self.source)], ['净重', '报价'])
        text = self.directory / 'a.txt'
        text.write_text('一段UTF-8说明', encoding='utf-8')
        self.assertEqual(read_document(text)[0]['text'], '一段UTF-8说明')

    def test_real_pdf_keeps_page(self):
        sections = read_document(ROOT / 'data/knowledge/07_demo_receiving.pdf')
        self.assertEqual(sections[0]['page'], 1)
        self.assertIn('pending review', sections[0]['text'])

    def test_split_covers_long_text_and_keeps_metadata(self):
        section = {'source': 'long.txt', 'section': '测试', 'page': None, 'text': '字' * 1000}
        chunks = split_documents([section], size=120, overlap=20)
        covered = set().union(*(set(range(c['start'], c['end'])) for c in chunks))
        self.assertEqual(covered, set(range(1000)))
        self.assertTrue(all(len(c['text']) <= 120 and c['source'] == 'long.txt' for c in chunks))
        with self.assertRaises(ValueError):
            split_documents([section], size=100, overlap=100)

    def test_upload_filename_cannot_escape_directory(self):
        saved = save_upload('../../escape.md', b'# example', self.directory)
        self.assertEqual(saved.parent, self.directory)
        with self.assertRaises(ValueError):
            save_upload('.env', b'secret', self.directory)

    def test_bm25_and_rrf(self):
        self.assertGreater(BM25(['净重不包含筐', '报价统一单位']).score('净重')[0], 0)
        self.assertEqual(BM25(['纯中文']).score('nonexistent').tolist(), [0])
        ordered, scores = rrf([[1, 2, 2], [2, 3]])
        self.assertEqual(ordered[0], 2)
        self.assertAlmostEqual(scores[2], 1 / 62 + 1 / 61)

    def test_index_roundtrip_and_stale_source(self):
        path, embedding = self.directory / 'index.npz', FakeEmbedder()
        build_index(embedding, [self.source], path)
        with patch('retrieval.document_paths', return_value=[self.source]):
            retriever = Retriever(path, embedding)
            for method in ('bm25', 'vector', 'hybrid'):
                self.assertTrue(retriever.search('净重', method)['contexts'])
            self.source.write_text('文档被更新', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '过期'):
                Retriever(path, embedding)

    def test_embedding_change_requires_rebuild(self):
        path, embedding = self.directory / 'index.npz', FakeEmbedder()
        build_index(embedding, [self.source], path)
        embedding.identity['model'] = 'another-model'
        retriever = Retriever(path, embedding, check_documents=False)
        with self.assertRaisesRegex(ValueError, '不能混用'):
            retriever.search('净重', 'vector')

    def test_failed_build_keeps_previous_index(self):
        path, embedding = self.directory / 'index.npz', FakeEmbedder()
        build_index(embedding, [self.source], path)
        previous = path.read_bytes()
        with patch.object(embedding, 'encode', side_effect=RuntimeError('模拟接口失败')):
            with self.assertRaises(RuntimeError):
                build_index(embedding, [self.source], path)
        self.assertEqual(path.read_bytes(), previous)

    def test_embedding_rejects_zero_vector_and_wrong_order(self):
        with patch('retrieval.model_config', return_value={'key': 'fake', 'base': 'https://example.test/v1', 'model': 'fake'}):
            embedder = Embedder()
        for data in ([{'index': 0, 'embedding': [0, 0]}], [{'index': 2, 'embedding': [1, 0]}]):
            with patch('retrieval.post_model', return_value={'data': data}):
                with self.assertRaises(RuntimeError):
                    embedder.encode(['text'])

    def test_prices_never_fall_back_to_another_date(self):
        self.assertEqual(lookup_prices('番茄', '2026-10-27')['records'], [])
        self.assertEqual(lookup_prices('番茄')['as_of'], '2026-10-26')
        self.assertTrue(lookup_prices('番茄')['is_demo'])
        with self.assertRaises(ValueError):
            lookup_prices('番茄', '2026-99-99')

    def test_unknown_citation_is_reported(self):
        response = {'choices': [{'message': {'content': '错误结论[S99]'}, 'finish_reason': 'stop'}]}
        context = {'id': 'a', 'text': '原文', 'source': 'a.md', 'section': '一', 'page': None}
        with patch('qa.model_config', return_value={'model': 'mock'}), patch('qa.llm_extra', return_value={}), \
             patch('qa.post_model', return_value=response):
            result = generate('问题', [context])
        self.assertEqual(result['status'], 'citation_error')
        self.assertNotIn('错误结论', result['answer'])

    def test_empty_evidence_does_not_call_model(self):
        class EmptyRetriever:
            def search(self, *args):
                return {'contexts': [], 'seconds': 0}
        with patch('qa.generate') as model:
            self.assertEqual(answer_question('不存在', retriever=EmptyRetriever())['status'], 'no_evidence')
            model.assert_not_called()

    def test_recall_deduplicates_overlapping_chunks(self):
        result = retrieval_metrics([{'source': 'a', 'section': 'x'}] * 3,
                                   [{'source': 'a', 'section': 'x'}, {'source': 'b', 'section': 'y'}])
        self.assertEqual(result['recall_at_3'], .5)

    def test_benchmark_evidence_quotes_really_exist(self):
        questions = [json.loads(line) for line in (ROOT / 'data/benchmark.jsonl').read_text(encoding='utf-8').splitlines()]
        for question in questions:
            for expected in question['evidence']:
                sections = read_document(ROOT / 'data/knowledge' / expected['source'])
                section = next(s for s in sections if s['section'] == expected['section'])
                self.assertIn(expected['quote'], section['text'], question['id'])


if __name__ == '__main__':
    unittest.main()
