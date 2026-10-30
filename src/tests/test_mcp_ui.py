"""真实 stdio 集成与 Streamlit 交互回归；不调用云端模型。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from mcp_client import call_mcp
from settings import ROOT, SOURCE_DIR


class MCPTests(unittest.TestCase):
    def test_real_tool_discovery_and_price_call(self):
        result = call_mcp('get_prices', {'vegetable': '番茄', 'date': '2026-10-26'})
        self.assertEqual(result['mcp']['server'], 'veggie-tools')
        self.assertEqual(result['mcp']['transport'], 'stdio')
        self.assertIn('search_documents', result['mcp']['tools'])
        self.assertEqual(len(result['data']['records']), 2)
        self.assertTrue(result['data']['is_demo'])

    def test_missing_price_date_stays_empty(self):
        result = call_mcp('get_prices', {'vegetable': '番茄', 'date': '2026-10-27'})
        self.assertEqual(result['data']['records'], [])

    def test_real_document_tool(self):
        result = call_mcp('search_documents', {'question': '周转筐重量是否计入净重', 'method': 'bm25'})
        self.assertEqual(result['data']['contexts'][0]['section'], '净重与去皮')


class UITests(unittest.TestCase):
    def test_initial_page_and_local_question(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(SOURCE_DIR / 'app.py'), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 4)
        app.radio[0].set_value('bm25')
        app.toggle[0].set_value(False)
        app.run()
        app.chat_input[0].set_value('周转筐的重量要计入净重吗？').run()
        self.assertEqual(len(app.exception), 0)
        result = app.session_state['history'][-1]
        self.assertEqual(result['status'], 'evidence_preview')
        self.assertEqual(result['sources'][0]['section'], '净重与去皮')
        self.assertIn('[S1]', result['answer'])

    def test_price_form_calls_real_mcp(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(SOURCE_DIR / 'app.py'), default_timeout=30).run()
        next(button for button in app.button if button.label == '通过 MCP 查询菜价').click().run()
        self.assertEqual(len(app.exception), 0)
        result = app.session_state['price_result']
        self.assertEqual(result['mcp']['tool'], 'get_prices')
        self.assertEqual(result['data']['as_of'], '2026-10-26')
        self.assertEqual(len(result['data']['records']), 8)


if __name__ == '__main__':
    unittest.main()
