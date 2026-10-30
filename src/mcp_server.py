"""官方 MCP 2.3 SDK，stdio 传输；不要向标准输出写普通日志。"""
from typing import Literal
from mcp.server.mcpserver import MCPServer
from prices import lookup_prices
from retrieval import Retriever

server = MCPServer('veggie-tools', version='1.0.0')


@server.tool()
def get_prices(vegetable: str = '', date: str = '') -> dict:
    """查询教学菜价。date 为 YYYY-MM-DD，留空取最新记录；结果不是实时行情。"""
    return lookup_prices(vegetable, date)


@server.tool()
def search_documents(question: str, method: Literal['bm25', 'vector', 'hybrid'] = 'hybrid') -> dict:
    """检索项目采购知识库，返回原文与文件位置，需要先完成建库。"""
    return Retriever().search(question, method)


if __name__ == '__main__':
    server.run(transport='stdio')
