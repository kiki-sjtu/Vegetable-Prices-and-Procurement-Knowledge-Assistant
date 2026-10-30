"""真正的 MCP 客户端：启动子进程，握手，发现工具，执行工具并读取结果。"""
import argparse
import asyncio
import json
import os
import sys
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from settings import ROOT, SOURCE_DIR


async def exchange(tool_name=None, arguments=None):
    if tool_name not in (None, 'get_prices', 'search_documents'):
        raise ValueError('未知工具名称。')
    started = time.perf_counter()
    parameters = StdioServerParameters(command=sys.executable, args=[str(SOURCE_DIR / 'mcp_server.py')],
        cwd=str(ROOT), env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
    async with asyncio.timeout(65):
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=55) as session:
                initialized = await session.initialize()
                tools = await session.list_tools()
                metadata = {'server': initialized.server_info.name, 'transport': 'stdio',
                            'protocol': initialized.protocol_version,
                            'tools': [tool.name for tool in tools.tools]}
                if not tool_name:
                    return metadata
                result = await session.call_tool(tool_name, arguments or {})
                if result.is_error:
                    raise RuntimeError('MCP 工具返回错误，请核对名称、参数、日期及索引状态。')
                data = result.structured_content
                if data is None:
                    data = json.loads('\n'.join(block.text for block in result.content if block.type == 'text'))
                return {'data': data, 'mcp': {**metadata, 'tool': tool_name,
                                           'seconds': round(time.perf_counter() - started, 4)}}


def call_mcp(tool_name=None, arguments=None):
    return asyncio.run(exchange(tool_name, arguments))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCP 工具发现与菜价查询')
    parser.add_argument('--vegetable')
    parser.add_argument('--date', default='')
    args = parser.parse_args()
    result = call_mcp('get_prices', {'vegetable': args.vegetable, 'date': args.date}) if args.vegetable else call_mcp()
    print(json.dumps(result, ensure_ascii=False, indent=2))
