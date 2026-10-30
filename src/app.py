"""唯一 Web 入口；界面调用普通 Python 函数，避免额外后端与复杂 Agent。"""
import json
from datetime import date
import streamlit as st
from documents import document_paths, read_document, save_upload, split_documents
from mcp_client import call_mcp
from prices import read_prices
from qa import answer_question, generate
from retrieval import Retriever, build_index
from settings import DOCUMENT_DIRS, INDEX_FILE, METHODS, ROOT, model_config

st.set_page_config(page_title='青蔬 · 菜价与采购知识助手', page_icon='🥬', layout='wide')
st.markdown('''<style>
.stApp{background:#f7f8f4;color:#203c31}.block-container{max-width:1260px;padding-top:2rem}
h1,h2,h3{color:#203c31}section[data-testid="stSidebar"]{background:#ecf1e9}
[data-testid="stMetric"]{background:white;border:1px solid #dce5d7;border-radius:12px;padding:15px}
[data-testid="stChatMessage"]{background:white;border:1px solid #e1e8dc;border-radius:12px}
.hero{padding:28px 32px;background:#e5eedf;border:1px solid #d7e2ce;border-radius:18px;margin-bottom:24px}
.hero small{color:#647b58;letter-spacing:.16em}.hero h1{font-size:32px;margin:8px 0}
.hero p{color:#64735f;margin:0;font-size:15px}button{border-radius:8px!important}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><small>QINGSHU / KNOWLEDGE & PRICES</small>
<h1>菜价与采购知识助手</h1><p>查资料、看依据，对齐日期与单位后再比较。</p></div>''', unsafe_allow_html=True)
st.caption('教学演示：内置采购约定与价格均为项目自编示例，不是行业标准或实时行情。')

with st.sidebar:
    st.title('🥬 青蔬')
    method = st.radio('问答检索方式', list(METHODS), format_func=METHODS.get)
    try:
        model_config('chat')
        configured = True
    except ValueError:
        configured = False
    use_llm = st.toggle('根据证据生成回答', value=configured, disabled=not configured)
    if not configured:
        st.info('生成模型未配置。填写 .env 后可启用；也可以先查看检索原文。')
    st.caption('三种方法使用同一套文档和切分结果。分数只用于排序，不表示答案正确概率。')
    st.divider()
    st.write('使用顺序')
    st.write('1. 资料与索引 → 读取并建库\n\n2. 知识问答 → 查看回答与引用\n\n3. 方法对比 → 比较同一道题\n\n4. 菜价工具 → 按日期查示例价格')

paths = document_paths()
index = None
try:
    index = Retriever()
except ValueError as error:
    st.info(str(error))
columns = st.columns(3)
columns[0].metric('知识文档', len(paths))
columns[1].metric('当前索引片段', len(index.chunks) if index else '待建库')
columns[2].metric('检索方法', len(METHODS))
chat, compare, tools, library = st.tabs(['知识问答', '方法对比', '菜价工具 · MCP', '资料与索引'])


def show_sources(sources):
    for number, source in enumerate(sources, 1):
        location = f"第 {source['page']} 页" if source['page'] else source['section']
        label = source.get('source_id', f'第 {number} 条')
        with st.expander(f"{label} · {source['source']} · {location}"):
            st.write(source['text'])
            st.caption(f"块 ID：{source['id']} · 排序分数：{source['score']}")


def show_answer(result):
    st.markdown(result['answer'])
    st.caption(f"{METHODS[result['method']]} · 检索 {result['retrieval_seconds']:.3f} 秒 · "
               f"生成 {result['generation_seconds']:.3f} 秒 · 状态 {result['status']}")
    for warning in result.get('warnings', []):
        st.warning(warning)
    show_sources(result['sources'])


with chat:
    st.subheader('让每个结论有原文可查')
    st.caption('试问：周转筐的重量要计入菜品净重吗？ / 每箱报价缺少什么就不能比较？')
    for result in st.session_state.get('history', []):
        with st.chat_message('user'):
            st.write(result['question'])
        with st.chat_message('assistant'):
            show_answer(result)
    question = st.chat_input('输入采购知识问题；具体菜价请到“菜价工具”查询', disabled=index is None)
    if question:
        with st.chat_message('user'):
            st.write(question)
        with st.chat_message('assistant'):
            try:
                with st.spinner('正在检索文档并整理证据…'):
                    result = answer_question(question, method, use_llm, index)
                show_answer(result)
                st.session_state.setdefault('history', []).append(result)
            except Exception as error:
                st.error(str(error))

with compare:
    st.subheader('同一道问题，比较三种方法')
    with st.form('compare_form'):
        query = st.text_input('比较问题', value='每箱报价没有净重，可以直接比较哪个供应商便宜吗？')
        compare_answers = st.checkbox('同时生成三个回答（会调用模型 3 次）', disabled=not configured)
        run_comparison = st.form_submit_button('开始比较', disabled=index is None)
    if run_comparison:
        results = []
        for name in METHODS:
            try:
                result = index.search(query, name)
                if compare_answers:
                    result['generated'] = generate(query, result['contexts'])
                results.append(result)
            except Exception as error:
                results.append({'method': name, 'error': str(error)})
        st.session_state['comparison'] = results
    if 'comparison' in st.session_state:
        for column, result in zip(st.columns(3), st.session_state['comparison']):
            with column:
                st.markdown(f"**{METHODS[result['method']]}**")
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.caption(f"检索 {result['seconds']:.3f} 秒")
                    if 'generated' in result:
                        st.write(result['generated']['answer'])
                    show_sources(result['contexts'])
    report_path = ROOT / 'reports' / 'retrieval_report.json'
    if report_path.exists():
        with st.expander('查看已保存的检索评测结果'):
            report = json.loads(report_path.read_text(encoding='utf-8'))
            st.dataframe(report['summary'], hide_index=True, width='stretch')
            st.caption('这是自编小样本的检索诊断，不是生成答案正确率，也不代表真实业务效果。')

with tools:
    st.subheader('按品种和日期查询示例菜价')
    st.info('本页通过真正的 MCP stdio 客户端调用独立工具进程；价格来自教学 CSV。')
    rows = read_prices()
    with st.form('prices_form'):
        vegetable = st.selectbox('蔬菜品种', ['', *sorted({r['vegetable'] for r in rows})],
                                 format_func=lambda name: name or '全部品种')
        latest = st.checkbox('使用最新有记录日期', value=True)
        selected_date = st.date_input('指定日期（不勾选“最新”时生效）', value=date.fromisoformat(max(r['date'] for r in rows)))
        run_prices = st.form_submit_button('通过 MCP 查询菜价')
    if run_prices:
        try:
            with st.spinner('连接 MCP 工具进程…'):
                result = call_mcp('get_prices', {'vegetable': vegetable, 'date': '' if latest else selected_date.isoformat()})
            st.session_state['price_result'] = result
        except Exception as error:
            st.error(str(error))
    if 'price_result' in st.session_state:
        result = st.session_state['price_result']
        data = result['data']
        st.write(f"资料日期：{data['as_of'] or '无记录'} · 单位：{data['unit']}")
        st.warning(data['notice'])
        if data['records']:
            st.dataframe(data['records'], hide_index=True, width='stretch')
        with st.expander('查看本次工具调用详情'):
            st.json(result['mcp'])

with library:
    st.subheader('文档加载、读取与建库')
    uploads = st.file_uploader('上传 UTF-8 TXT、Markdown 或文字型 PDF', type=['md', 'txt', 'pdf'], accept_multiple_files=True)
    st.caption('单文件最多 10 MB；同名上传会覆盖用户资料，保存后须重建索引。扫描 PDF 需先做 OCR。')
    if st.button('保存所选资料', disabled=not uploads):
        try:
            for upload in uploads:
                save_upload(upload.name, upload.getvalue())
            st.rerun()
        except Exception as error:
            st.error(str(error))
    if st.button('建立 / 更新索引', type='primary'):
        try:
            with st.spinner('读取文档 → 切分 → 调用 Embedding → 保存向量…'):
                manifest = build_index()
            st.session_state['last_build'] = manifest
            st.rerun()
        except Exception as error:
            st.error(str(error))
    if st.session_state.get('last_build'):
        st.success(f"最近建库：{st.session_state['last_build']['chunks']} 个块，"
                   f"{st.session_state['last_build']['dimensions']} 维真实向量。")
    if paths:
        selected = st.selectbox('选择文档，查看读取和切分结果', paths, format_func=lambda path: path.name)
        try:
            sections = read_document(selected)
            with st.expander('读取结果：原文与章节 / 页码'):
                st.json(sections)
            with st.expander('切分结果：块、字符位置与来源'):
                st.dataframe(split_documents(sections), hide_index=True, width='stretch')
        except Exception as error:
            st.error(str(error))
    user_paths = sorted(DOCUMENT_DIRS[1].glob('*')) if DOCUMENT_DIRS[1].exists() else []
    if user_paths:
        remove = st.selectbox('删除已上传资料（内置样例不在此列）', user_paths, format_func=lambda path: path.name)
        if st.button('删除所选上传资料'):
            remove.unlink()
            st.rerun()
st.caption('青蔬 · 小规模学习项目 / 引用编号校验不等于语义支持验证 / 模型服务可能按用量计费')
