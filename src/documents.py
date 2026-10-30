"""发现文件 → 读取页/章节 → 按字符切分；加载与切分分开，便于逐步调试。"""
import hashlib
import re
from pathlib import Path
from pypdf import PdfReader
from settings import CHUNK_SIZE, DOCUMENT_DIRS, OVERLAP


def document_paths(directories=DOCUMENT_DIRS):
    return sorted(p for folder in directories if folder.exists()
                  for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.md', '.txt', '.pdf'))


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_document(path):
    """PDF 保留物理页码；MD 保留标题；扫描 PDF 不静默跳过。"""
    path = Path(path)
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError(f'{path.name} 超过本演示的 10 MB 文件上限。')
    if path.suffix.lower() == '.pdf':
        reader = PdfReader(path)
        if len(reader.pages) > 200:
            raise ValueError('本演示单个 PDF 最多 200 页。')
        sections = [{'source': path.name, 'page': i + 1, 'section': f'第 {i + 1} 页',
                     'text': page.extract_text() or ''} for i, page in enumerate(reader.pages)]
        empty = [s['page'] for s in sections if not s['text'].strip()]
        if empty:
            raise ValueError(f'{path.name} 的 {empty} 页未提取到文字；请检查空页或先做 OCR。')
        return sections
    if path.suffix.lower() not in ('.txt', '.md'):
        raise ValueError('仅支持 UTF-8 TXT、Markdown 和文字型 PDF。')
    raw = path.read_text(encoding='utf-8-sig')
    sections, heading, lines = [], path.stem, []
    for line in raw.splitlines() + ['# __EOF__']:
        if re.match(r'^#{1,6}\s+', line):
            text = '\n'.join(lines).strip()
            if text:
                sections.append({'source': path.name, 'page': None, 'section': heading, 'text': text})
            heading, lines = line.lstrip('# ').strip(), []
        else:
            lines.append(line)
    if not sections:
        raise ValueError(f'{path.name} 没有可读取的文字。')
    return sections


def split_documents(sections, size=CHUNK_SIZE, overlap=OVERLAP):
    if not 0 <= overlap < size:
        raise ValueError('重叠长度必须不小于 0，且小于块长度。')
    chunks = []
    for section in sections:
        text = re.sub(r'\n{3,}', '\n\n', section['text']).strip()
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = max(text.rfind(mark, start + size // 2, end) for mark in ('\n', '。', '；'))
                if boundary >= 0:
                    end = boundary + 1
            body = text[start:end].strip()
            if body:
                identity = f"{section['source']}:{section['page']}:{section['section']}:{start}:{body}"
                chunks.append({**section, 'text': body, 'start': start, 'end': end,
                               'id': 'C' + hashlib.sha256(identity.encode()).hexdigest()[:10]})
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


def save_upload(name, data, directory=DOCUMENT_DIRS[1]):
    name = Path(name.replace('\\', '/')).name
    if not name or name.startswith('.') or Path(name).suffix.lower() not in ('.md', '.txt', '.pdf'):
        raise ValueError('请选择名称正常的 MD、TXT 或 PDF 文件。')
    if not data or len(data) > 10 * 1024 * 1024:
        raise ValueError('文件不能为空，且不得超过 10 MB。')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if any((folder / name).exists() for folder in DOCUMENT_DIRS if folder != directory):
        raise ValueError('该名称与内置资料重复，请先重命名。')
    path.write_bytes(data)
    return path
