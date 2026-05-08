"""
文档加载器 —— 统一入口

根据文件扩展名分发给对应的解析器，统一返回 ParsedDocument。
支持 PDF、DOCX、Markdown、TXT 四种格式。

使用方式：
    from rag.loader import load_document
    doc = load_document("data/documents/行李运输规定.md")
    print(doc.char_count, doc.table_count)
"""

import hashlib
from pathlib import Path

from rag.models import ParsedDocument
from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.docx_loader import DocxLoader
from rag.loaders.md_loader import MDLoader


# ---------------------------------------------------------------------------
# 扩展名 → 解析器映射
# 新增格式只需注册新的 Loader 类即可，无需修改其他代码（策略模式）
# ---------------------------------------------------------------------------
LOADER_REGISTRY = {
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    ".md": MDLoader,
    ".txt": MDLoader,  # 纯文本复用 Markdown 解析器
}


def _compute_hash(file_path: str) -> str:
    """计算文件 SHA256 哈希，用于增量索引去重"""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_document(file_path: str) -> ParsedDocument:
    """加载并解析单个文档

    自动根据扩展名选择解析器，返回统一格式的 ParsedDocument。
    如果格式不支持则抛出 ValueError。

    Args:
        file_path: 文档文件路径

    Returns:
        ParsedDocument 对象，含 raw_text + sections + tables + hash

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文档格式
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文档不存在: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in LOADER_REGISTRY:
        raise ValueError(
            f"不支持的文档格式: {suffix}。"
            f"支持: {', '.join(LOADER_REGISTRY.keys())}"
        )

    loader_class = LOADER_REGISTRY[suffix]
    loader = loader_class()

    # 解析文档
    parsed = loader.parse(file_path)

    # 补充元信息
    parsed.file_name = path.name
    parsed.file_type = suffix.lstrip(".")
    parsed.file_hash = _compute_hash(file_path)

    return parsed


def load_documents(file_paths: list[str]) -> list[ParsedDocument]:
    """批量加载文档"""
    return [load_document(fp) for fp in file_paths]
