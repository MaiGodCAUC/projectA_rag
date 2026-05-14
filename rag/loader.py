"""
文档加载器 —— 统一入口

根据文件扩展名分发给对应的解析器，统一返回 ParsedDocument。
支持 PDF、DOCX、Markdown、TXT 四种格式。

使用方式：
    from rag.loader import load_document
    doc = load_document("data/documents/行李运输规定.md")
    print(doc.char_count, doc.table_count)

----------------------------------------------------------------------
## 你需要自己写的部分

本文件由工程辅助生成，但建议你理解后自己手写一遍，因为：
1. 体现了「策略模式」—— 这是面试中可以说"我用了策略模式解耦文档格式"的点
2. LOADER_REGISTRY 字典映射是常见设计，以后项目中可以复用这个思路
3. _compute_hash 的流式读取方式（分块读大文件）是工程细节

没有 TODO(用户) 标记 —— 本文件逻辑完整，你可以照抄学习。
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# hashlib: Python 标准库，用于计算文件 SHA256 哈希值
# 哈希值用于增量索引去重 —— 同一文件内容不变就跳过索引
import hashlib

# Path: pathlib 模块的核心类，提供面向对象的文件路径操作
# 比字符串路径更安全、更易读，支持 .suffix、.name、.exists() 等方法
from pathlib import Path

# ParsedDocument: 统一数据模型，所有 Loader 的输出格式
from rag.models import ParsedDocument

# 导入三个具体的文档解析器类
# PDFLoader: 使用 PyMuPDF + pdfplumber 解析 PDF，亮点是表格感知
from rag.loaders.pdf_loader import PDFLoader

# DocxLoader: 使用 python-docx 解析 Word 文档
from rag.loaders.docx_loader import DocxLoader

# MDLoader: 使用正则表达式解析 Markdown 文件
from rag.loaders.md_loader import MDLoader


# ---------------------------------------------------------------------------
# 扩展名 → 解析器映射
# 新增格式只需注册新的 Loader 类即可，无需修改其他代码（策略模式）
# ---------------------------------------------------------------------------

# LOADER_REGISTRY: 核心设计 —— 文件扩展名到解析器类的映射字典
# 这是「策略模式」的 Python 实现：
#   - key 是文件扩展名（含点号），如 ".pdf"、".docx"
#   - value 是对应的 Loader 类（不是实例），使用时才实例化
# 好处：新增格式只需在这里加一行，load_document 函数不用改
LOADER_REGISTRY = {
    ".pdf": PDFLoader,      # PDF 文件 → PDFLoader 类
    ".docx": DocxLoader,    # Word 文件 → DocxLoader 类
    ".md": MDLoader,        # Markdown 文件 → MDLoader 类
    ".txt": MDLoader,       # 纯文本文件 → 复用 MDLoader（文本和 md 结构类似）
}


def _compute_hash(file_path: str) -> str:
    """计算文件 SHA256 哈希，用于增量索引去重

    为什么用 SHA256 而不用 MD5？
    - SHA256 碰撞概率极低，适合去重场景
    - 虽然比 MD5 慢一点，但文档解析是离线任务，这点差距可忽略

    为什么用流式读取（分块读）？
    - 一次性读入大文件（100MB+ 的 PDF）可能撑爆内存
    - 分块读取每次只读 8KB，内存友好
    - 这是工程细节，面试时可以提"大文件处理做了流式优化"

    Args:
        file_path: 文件路径

    Returns:
        64 位十六进制哈希字符串，如 "a1b2c3d4..."
    """
    # 创建 SHA256 哈希对象，相当于一个"计算器"
    sha = hashlib.sha256()

    # 以二进制读取模式打开文件（"rb"）
    # 用 with 自动管理文件关闭，无论是否异常都会关闭
    with open(file_path, "rb") as f:
        # 流式读取：每次读 8192 字节（8KB），直到文件末尾
        # iter(lambda: f.read(8192), b"") 的含义：
        #   - iter(函数, 哨兵值) 反复调用函数，直到返回值等于哨兵值
        #   - lambda: f.read(8192) 每次读 8KB
        #   - b"" 是哨兵值，读到空字节串说明文件结束
        # 这样写比 while True 循环更 Pythonic
        for chunk in iter(lambda: f.read(8192), b""):
            # 把当前读取的 8KB 数据喂给哈希对象
            # update() 是增量更新，不会覆盖之前的数据
            sha.update(chunk)

    # 返回最终计算出的十六进制哈希字符串
    return sha.hexdigest()


def load_document(file_path: str) -> ParsedDocument:
    """加载并解析单个文档

    自动根据扩展名选择解析器，返回统一格式的 ParsedDocument。
    如果格式不支持则抛出 ValueError。

    这是整个文档加载模块的唯一对外接口。
    调用者不需要知道文件是 PDF 还是 DOCX —— 这是「门面模式」的思想。

    Args:
        file_path: 文档文件路径

    Returns:
        ParsedDocument 对象，含 raw_text + sections + tables + hash

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文档格式
    """
    # 将字符串路径转为 Path 对象，方便后续操作
    # Path 对象有 .exists()、.suffix、.name 等便捷方法
    path = Path(file_path)

    # 防御性检查：文件必须存在，否则报错
    # Path.exists() 返回 bool，检查文件或目录是否存在
    if not path.exists():
        # raise 抛出 FileNotFoundError 异常，调用方可以 try/except 捕获
        raise FileNotFoundError(f"文档不存在: {file_path}")

    # 获取文件扩展名并转为小写
    # path.suffix 返回 '.pdf'、'.md' 等（含点号）
    # .lower() 确保大小写不敏感 —— '.PDF' 和 '.pdf' 等价
    suffix = path.suffix.lower()

    # 检查扩展名是否在注册表中
    # 如果不在，说明是 .xlsx、.pptx 等暂不支持的格式
    if suffix not in LOADER_REGISTRY:
        # 抛出带友好提示的异常，列出当前支持的格式
        raise ValueError(
            f"不支持的文档格式: {suffix}。"
            f"支持: {', '.join(LOADER_REGISTRY.keys())}"
        )

    # 策略模式核心：根据扩展名从注册表取出对应的 Loader 类
    # 注意：这里取到的是「类」不是「实例」
    loader_class = LOADER_REGISTRY[suffix]

    # 实例化 Loader
    # 调用类名() 创建实例，等同于 PDFLoader() 或 MDLoader()
    loader = loader_class()

    # 调用解析器的 parse() 方法，传入文件路径，得到 ParsedDocument 对象
    # 注意：此时 parsed 中的 file_name、file_type、file_hash 还是空/默认值
    parsed = loader.parse(file_path)

    # ---------- 补充元信息 ----------
    # 这些信息是 Loader 不关心的，由统一入口统一补充

    # 设置文件名（从路径中提取，如 '行李运输规定.md'）
    parsed.file_name = path.name

    # 设置文件类型（去掉点号的扩展名，如 'pdf'、'md'、'docx'）
    # lstrip(".") 去掉开头的点号
    parsed.file_type = suffix.lstrip(".")

    # 计算文件哈希并设置
    # 这是增量索引去重的关键数据
    parsed.file_hash = _compute_hash(file_path)

    # 返回组装好的完整 ParsedDocument 对象
    return parsed


def load_documents(file_paths: list[str]) -> list[ParsedDocument]:
    """批量加载文档

    对每个文件路径依次调用 load_document()，返回结果列表。
    使用列表推导式，简洁高效。

    Args:
        file_paths: 文档文件路径列表

    Returns:
        ParsedDocument 列表，顺序与输入一致
    """
    # 列表推导式：等价于
    # result = []
    # for fp in file_paths:
    #     result.append(load_document(fp))
    # return result
    return [load_document(fp) for fp in file_paths]
