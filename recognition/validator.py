"""证件字段校验及衍生信息提取。

提供身份证号校验、出生日期/性别/年龄提取，以及基于身份证号补全 DocumentInfo。
"""

import re
from datetime import datetime
from typing import Optional
from .models import DocumentInfo

# 18 位身份证号前 17 位加权系数（GB 11643-1999）
_ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
# 校验码对照表：余数 0-10 对应 '10X98765432'
_CHECK_CODES = "10X98765432"

# 18 位身份证号正则：地区码(6) + 出生年月日(8) + 顺序码(3) + 校验码(1)
_ID_PATTERN = re.compile(
    r'^[1-9]\d{5}(18|19|20)\d{2}'
    r'((0[1-9])|(1[0-2]))'
    r'(([0-2][1-9])|10|20|30|31)'
    r'\d{3}[0-9Xx]$'
)


def validate_id_number(id_number: str) -> bool:
    """校验 18 位中国居民身份证号是否符合 GB 11643-1999 标准。

    Args:
        id_number: 待校验的身份证号字符串

    Returns:
        格式正确且校验码通过返回 True，否则 False
    """
    if not id_number or len(id_number) != 18:
        return False
    if not _ID_PATTERN.match(id_number):
        return False

    total = sum(int(id_number[i]) * _ID_WEIGHTS[i] for i in range(17))
    return id_number[17].upper() == _CHECK_CODES[total % 11]


def extract_birth_date(id_number: str) -> Optional[str]:
    """从身份证号中提取出生日期，格式为 YYYY-MM-DD。

    支持 18 位和 15 位身份证号。

    Args:
        id_number: 身份证号

    Returns:
        出生日期字符串，无效时返回 None
    """
    if not id_number:
        return None

    if len(id_number) == 18:
        year, month, day = id_number[6:10], id_number[10:12], id_number[12:14]
    elif len(id_number) == 15:
        year = "19" + id_number[6:8]
        month, day = id_number[8:10], id_number[10:12]
    else:
        return None

    try:
        y, m, d = int(year), int(month), int(day)
        if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
            return f"{year}-{month}-{day}"
    except ValueError:
        pass
    return None


def extract_gender(id_number: str) -> Optional[str]:
    """从 18 位身份证号中提取性别。

    第 17 位（顺序码）：奇数为男，偶数为女。

    Args:
        id_number: 18 位身份证号

    Returns:
        '男' 或 '女'，无效时返回 None
    """
    if id_number and len(id_number) == 18:
        return "男" if int(id_number[16]) % 2 == 1 else "女"
    return None


def extract_age(id_number: str) -> Optional[int]:
    """根据身份证号中的出生日期计算当前年龄（周岁）。

    Args:
        id_number: 身份证号

    Returns:
        周岁年龄，无法计算时返回 None
    """
    birth_str = extract_birth_date(id_number)
    if not birth_str:
        return None
    try:
        birth = datetime.strptime(birth_str, "%Y-%m-%d")
        today = datetime.today()
        return today.year - birth.year - (
            (today.month, today.day) < (birth.month, birth.day)
        )
    except ValueError:
        return None


def enrich_document_info(doc_info: DocumentInfo) -> DocumentInfo:
    """根据身份证号补全 DocumentInfo 中的性别、出生日期、年龄。

    仅当对应字段为空时从身份证号推导并填充。

    Args:
        doc_info: 包含身份证号的证件信息对象

    Returns:
        补全后的 DocumentInfo（原地修改并返回）
    """
    if not doc_info.id_number:
        return doc_info
    if not doc_info.gender:
        doc_info.gender = extract_gender(doc_info.id_number)
    if not doc_info.birth_date:
        doc_info.birth_date = extract_birth_date(doc_info.id_number)
    if doc_info.age is None:
        doc_info.age = extract_age(doc_info.id_number)
    return doc_info
