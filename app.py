from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR


# =========================================================
# 0. 앱 기본 설정
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="상속 금융비서",
    page_icon="📑",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #666;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 1. 상수
# =========================================================

ALLOWED_CATEGORIES = [
    "예금",
    "대출",
    "자동이체",
    "보험",
    "카드",
    "휴면예금",
    "기타",
]

CATEGORY_ALIASES = {
    "입출금": "예금",
    "입출금계좌": "예금",
    "보통예금": "예금",
    "정기예금": "예금",
    "예적금": "예금",
    "신용대출": "대출",
    "담보대출": "대출",
    "전기요금": "자동이체",
    "통신요금": "자동이체",
    "카드대금": "카드",
    "보험료": "보험",
    "휴면성 예금": "휴면예금",
}

ITEM_CATEGORY_MAP = {
    "입출금계좌": "예금",
    "보통예금": "예금",
    "정기예금": "예금",
    "신용대출": "대출",
    "담보대출": "대출",
    "전기요금": "자동이체",
    "통신요금": "자동이체",
    "카드대금": "카드",
    "보험료": "보험",
    "휴면성 예금": "휴면예금",
}

SEVERITY_SCORE = {
    "low": 10,
    "medium": 25,
    "high": 45,
}

ANOMALY_TYPE_LABELS = {
    "DOCUMENT_FIELD_MISSING": "문서 필드 누락",
    "FIELD_MISSING": "필수 필드 누락",
    "AMOUNT_MISMATCH": "금액 불일치",
    "SOURCE_MISMATCH": "원문 근거 불일치",
    "CATEGORY_MISMATCH": "분류 불일치",
    "DATE_MISSING": "기일 누락",
    "OCR_GARBLED": "OCR 깨짐 가능성",
    "DUPLICATE": "중복 가능성",
    "INVALID_VALUE": "값 형식 오류",
}

FINANCE_CORE_COLUMNS = [
    "category",
    "item_name",
    "institution",
    "amount_text",
    "amount_won",
    "event_type",
    "event_date",
    "status",
    "note",
    "source_text",
    "needs_review",
    "review_reason",
]

FINANCE_DISPLAY_COLUMNS = [
    "category",
    "item_name",
    "institution",
    "amount_text",
    "amount_won",
    "event_type",
    "event_date",
    "status",
    "note",
    "priority",
    "priority_score",
    "priority_reason",
    "days_remaining",
    "needs_review",
    "review_reason",
    "anomaly_score",
    "anomaly_count",
    "source_text",
]


APP_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = APP_DIR / "knowledge_base"
RAG_INDEX_PATH = APP_DIR / "rag_index.json"
RAG_INDEX_VERSION = 1
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
SUPPORTED_KNOWLEDGE_EXTENSIONS = {".txt", ".md", ".pdf"}

DEMO_KNOWLEDGE_FILES = {
    "demo_금융거래조회_결과이해.txt": """문서명: 금융거래 조회결과 이해
출처: 상속 금융비서 데모 지식문서
기준일: 2026-07-24
문서유형: 데모 안내서

# 조회결과의 의미
상속인 금융거래 조회결과는 피상속인의 금융재산과 금융채무가 존재하는지 확인하기 위한 출발점이다. 조회 결과만으로 실제 지급 가능 금액, 최종 대출 잔액, 계약 유지 여부 또는 법률적 결론이 확정되는 것은 아니다.

# 사용자 확인
추출 결과에 사용자 확인 필요 표시가 있거나 OCR 이상치가 남아 있으면 원문 문서와 해당 금융기관을 통해 값을 다시 확인한다. 금액, 기관명, 기일은 자동 보정값을 바로 확정하지 않고 사용자가 검토한다.

# 후속 확인
예금, 대출, 카드, 보험 등 각 항목의 세부 조건은 해당 금융기관의 안내를 확인한다. 이 문서는 데모용 일반 안내이며 실제 처리 기준을 대신하지 않는다.
""",
    "demo_예금과_휴면예금.txt": """문서명: 예금과 휴면예금 확인 안내
출처: 상속 금융비서 데모 지식문서
기준일: 2026-07-24
문서유형: 데모 안내서

# 예금 확인
입출금계좌와 정기예금은 계좌 상태, 잔액, 만기, 해지 또는 지급에 필요한 서류를 금융기관에 확인한다. 정기예금에 만기가 표시된 경우 만기일과 중도해지 가능 여부를 구분하여 확인한다.

# 휴면예금 확인
휴면성 예금이 조회된 경우 지급 신청 가능 여부와 신청 창구를 확인한다. 조회 금액과 실제 지급 금액이 다를 수 있으므로 최종 금액은 처리 기관의 확인을 거친다.

# 주의사항
공동상속인, 대리 신청, 제출서류의 유효기간 등에 따라 필요한 절차가 달라질 수 있다. 이 문서는 데모용 일반 안내이다.
""",
    "demo_대출과_카드채무.txt": """문서명: 대출과 카드채무 확인 안내
출처: 상속 금융비서 데모 지식문서
기준일: 2026-07-24
문서유형: 데모 안내서

# 대출 확인
대출이 조회되면 조회 금액만으로 최종 채무를 단정하지 않는다. 금융기관에 원금, 이자, 연체 여부, 다음 납부일과 상속 관련 처리 창구를 확인한다.

# 카드대금 확인
카드대금은 결제예정액, 결제계좌, 추가 승인 또는 취소 내역을 확인한다. 결제일이 가까운 경우 우선 확인 대상으로 볼 수 있지만 실제 조치 방법은 카드사 또는 금융기관의 안내를 따른다.

# 법률적 판단 제한
상속 승인, 한정승인 또는 포기 여부는 이 시스템이 결정하지 않는다. 채무가 확인되면 관련 전문가 또는 공식 기관을 통해 별도로 검토한다.
""",
    "demo_자동이체와_보험.txt": """문서명: 자동이체와 보험 확인 안내
출처: AI 상속 안심 가이드 데모 지식문서
기준일: 2026-07-24
문서유형: 데모 안내서

# 자동이체 확인
전기요금, 통신요금 등 자동이체는 출금 예정일, 납부계좌, 서비스 유지 필요성을 확인한다. 사망 이후에도 출금될 수 있는 항목이 있으므로 불필요한 출금 여부를 점검한다.

# 보험 확인
보험료가 조회되면 보험 계약의 존재, 계약 상태, 납부일, 보험금 또는 해지 관련 문의 창구를 확인한다. 보험료 항목만으로 보험금 지급 여부를 판단하지 않는다.

# 사용자 검토
자동이체 또는 보험 항목의 기관명과 비고가 다른 금융항목의 내용과 섞여 있으면 원문 행을 다시 확인한다.
""",
    "demo_제출서류.txt": """문서명: 상속 금융업무 제출서류 안내
출처: 상속 금융비서 데모 지식문서
기준일: 2026-07-24
문서유형: 데모 안내서

# 기본 확인서류
데모 문서에는 가족관계증명서, 기본증명서, 사망진단서 또는 제적등본, 상속인 신분증 사본이 필요서류로 제시되어 있다.

# 상황별 서류
공동상속인이 존재하는 경우 공동상속인 동의서를 확인할 수 있고, 대리 신청인 경우 위임장이 필요할 수 있다. 실제 제출서류와 발급 기준은 신청 기관에 확인한다.

# 문서 관리
서류명, 제출 여부, 비고를 구조화하여 관리하고 제출 전 최신 발급본이 필요한지 확인한다. 시스템에 없는 서류를 임의로 필수라고 단정하지 않는다.
""",
    "demo_이상치와_사용자검토.txt": """문서명: OCR 이상치와 사용자 검토 절차
출처: 상속 금융비서 데모 지식문서
기준일: 2026-07-24
문서유형: 데모 안내서

# 이상치 유형
필수값 누락, 금액 불일치, 분류 불일치, 날짜 누락, 중복, OCR 문자 깨짐과 문맥상 모순은 검토 대상으로 표시할 수 있다.

# 검토 원칙
정규화는 날짜와 금액의 표기 형식을 통일하는 작업이다. 원문이 불명확한 기관명, 금액, 기일을 시스템이 자동으로 확정해서는 안 된다. 사용자는 원문 근거와 제안값을 비교한 뒤 확정하거나 이상 아님으로 처리한다.

# 재검증
사용자가 값을 수정하면 이전 탐지 결과는 오래된 결과가 될 수 있다. 수정 데이터에 대해 규칙 기반 탐지와 LLM 의미 탐지를 다시 실행하고, 검토가 끝난 데이터만 우선순위와 상담 근거에 사용한다.
""",
}


DEMO_OCR_TEXT = """[페이지 1]

가상은행 상속지원센터
문서번호: DEMO-INH-2026-0710-001
발급일자: 2026-07-10
1 / 2

상속인 금융거래 조회결과 통지서
OCR 데모용 · 가상 문서

1. 피상속인 기본정보

성명    김민수
생년월일    1968-03-12
조회기준일    2026-07-01
조회신청일    2026-07-10

2. 금융거래 조회 결과

구분    항목    기관    금액(원)    기일/상태    비고
예금    입출금계좌    가상은행    3,200,000    정상    지급 가능 여부 확인 필요
예금    정기예금    가상은행    10,000,000    만기 2026-08-30    해지 가능 여부 확인
대출    신용대출    가상은행    15,000,000    이자 납부일 2026-07-20    상속채무 확인 필요
자동이체    전기요금    가상은행    120,000    예정일 2026-07-15    납부계좌 확인
자동이체    통신요금    가상은행    65,000    예정일 2026-07-18    자동이체 유지 여부 검토

[페이지 2]

가상은행 상속지원센터
문서번호: DEMO-INH-2026-0710-001
발급일자: 2026-07-10
2 / 2

상속인 금융거래 조회결과 통지서
OCR 데모용 · 가상 문서

2. 금융거래 조회 결과 (계속)

구분    항목    기관    금액(원)    기일/상태    비고
카드    카드대금    가상은행    180,000    결제일 2026-07-25    결제계좌 확인 필요
보험    보험료    가상은행    39,000    납부일 2026-07-22    보험 계약 확인 필요
휴면예금    휴면성 예금    가상은행    85,000    조회됨    지급 신청 가능

3. 제출 필요서류

번호    서류명    제출 여부    비고
1    가족관계증명서    필요    최근 발급본 권장
2    기본증명서    필요    피상속인 기준
3    사망진단서 또는 제적등본    필요    확인 서류
4    상속인 신분증 사본    필요    대표 신청인 포함
5    공동상속인 동의서    확인 필요    공동상속인 존재 시
6    위임장    경우에 따라 필요    대리 신청 시

4. 안내 문구

대출이 존재하는 경우 상속채무 여부를 먼저 검토하시기 바랍니다.
자동이체 항목은 사망 이후에도 출금될 수 있으므로 유지 여부를 확인하시기 바랍니다.
"""


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "document": {
            "type": "object",
            "properties": {
                "document_number": {"type": ["string", "null"]},
                "issuer": {"type": ["string", "null"]},
                "issue_date": {"type": ["string", "null"]},
                "document_title": {"type": ["string", "null"]},
            },
            "required": [
                "document_number",
                "issuer",
                "issue_date",
                "document_title",
            ],
            "additionalProperties": False,
        },
        "deceased": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "birth_date": {"type": ["string", "null"]},
                "reference_date": {"type": ["string", "null"]},
                "request_date": {"type": ["string", "null"]},
            },
            "required": [
                "name",
                "birth_date",
                "reference_date",
                "request_date",
            ],
            "additionalProperties": False,
        },
        "financial_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ALLOWED_CATEGORIES,
                    },
                    "item_name": {"type": ["string", "null"]},
                    "institution": {"type": ["string", "null"]},
                    "amount_text": {"type": ["string", "null"]},
                    "amount_won": {"type": ["integer", "null"]},
                    "event_type": {"type": ["string", "null"]},
                    "event_date": {"type": ["string", "null"]},
                    "status": {"type": ["string", "null"]},
                    "note": {"type": ["string", "null"]},
                    "source_text": {"type": "string"},
                    "needs_review": {"type": "boolean"},
                    "review_reason": {"type": ["string", "null"]},
                },
                "required": [
                    "category",
                    "item_name",
                    "institution",
                    "amount_text",
                    "amount_won",
                    "event_type",
                    "event_date",
                    "status",
                    "note",
                    "source_text",
                    "needs_review",
                    "review_reason",
                ],
                "additionalProperties": False,
            },
        },
        "required_documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document_name": {"type": "string"},
                    "submission_status": {"type": ["string", "null"]},
                    "note": {"type": ["string", "null"]},
                    "source_text": {"type": "string"},
                },
                "required": [
                    "document_name",
                    "submission_status",
                    "note",
                    "source_text",
                ],
                "additionalProperties": False,
            },
        },
        "notices": {
            "type": "array",
            "items": {"type": "string"},
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "document",
        "deceased",
        "financial_items",
        "required_documents",
        "notices",
        "warnings",
    ],
    "additionalProperties": False,
}


EXTRACTION_INSTRUCTIONS = """
당신은 한국어 상속 금융조회 문서의 정보 추출 시스템이다.

반드시 다음 규칙을 지켜라.

1. 입력 텍스트에 실제로 존재하는 정보만 사용한다.
2. 입력에 없는 정보는 추측하지 말고 null로 반환한다.
3. 금융거래 표의 한 행을 하나의 financial_items 객체로 만든다.
4. amount_text에는 원문 금액을 그대로 기록한다.
5. amount_won에는 쉼표와 '원'을 제거한 정수를 기록한다.
6. 날짜는 가능한 경우 YYYY-MM-DD 형식으로 변환한다.
7. '만기 2026-08-30'은 event_type='만기',
   event_date='2026-08-30'으로 분리한다.
8. 결제일, 납부일, 예정일, 이자 납부일도 같은 방식으로 분리한다.
9. 날짜가 없는 '정상', '조회됨'은 status에 기록한다.
10. category는 허용된 카테고리 중 하나만 사용한다.
11. source_text에는 근거가 된 원문 행을 그대로 기록한다.
12. 글자가 불완전하거나 행 관계가 불명확하면
    needs_review=true로 표시한다.
13. 법률 판단이나 우선순위 판단은 수행하지 않는다.
14. 제출 필요서류와 일반 안내 문구를 각각 분리한다.
""".strip()


# =========================================================
# 2. 세션 상태
# =========================================================

SESSION_DEFAULTS = {
    "ocr_text_editor": "",
    "ocr_result_info": {},
    "extracted_result": None,
    "normalized_result": None,
    "extracted_finance_df": None,
    "required_documents_df": None,
    "rule_anomalies": [],
    "anomaly_df": None,
    "chat_history": [],
    "rag_last_context": None,
    "rag_index_info": {},
}

for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        if isinstance(default_value, dict):
            st.session_state[key] = {}
        elif isinstance(default_value, list):
            st.session_state[key] = []
        else:
            st.session_state[key] = default_value


# =========================================================
# 3. 공통 유틸리티
# =========================================================

def is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except (TypeError, ValueError):
        pass

    return False


def safe_bool(value: Any) -> bool:
    if is_missing(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
            "예",
            "필요",
        }

    return bool(value)


def clean_text(value: Any) -> str | None:
    if is_missing(value):
        return None

    text = str(value).strip()

    if not text or text.lower() in {"none", "null", "nan"}:
        return None

    return re.sub(r"\s+", " ", text)


def compact_text(value: Any) -> str:
    text = clean_text(value)
    return re.sub(r"\s+", "", text) if text else ""


def normalize_amount(value: Any) -> int | None:
    if is_missing(value):
        return None

    if isinstance(value, (int, np.integer)):
        amount = int(value)
        return amount if amount >= 0 else None

    if isinstance(value, float):
        if np.isnan(value):
            return None
        if value.is_integer():
            amount = int(value)
            return amount if amount >= 0 else None

    number_text = re.sub(r"[^0-9\-]", "", str(value))

    if not number_text:
        return None

    try:
        amount = int(number_text)
    except ValueError:
        return None

    return amount if amount >= 0 else None


def normalize_date(value: Any) -> str | None:
    if is_missing(value):
        return None

    text = clean_text(value)

    if text is None:
        return None

    patterns = [
        r"\b(\d{4})[./\-]\s*(\d{1,2})[./\-]\s*(\d{1,2})\b",
        r"\b(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\b",
    ]

    match = None

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            break

    if not match:
        return None

    year, month, day = map(int, match.groups())

    try:
        parsed = datetime(year=year, month=month, day=day)
    except ValueError:
        return None

    return parsed.strftime("%Y-%m-%d")


def parse_date(value: Any) -> date | None:
    normalized = normalize_date(value)

    if normalized is None:
        return None

    try:
        return datetime.strptime(normalized, "%Y-%m-%d").date()
    except ValueError:
        return None


def normalize_category(category: Any, item_name: Any) -> str:
    category_text = clean_text(category)
    item_text = clean_text(item_name)

    if category_text in ALLOWED_CATEGORIES:
        return category_text

    for candidate in (category_text, item_text):
        if candidate in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[candidate]

    return "기타"


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def safe_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


# =========================================================
# 4. RapidOCR
# =========================================================

def resolve_enum_member(enum_class: Any, *candidates: str) -> Any:
    normalized_candidates = {
        re.sub(r"[^a-z0-9]", "", candidate.lower())
        for candidate in candidates
    }

    for candidate in candidates:
        if hasattr(enum_class, candidate):
            return getattr(enum_class, candidate)

    for member in enum_class:
        member_name = re.sub(
            r"[^a-z0-9]",
            "",
            str(member.name).lower(),
        )
        member_value = re.sub(
            r"[^a-z0-9]",
            "",
            str(member.value).lower(),
        )

        if (
            member_name in normalized_candidates
            or member_value in normalized_candidates
        ):
            return member

    raise RuntimeError(
        f"{enum_class.__name__}에서 {candidates} 설정을 찾지 못했습니다."
    )


@st.cache_resource
def get_rapidocr_engine() -> RapidOCR:
    return RapidOCR(
        params={
            "Rec.engine_type": resolve_enum_member(
                EngineType,
                "ONNXRUNTIME",
                "onnxruntime",
            ),
            "Rec.lang_type": resolve_enum_member(
                LangRec,
                "KOREAN",
                "korean",
            ),
            "Rec.model_type": resolve_enum_member(
                ModelType,
                "MOBILE",
                "mobile",
            ),
            "Rec.ocr_version": resolve_enum_member(
                OCRVersion,
                "PPOCRV5",
                "PP-OCRv5",
                "ppocrv5",
            ),
        }
    )


def pdf_bytes_to_images(
    file_bytes: bytes,
    dpi: int = 300,
    max_pages: int = 10,
) -> list[Image.Image]:
    images: list[Image.Image] = []

    with fitz.open(stream=file_bytes, filetype="pdf") as document:
        page_count = min(len(document), max_pages)

        for page_index in range(page_count):
            page = document[page_index]
            pixmap = page.get_pixmap(
                dpi=dpi,
                colorspace=fitz.csRGB,
                alpha=False,
            )
            images.append(
                Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
            )

    return images


def uploaded_file_to_images(
    file_bytes: bytes,
    filename: str,
    dpi: int,
    max_pages: int,
) -> list[Image.Image]:
    suffix = filename.lower().rsplit(".", maxsplit=1)[-1]

    if suffix == "pdf":
        return pdf_bytes_to_images(
            file_bytes=file_bytes,
            dpi=dpi,
            max_pages=max_pages,
        )

    if suffix in {"png", "jpg", "jpeg", "bmp", "webp"}:
        return [Image.open(BytesIO(file_bytes)).convert("RGB")]

    raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")


def unpack_rapidocr_result(result: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if result is None:
        return items

    if hasattr(result, "txts"):
        boxes = result.boxes
        texts = result.txts
        scores = result.scores

        if boxes is None or texts is None:
            return items

        for box, text, score in zip(boxes, texts, scores):
            items.append(
                {
                    "box": np.asarray(box, dtype=float),
                    "text": str(text).strip(),
                    "score": float(score),
                }
            )

        return items

    raw_result = result[0] if isinstance(result, tuple) else result

    if isinstance(raw_result, list):
        for row in raw_result:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                items.append(
                    {
                        "box": np.asarray(row[0], dtype=float),
                        "text": str(row[1]).strip(),
                        "score": float(row[2]),
                    }
                )

    return items


def sort_ocr_items_into_lines(
    items: list[dict[str, Any]],
) -> list[str]:
    if not items:
        return []

    positioned_items: list[dict[str, Any]] = []

    for item in items:
        box = item["box"]

        if box.size == 0:
            continue

        positioned_items.append(
            {
                **item,
                "x": float(np.min(box[:, 0])),
                "y": float(np.mean(box[:, 1])),
                "height": max(
                    float(np.max(box[:, 1]) - np.min(box[:, 1])),
                    1.0,
                ),
            }
        )

    if not positioned_items:
        return []

    median_height = float(
        np.median([item["height"] for item in positioned_items])
    )
    line_threshold = max(8.0, median_height * 0.65)

    positioned_items.sort(key=lambda item: (item["y"], item["x"]))

    lines: list[dict[str, Any]] = []

    for item in positioned_items:
        selected_line = None

        for line in lines:
            if abs(line["y"] - item["y"]) <= line_threshold:
                selected_line = line
                break

        if selected_line is None:
            lines.append({"y": item["y"], "items": [item]})
        else:
            selected_line["items"].append(item)
            selected_line["y"] = float(
                np.mean([entry["y"] for entry in selected_line["items"]])
            )

    lines.sort(key=lambda line: line["y"])

    text_lines: list[str] = []

    for line in lines:
        line["items"].sort(key=lambda item: item["x"])
        line_text = "    ".join(
            item["text"] for item in line["items"] if item["text"]
        )

        if line_text.strip():
            text_lines.append(line_text.strip())

    return text_lines


def run_rapidocr(
    images: list[Image.Image],
    min_score: float = 0.35,
) -> tuple[str, dict[str, Any]]:
    engine = get_rapidocr_engine()

    page_texts: list[str] = []
    all_scores: list[float] = []
    low_confidence_count = 0
    total_detected_count = 0

    for page_number, image in enumerate(images, start=1):
        result = engine(np.asarray(image.convert("RGB")))
        raw_items = unpack_rapidocr_result(result)

        total_detected_count += len(raw_items)

        for item in raw_items:
            score = float(item["score"])
            all_scores.append(score)

            if score < 0.70:
                low_confidence_count += 1

        filtered_items = [
            item for item in raw_items if item["score"] >= min_score
        ]

        page_text = "\n".join(sort_ocr_items_into_lines(filtered_items))
        page_texts.append(f"[페이지 {page_number}]\n\n{page_text}")

    result_info = {
        "page_count": len(images),
        "detected_text_count": total_detected_count,
        "average_confidence": round(
            float(np.mean(all_scores)) if all_scores else 0.0,
            4,
        ),
        "low_confidence_count": low_confidence_count,
    }

    return "\n\n".join(page_texts), result_info


# =========================================================
# 5. OpenAI LLM 정보 추출
# =========================================================

def resolve_api_key(sidebar_key: str) -> str | None:
    if sidebar_key.strip():
        return sidebar_key.strip()

    environment_key = os.getenv("OPENAI_API_KEY")

    if environment_key:
        return environment_key.strip()

    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")

        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass

    return None


def get_openai_client(api_key: str) -> OpenAI:
    if not api_key:
        raise RuntimeError("OpenAI API 키가 설정되지 않았습니다.")

    return OpenAI(api_key=api_key)


def extract_information_with_llm(
    ocr_text: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    if not ocr_text.strip():
        raise ValueError("OCR 텍스트가 비어 있습니다.")

    if not model.strip():
        raise ValueError("OpenAI 모델명이 비어 있습니다.")

    client = get_openai_client(api_key)

    response = client.responses.create(
        model=model.strip(),
        store=False,
        instructions=EXTRACTION_INSTRUCTIONS,
        input=(
            "다음 OCR 텍스트에서 상속 금융 정보를 추출하세요.\n\n"
            "===== OCR 텍스트 시작 =====\n"
            f"{ocr_text}\n"
            "===== OCR 텍스트 끝 ====="
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "inheritance_financial_extraction",
                "description": "상속 금융조회 문서의 구조화 결과",
                "strict": True,
                "schema": EXTRACTION_SCHEMA,
            }
        },
    )

    if not response.output_text:
        raise RuntimeError("LLM이 추출 결과를 반환하지 않았습니다.")

    try:
        return json.loads(response.output_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "LLM 결과를 JSON으로 해석하지 못했습니다."
        ) from error


# =========================================================
# 6. 정규화
# =========================================================

def normalize_financial_item(
    item: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(item)

    normalized["category"] = normalize_category(
        item.get("category"),
        item.get("item_name"),
    )
    normalized["item_name"] = clean_text(item.get("item_name"))
    normalized["institution"] = clean_text(item.get("institution"))
    normalized["amount_text"] = clean_text(item.get("amount_text"))

    normalized["amount_won"] = normalize_amount(item.get("amount_won"))

    if normalized["amount_won"] is None:
        normalized["amount_won"] = normalize_amount(
            item.get("amount_text")
        )

    normalized["event_type"] = clean_text(item.get("event_type"))
    normalized["event_date"] = normalize_date(item.get("event_date"))
    normalized["status"] = clean_text(item.get("status"))
    normalized["note"] = clean_text(item.get("note"))
    normalized["source_text"] = clean_text(item.get("source_text")) or ""
    normalized["needs_review"] = safe_bool(
        item.get("needs_review", False)
    )
    normalized["review_reason"] = clean_text(
        item.get("review_reason")
    )

    return normalized


def normalize_extraction_result(
    extracted: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(extracted)

    document = dict(normalized.get("document", {}))
    deceased = dict(normalized.get("deceased", {}))

    document["document_number"] = clean_text(
        document.get("document_number")
    )
    document["issuer"] = clean_text(document.get("issuer"))
    document["issue_date"] = normalize_date(document.get("issue_date"))
    document["document_title"] = clean_text(
        document.get("document_title")
    )

    deceased["name"] = clean_text(deceased.get("name"))
    deceased["birth_date"] = normalize_date(deceased.get("birth_date"))
    deceased["reference_date"] = normalize_date(
        deceased.get("reference_date")
    )
    deceased["request_date"] = normalize_date(
        deceased.get("request_date")
    )

    normalized["document"] = document
    normalized["deceased"] = deceased
    normalized["financial_items"] = [
        normalize_financial_item(item)
        for item in normalized.get("financial_items", [])
    ]

    normalized["required_documents"] = [
        {
            "document_name": clean_text(item.get("document_name"))
            or "확인 필요",
            "submission_status": clean_text(
                item.get("submission_status")
            ),
            "note": clean_text(item.get("note")),
            "source_text": clean_text(item.get("source_text")) or "",
        }
        for item in normalized.get("required_documents", [])
    ]

    normalized["notices"] = [
        text
        for text in (
            clean_text(value)
            for value in normalized.get("notices", [])
        )
        if text
    ]

    normalized["warnings"] = [
        text
        for text in (
            clean_text(value)
            for value in normalized.get("warnings", [])
        )
        if text
    ]

    return normalized


# =========================================================
# 7. 규칙 기반 이상 탐지
# =========================================================

def contains_suspicious_characters(value: Any) -> bool:
    text = clean_text(value)

    if text is None:
        return False

    # 중국어·일본어 한자 범위가 있으면 OCR 깨짐 가능성으로 표시
    if re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF]", text):
        return True

    # 한글, 영문, 숫자, 금융 문서에서 흔한 기호만 허용
    remaining = re.sub(
        r"[가-힣A-Za-z0-9\s,./()\-_:·㈜&+]",
        "",
        text,
    )

    return bool(remaining.strip())


def make_anomaly(
    item_index: int | None,
    anomaly_type: str,
    field: str | None,
    severity: str,
    reason: str,
    source_value: Any = None,
    extracted_value: Any = None,
) -> dict[str, Any]:
    base_score = SEVERITY_SCORE.get(severity, 0)

    return {
        "item_index": item_index,
        "anomaly_type": anomaly_type,
        "anomaly_label": ANOMALY_TYPE_LABELS.get(
            anomaly_type,
            anomaly_type,
        ),
        "field": field,
        "severity": severity,
        "reason": reason,
        "source_value": source_value,
        "extracted_value": extracted_value,
        "detector": "rule",
        "confidence": 1.0,
        "anomaly_score": base_score,
        "requires_review": base_score >= 25,
    }


def detect_rule_anomalies(
    normalized: dict[str, Any],
) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []

    document = normalized.get("document", {})
    deceased = normalized.get("deceased", {})

    document_required_fields = {
        "document.document_number": (
            document.get("document_number"),
            "문서번호",
        ),
        "document.issue_date": (
            document.get("issue_date"),
            "발급일자",
        ),
        "deceased.name": (
            deceased.get("name"),
            "피상속인 성명",
        ),
        "deceased.birth_date": (
            deceased.get("birth_date"),
            "피상속인 생년월일",
        ),
    }

    for field_name, (value, label) in document_required_fields.items():
        if clean_text(value) is None:
            anomalies.append(
                make_anomaly(
                    item_index=None,
                    anomaly_type="DOCUMENT_FIELD_MISSING",
                    field=field_name,
                    severity="high",
                    reason=f"{label}이 누락되었습니다.",
                )
            )

    financial_items = normalized.get("financial_items", [])
    seen_keys: dict[tuple[Any, ...], int] = {}

    for index, item in enumerate(financial_items):
        category = clean_text(item.get("category"))
        item_name = clean_text(item.get("item_name"))
        institution = clean_text(item.get("institution"))
        amount_text = clean_text(item.get("amount_text"))
        amount_won = normalize_amount(item.get("amount_won"))
        event_type = clean_text(item.get("event_type"))
        event_date = normalize_date(item.get("event_date"))
        status = clean_text(item.get("status"))
        source_text = clean_text(item.get("source_text")) or ""

        # 1) 필수 필드 누락
        for field_name, value, label in [
            ("item_name", item_name, "금융 항목명"),
            ("institution", institution, "금융기관명"),
            ("amount_won", amount_won, "금액"),
        ]:
            if value is None:
                anomalies.append(
                    make_anomaly(
                        item_index=index,
                        anomaly_type="FIELD_MISSING",
                        field=field_name,
                        severity="high",
                        reason=f"{label}이 누락되었습니다.",
                        source_value=source_text,
                    )
                )

        # 2) 원문 금액 문자열과 정규화 금액 불일치
        amount_from_text = normalize_amount(amount_text)

        if (
            amount_from_text is not None
            and amount_won is not None
            and amount_from_text != amount_won
        ):
            anomalies.append(
                make_anomaly(
                    item_index=index,
                    anomaly_type="AMOUNT_MISMATCH",
                    field="amount_won",
                    severity="high",
                    reason=(
                        "원문 금액 문자열과 정규화된 정수 금액이 "
                        "일치하지 않습니다."
                    ),
                    source_value=amount_text,
                    extracted_value=amount_won,
                )
            )

        # 3) source_text에 항목명·금액이 실제 존재하는지 확인
        if (
            amount_text
            and source_text
            and compact_text(amount_text)
            not in compact_text(source_text)
        ):
            anomalies.append(
                make_anomaly(
                    item_index=index,
                    anomaly_type="SOURCE_MISMATCH",
                    field="amount_text",
                    severity="medium",
                    reason=(
                        "추출된 금액 문자열을 해당 원문 근거에서 "
                        "찾지 못했습니다."
                    ),
                    source_value=source_text,
                    extracted_value=amount_text,
                )
            )

        if (
            item_name
            and source_text
            and compact_text(item_name)
            not in compact_text(source_text)
        ):
            anomalies.append(
                make_anomaly(
                    item_index=index,
                    anomaly_type="SOURCE_MISMATCH",
                    field="item_name",
                    severity="medium",
                    reason=(
                        "추출된 항목명을 해당 원문 근거에서 "
                        "찾지 못했습니다."
                    ),
                    source_value=source_text,
                    extracted_value=item_name,
                )
            )

        # 4) 항목명과 분류 불일치
        expected_category = ITEM_CATEGORY_MAP.get(item_name)

        if (
            expected_category is not None
            and category != expected_category
        ):
            anomalies.append(
                make_anomaly(
                    item_index=index,
                    anomaly_type="CATEGORY_MISMATCH",
                    field="category",
                    severity="high",
                    reason=(
                        f"'{item_name}'의 규칙상 분류는 "
                        f"'{expected_category}'이지만 "
                        f"'{category}'로 분류되었습니다."
                    ),
                    source_value=item_name,
                    extracted_value=category,
                )
            )

        # 5) 기일 종류가 있는데 날짜도 상태도 없음
        if event_type and not event_date and not status:
            anomalies.append(
                make_anomaly(
                    item_index=index,
                    anomaly_type="DATE_MISSING",
                    field="event_date",
                    severity="medium",
                    reason=(
                        f"기일 종류 '{event_type}'가 있으나 "
                        "유효한 날짜가 없습니다."
                    ),
                    source_value=source_text,
                )
            )

        # 6) OCR 깨짐 가능 문자
        for field_name, value in {
            "category": category,
            "item_name": item_name,
            "institution": institution,
        }.items():
            if contains_suspicious_characters(value):
                anomalies.append(
                    make_anomaly(
                        item_index=index,
                        anomaly_type="OCR_GARBLED",
                        field=field_name,
                        severity="medium",
                        reason=(
                            "한글 금융 문서에서 예상하지 않은 문자나 "
                            "한자가 발견되었습니다."
                        ),
                        extracted_value=value,
                    )
                )

        # 7) 음수 금액 등 값 형식 오류
        if amount_won is not None and amount_won < 0:
            anomalies.append(
                make_anomaly(
                    item_index=index,
                    anomaly_type="INVALID_VALUE",
                    field="amount_won",
                    severity="high",
                    reason="금액이 음수로 입력되었습니다.",
                    extracted_value=amount_won,
                )
            )

        # 8) 중복 가능성
        duplicate_key = (
            category,
            item_name,
            institution,
            amount_won,
            event_date,
        )

        if duplicate_key in seen_keys:
            first_index = seen_keys[duplicate_key]

            anomalies.append(
                make_anomaly(
                    item_index=index,
                    anomaly_type="DUPLICATE",
                    field=None,
                    severity="medium",
                    reason=(
                        f"{first_index + 1}번 항목과 동일한 "
                        "금융항목일 가능성이 있습니다."
                    ),
                    extracted_value=str(duplicate_key),
                )
            )
        else:
            seen_keys[duplicate_key] = index

    return anomalies


def apply_rule_anomalies(
    normalized: dict[str, Any],
    anomalies: list[dict[str, Any]],
) -> dict[str, Any]:
    result = dict(normalized)
    items = [
        dict(item)
        for item in result.get("financial_items", [])
    ]

    anomalies_by_index: dict[int, list[dict[str, Any]]] = {}

    for anomaly in anomalies:
        index = anomaly.get("item_index")

        if isinstance(index, int) and 0 <= index < len(items):
            anomalies_by_index.setdefault(index, []).append(anomaly)

    for index, item in enumerate(items):
        item_anomalies = anomalies_by_index.get(index, [])
        review_anomalies = [
            anomaly
            for anomaly in item_anomalies
            if anomaly.get("requires_review")
        ]

        if not review_anomalies:
            item["anomaly_score"] = 0
            item["anomaly_count"] = 0
            continue

        item["needs_review"] = True

        reasons = [
            anomaly["reason"]
            for anomaly in review_anomalies
            if anomaly.get("reason")
        ]

        previous_reason = clean_text(item.get("review_reason"))

        if previous_reason:
            reasons.insert(0, previous_reason)

        item["review_reason"] = " / ".join(
            dict.fromkeys(reasons)
        )
        item["anomaly_score"] = max(
            anomaly.get("anomaly_score", 0)
            for anomaly in review_anomalies
        )
        item["anomaly_count"] = len(review_anomalies)

    result["financial_items"] = items
    result["rule_anomalies"] = anomalies

    return result


# =========================================================
# 8. 우선순위 산정
# =========================================================

def determine_base_date(
    normalized: dict[str, Any],
) -> date:
    possible_dates = [
        normalized.get("document", {}).get("issue_date"),
        normalized.get("deceased", {}).get("request_date"),
        normalized.get("deceased", {}).get("reference_date"),
    ]

    for value in possible_dates:
        parsed = parse_date(value)

        if parsed is not None:
            return parsed

    return date.today()


def calculate_priority(
    item: dict[str, Any],
    base_date: date,
) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []

    category = item.get("category", "기타")
    amount = normalize_amount(item.get("amount_won")) or 0
    event_date = parse_date(item.get("event_date"))
    note = clean_text(item.get("note")) or ""

    if category == "대출":
        score += 40
        reasons.append("상속채무 확인 필요")
    elif category == "자동이체":
        score += 25
        reasons.append("추가 출금 가능성")
    elif category == "카드":
        score += 25
        reasons.append("결제계좌 확인 필요")
    elif category == "보험":
        score += 20
        reasons.append("보험 계약 확인 필요")
    elif category == "휴면예금":
        score += 10
        reasons.append("지급 신청 가능성")
    elif category == "예금":
        score += 10
        reasons.append("지급·해지 절차 확인")

    days_remaining = None

    if event_date is not None:
        days_remaining = (event_date - base_date).days

        if days_remaining < 0:
            score += 40
            reasons.append("기일이 이미 지남")
        elif days_remaining <= 3:
            score += 35
            reasons.append("3일 이내 기일")
        elif days_remaining <= 7:
            score += 30
            reasons.append("7일 이내 기일")
        elif days_remaining <= 30:
            score += 20
            reasons.append("30일 이내 기일")
        else:
            score += 5

    if amount >= 10_000_000:
        score += 15
        reasons.append("금액 1천만 원 이상")
    elif amount >= 1_000_000:
        score += 8
        reasons.append("금액 1백만 원 이상")

    if any(
        keyword in note
        for keyword in ["확인 필요", "상속채무", "검토"]
    ):
        score += 10

    if safe_bool(item.get("needs_review")):
        score += 5
        reasons.append("추출 결과 사용자 확인 필요")

    if score >= 70:
        priority = "긴급"
    elif score >= 45:
        priority = "높음"
    elif score >= 25:
        priority = "보통"
    else:
        priority = "낮음"

    result = dict(item)
    result["priority_score"] = score
    result["priority"] = priority
    result["priority_reason"] = " / ".join(
        dict.fromkeys(reasons)
    )
    result["days_remaining"] = days_remaining

    return result


def add_priorities(
    normalized: dict[str, Any],
) -> dict[str, Any]:
    result = dict(normalized)
    base_date = determine_base_date(result)

    result["financial_items"] = [
        calculate_priority(item, base_date)
        for item in result.get("financial_items", [])
    ]
    result["base_date"] = base_date.isoformat()

    return result


# =========================================================
# 9. DataFrame 변환 및 통합 처리
# =========================================================

def financial_items_to_dataframe(
    normalized: dict[str, Any],
) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        normalized.get("financial_items", [])
    )

    for column in FINANCE_DISPLAY_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = None

    return dataframe[FINANCE_DISPLAY_COLUMNS]


def required_documents_to_dataframe(
    normalized: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "document_name",
        "submission_status",
        "note",
        "source_text",
    ]

    dataframe = pd.DataFrame(
        normalized.get("required_documents", [])
    )

    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = None

    return dataframe[columns]


def anomaly_list_to_dataframe(
    anomalies: list[dict[str, Any]],
) -> pd.DataFrame:
    columns = [
        "item_index",
        "anomaly_label",
        "anomaly_type",
        "field",
        "severity",
        "reason",
        "source_value",
        "extracted_value",
        "confidence",
        "anomaly_score",
        "requires_review",
        "detector",
    ]

    dataframe = pd.DataFrame(anomalies)

    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = None

    return dataframe[columns]


def process_extracted_result(
    extracted: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    normalized = normalize_extraction_result(extracted)
    rule_anomalies = detect_rule_anomalies(normalized)
    normalized = apply_rule_anomalies(normalized, rule_anomalies)
    normalized = add_priorities(normalized)

    finance_df = financial_items_to_dataframe(normalized)
    documents_df = required_documents_to_dataframe(normalized)
    anomaly_df = anomaly_list_to_dataframe(rule_anomalies)

    return (
        normalized,
        rule_anomalies,
        finance_df,
        documents_df,
        anomaly_df,
    )


def dataframe_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    clean_df = dataframe.copy()
    clean_df = clean_df.where(pd.notna(clean_df), None)

    return clean_df.to_dict(orient="records")


def revalidate_edited_finance_data() -> None:
    finance_df = st.session_state.get("extracted_finance_df")
    normalized = st.session_state.get("normalized_result")

    if finance_df is None or normalized is None:
        raise RuntimeError("재검증할 데이터가 없습니다.")

    updated = dict(normalized)

    core_records = []

    for record in dataframe_records(finance_df):
        core_record = {
            column: record.get(column)
            for column in FINANCE_CORE_COLUMNS
        }
        core_records.append(
            normalize_financial_item(core_record)
        )

    updated["financial_items"] = core_records
    updated.pop("rule_anomalies", None)
    updated.pop("base_date", None)

    rule_anomalies = detect_rule_anomalies(updated)
    updated = apply_rule_anomalies(updated, rule_anomalies)
    updated = add_priorities(updated)

    st.session_state["normalized_result"] = updated
    st.session_state["rule_anomalies"] = rule_anomalies
    st.session_state["anomaly_df"] = anomaly_list_to_dataframe(
        rule_anomalies
    )
    st.session_state["extracted_finance_df"] = (
        financial_items_to_dataframe(updated)
    )


# =========================================================
# 10. 데모 구조화 데이터
# =========================================================

def build_demo_extracted_result() -> dict[str, Any]:
    return {
        "document": {
            "document_number": "DEMO-INH-2026-0710-001",
            "issuer": "가상은행 상속지원센터",
            "issue_date": "2026-07-10",
            "document_title": "상속인 금융거래 조회결과 통지서",
        },
        "deceased": {
            "name": "김민수",
            "birth_date": "1968-03-12",
            "reference_date": "2026-07-01",
            "request_date": "2026-07-10",
        },
        "financial_items": [
            {
                "category": "예금",
                "item_name": "입출금계좌",
                "institution": "가상은행",
                "amount_text": "3,200,000",
                "amount_won": 3_200_000,
                "event_type": None,
                "event_date": None,
                "status": "정상",
                "note": "지급 가능 여부 확인 필요",
                "source_text": (
                    "예금 입출금계좌 가상은행 3,200,000 "
                    "정상 지급 가능 여부 확인 필요"
                ),
                "needs_review": False,
                "review_reason": None,
            },
            {
                "category": "예금",
                "item_name": "정기예금",
                "institution": "가상은행",
                "amount_text": "10,000,000",
                "amount_won": 10_000_000,
                "event_type": "만기",
                "event_date": "2026-08-30",
                "status": None,
                "note": "해지 가능 여부 확인",
                "source_text": (
                    "예금 정기예금 가상은행 10,000,000 "
                    "만기 2026-08-30 해지 가능 여부 확인"
                ),
                "needs_review": False,
                "review_reason": None,
            },
            {
                "category": "대출",
                "item_name": "신용대출",
                "institution": "가상은행",
                "amount_text": "15,000,000",
                "amount_won": 15_000_000,
                "event_type": "이자 납부일",
                "event_date": "2026-07-20",
                "status": None,
                "note": "상속채무 확인 필요",
                "source_text": (
                    "대출 신용대출 가상은행 15,000,000 "
                    "이자 납부일 2026-07-20 상속채무 확인 필요"
                ),
                "needs_review": False,
                "review_reason": None,
            },
            {
                "category": "자동이체",
                "item_name": "전기요금",
                "institution": "가상은행",
                "amount_text": "120,000",
                "amount_won": 120_000,
                "event_type": "예정일",
                "event_date": "2026-07-15",
                "status": None,
                "note": "납부계좌 확인",
                "source_text": (
                    "자동이체 전기요금 가상은행 120,000 "
                    "예정일 2026-07-15 납부계좌 확인"
                ),
                "needs_review": False,
                "review_reason": None,
            },
            {
                "category": "자동이체",
                "item_name": "통신요금",
                "institution": "가상은행",
                "amount_text": "65,000",
                "amount_won": 65_000,
                "event_type": "예정일",
                "event_date": "2026-07-18",
                "status": None,
                "note": "자동이체 유지 여부 검토",
                "source_text": (
                    "자동이체 통신요금 가상은행 65,000 "
                    "예정일 2026-07-18 자동이체 유지 여부 검토"
                ),
                "needs_review": False,
                "review_reason": None,
            },
            {
                "category": "카드",
                "item_name": "카드대금",
                "institution": "가상은행",
                "amount_text": "180,000",
                "amount_won": 180_000,
                "event_type": "결제일",
                "event_date": "2026-07-25",
                "status": None,
                "note": "결제계좌 확인 필요",
                "source_text": (
                    "카드 카드대금 가상은행 180,000 "
                    "결제일 2026-07-25 결제계좌 확인 필요"
                ),
                "needs_review": False,
                "review_reason": None,
            },
            {
                "category": "보험",
                "item_name": "보험료",
                "institution": "가상은행",
                "amount_text": "39,000",
                "amount_won": 39_000,
                "event_type": "납부일",
                "event_date": "2026-07-22",
                "status": None,
                "note": "보험 계약 확인 필요",
                "source_text": (
                    "보험 보험료 가상은행 39,000 "
                    "납부일 2026-07-22 보험 계약 확인 필요"
                ),
                "needs_review": False,
                "review_reason": None,
            },
            {
                "category": "휴면예금",
                "item_name": "휴면성 예금",
                "institution": "가상은행",
                "amount_text": "85,000",
                "amount_won": 85_000,
                "event_type": None,
                "event_date": None,
                "status": "조회됨",
                "note": "지급 신청 가능",
                "source_text": (
                    "휴면예금 휴면성 예금 가상은행 85,000 "
                    "조회됨 지급 신청 가능"
                ),
                "needs_review": False,
                "review_reason": None,
            },
        ],
        "required_documents": [
            {
                "document_name": "가족관계증명서",
                "submission_status": "필요",
                "note": "최근 발급본 권장",
                "source_text": (
                    "가족관계증명서 필요 최근 발급본 권장"
                ),
            },
            {
                "document_name": "기본증명서",
                "submission_status": "필요",
                "note": "피상속인 기준",
                "source_text": "기본증명서 필요 피상속인 기준",
            },
            {
                "document_name": "사망진단서 또는 제적등본",
                "submission_status": "필요",
                "note": "확인 서류",
                "source_text": (
                    "사망진단서 또는 제적등본 필요 확인 서류"
                ),
            },
            {
                "document_name": "상속인 신분증 사본",
                "submission_status": "필요",
                "note": "대표 신청인 포함",
                "source_text": (
                    "상속인 신분증 사본 필요 대표 신청인 포함"
                ),
            },
            {
                "document_name": "공동상속인 동의서",
                "submission_status": "확인 필요",
                "note": "공동상속인 존재 시",
                "source_text": (
                    "공동상속인 동의서 확인 필요 공동상속인 존재 시"
                ),
            },
            {
                "document_name": "위임장",
                "submission_status": "경우에 따라 필요",
                "note": "대리 신청 시",
                "source_text": (
                    "위임장 경우에 따라 필요 대리 신청 시"
                ),
            },
        ],
        "notices": [
            (
                "대출이 존재하는 경우 상속채무 여부를 먼저 "
                "검토하시기 바랍니다."
            ),
            (
                "자동이체 항목은 사망 이후에도 출금될 수 있으므로 "
                "유지 여부를 확인하시기 바랍니다."
            ),
        ],
        "warnings": [],
    }


def load_structured_result(
    extracted: dict[str, Any],
) -> None:
    (
        normalized,
        rule_anomalies,
        finance_df,
        documents_df,
        anomaly_df,
    ) = process_extracted_result(extracted)

    st.session_state["extracted_result"] = extracted
    st.session_state["normalized_result"] = normalized
    st.session_state["rule_anomalies"] = rule_anomalies
    st.session_state["extracted_finance_df"] = finance_df
    st.session_state["required_documents_df"] = documents_df
    st.session_state["anomaly_df"] = anomaly_df


# =========================================================
# 11. 임베딩 기반 RAG 및 상담
# =========================================================

def tokenize_korean_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[가-힣A-Za-z0-9]+",
            str(text).lower(),
        )
        if len(token) >= 2
    }


def ensure_default_knowledge_base() -> None:
    """처음 실행할 때 데모용 지식문서를 생성한다."""
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    supported_files = [
        path
        for path in KNOWLEDGE_BASE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_KNOWLEDGE_EXTENSIONS
    ]

    if supported_files:
        return

    for filename, content in DEMO_KNOWLEDGE_FILES.items():
        target = KNOWLEDGE_BASE_DIR / filename
        target.write_text(content.strip() + "\n", encoding="utf-8")


def safe_knowledge_filename(filename: str) -> str:
    basename = Path(filename).name
    basename = re.sub(r"[^가-힣A-Za-z0-9._()\- ]", "_", basename)
    return basename or "knowledge_document.txt"


def save_uploaded_knowledge_files(uploaded_files: list[Any]) -> list[str]:
    ensure_default_knowledge_base()
    saved: list[str] = []

    for uploaded in uploaded_files:
        filename = safe_knowledge_filename(uploaded.name)
        suffix = Path(filename).suffix.lower()

        if suffix not in SUPPORTED_KNOWLEDGE_EXTENSIONS:
            continue

        target = KNOWLEDGE_BASE_DIR / filename
        target.write_bytes(uploaded.getvalue())
        saved.append(filename)

    return saved


def list_knowledge_files(
    include_demo: bool = True,
) -> list[Path]:
    ensure_default_knowledge_base()

    files = [
        path
        for path in KNOWLEDGE_BASE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_KNOWLEDGE_EXTENSIONS
    ]

    if not include_demo:
        files = [
            path
            for path in files
            if not path.name.lower().startswith("demo_")
        ]

    return sorted(files, key=lambda item: str(item).lower())


def knowledge_base_fingerprint(
    include_demo: bool = True,
) -> str:
    digest = hashlib.sha256()

    for file_path in list_knowledge_files(include_demo=include_demo):
        relative = file_path.relative_to(KNOWLEDGE_BASE_DIR)
        digest.update(str(relative).encode("utf-8"))
        digest.update(file_path.read_bytes())

    digest.update(str(include_demo).encode("utf-8"))
    return digest.hexdigest()


def parse_knowledge_metadata(
    text: str,
    default_title: str,
) -> dict[str, str | None]:
    metadata: dict[str, str | None] = {
        "title": default_title,
        "source": None,
        "effective_date": None,
        "document_type": None,
    }

    patterns = {
        "title": r"^\s*문서명\s*:\s*(.+?)\s*$",
        "source": r"^\s*출처\s*:\s*(.+?)\s*$",
        "effective_date": r"^\s*기준일\s*:\s*(.+?)\s*$",
        "document_type": r"^\s*문서유형\s*:\s*(.+?)\s*$",
    }

    for line in text.splitlines()[:20]:
        for key, pattern in patterns.items():
            match = re.match(pattern, line)
            if match:
                metadata[key] = match.group(1).strip()

    return metadata


def read_knowledge_documents(
    include_demo: bool = True,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    for file_path in list_knowledge_files(include_demo=include_demo):
        relative_path = str(file_path.relative_to(KNOWLEDGE_BASE_DIR))
        suffix = file_path.suffix.lower()

        if suffix in {".txt", ".md"}:
            content = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            metadata = parse_knowledge_metadata(
                content,
                default_title=file_path.stem,
            )
            documents.append(
                {
                    "text": content,
                    "metadata": {
                        **metadata,
                        "filename": file_path.name,
                        "relative_path": relative_path,
                        "page": None,
                        "is_demo": file_path.name.lower().startswith(
                            "demo_"
                        ),
                    },
                }
            )
            continue

        if suffix == ".pdf":
            with fitz.open(file_path) as pdf:
                full_text = "\n".join(
                    page.get_text("text") for page in pdf
                )
                common_metadata = parse_knowledge_metadata(
                    full_text,
                    default_title=file_path.stem,
                )

                for page_number, page in enumerate(pdf, start=1):
                    page_text = page.get_text("text").strip()
                    if not page_text:
                        continue

                    documents.append(
                        {
                            "text": page_text,
                            "metadata": {
                                **common_metadata,
                                "filename": file_path.name,
                                "relative_path": relative_path,
                                "page": page_number,
                                "is_demo": False,
                            },
                        }
                    )

    return documents


def split_knowledge_text(
    text: str,
    max_chars: int = 650,
    overlap_chars: int = 100,
) -> list[dict[str, str | None]]:
    """문단과 제목을 우선 보존하면서 지식문서를 청크로 나눈다."""
    clean = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", clean)
        if paragraph.strip()
    ]

    chunks: list[dict[str, str | None]] = []
    current_parts: list[str] = []
    current_length = 0
    current_section: str | None = None

    def flush() -> None:
        nonlocal current_parts, current_length
        if not current_parts:
            return

        chunk_text = "\n\n".join(current_parts).strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "section": current_section,
                }
            )

        if overlap_chars > 0 and chunk_text:
            tail = chunk_text[-overlap_chars:].strip()
            current_parts = [tail] if tail else []
            current_length = len(tail)
        else:
            current_parts = []
            current_length = 0

    for paragraph in paragraphs:
        heading_match = re.match(
            r"^(?:#{1,6}\s+|\d+[.)]\s*)(.+)$",
            paragraph,
        )
        if heading_match and len(paragraph) <= 120:
            current_section = heading_match.group(1).strip()

        if len(paragraph) <= max_chars:
            additional = len(paragraph) + (2 if current_parts else 0)
            if current_parts and current_length + additional > max_chars:
                flush()

            current_parts.append(paragraph)
            current_length += additional
            continue

        # 매우 긴 문단은 문장 단위로 다시 분할한다.
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?다요])\s+", paragraph)
            if sentence.strip()
        ] or [paragraph]

        for sentence in sentences:
            if len(sentence) > max_chars:
                for start in range(0, len(sentence), max_chars - overlap_chars):
                    piece = sentence[start : start + max_chars].strip()
                    if piece:
                        if current_parts:
                            flush()
                        chunks.append(
                            {
                                "text": piece,
                                "section": current_section,
                            }
                        )
                continue

            additional = len(sentence) + (1 if current_parts else 0)
            if current_parts and current_length + additional > max_chars:
                flush()

            current_parts.append(sentence)
            current_length += additional

    flush()

    # 완전히 동일한 청크가 중복 생성되는 것을 제거한다.
    deduplicated: list[dict[str, str | None]] = []
    seen: set[str] = set()

    for chunk in chunks:
        compact = re.sub(r"\s+", " ", chunk["text"] or "").strip()
        if not compact or compact in seen:
            continue
        seen.add(compact)
        deduplicated.append(chunk)

    return deduplicated


def build_knowledge_chunks(
    include_demo: bool = True,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for document in read_knowledge_documents(include_demo=include_demo):
        metadata = document["metadata"]
        split_chunks = split_knowledge_text(document["text"])

        for chunk_index, chunk in enumerate(split_chunks, start=1):
            raw_id = (
                f"{metadata.get('relative_path')}|"
                f"{metadata.get('page')}|{chunk_index}|{chunk['text']}"
            )
            chunk_id = hashlib.sha256(
                raw_id.encode("utf-8")
            ).hexdigest()[:16]

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "metadata": {
                        **metadata,
                        "section": chunk.get("section"),
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks


def create_embeddings(
    texts: list[str],
    api_key: str,
    embedding_model: str,
    batch_size: int = 64,
) -> list[list[float]]:
    if not texts:
        return []

    if not embedding_model.strip():
        raise ValueError("임베딩 모델명이 비어 있습니다.")

    client = get_openai_client(api_key)
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = [
            text.replace("\n", " ").strip()
            for text in texts[start : start + batch_size]
        ]
        response = client.embeddings.create(
            model=embedding_model.strip(),
            input=batch,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        embeddings.extend(
            [list(map(float, item.embedding)) for item in ordered]
        )

    if len(embeddings) != len(texts):
        raise RuntimeError("임베딩 결과 개수가 입력 청크 수와 다릅니다.")

    return embeddings


def build_rag_index(
    api_key: str,
    embedding_model: str,
    include_demo: bool = True,
) -> dict[str, Any]:
    chunks = build_knowledge_chunks(include_demo=include_demo)

    if not chunks:
        raise RuntimeError(
            "knowledge_base 폴더에서 인덱싱할 문서를 찾지 못했습니다."
        )

    vectors = create_embeddings(
        texts=[chunk["text"] for chunk in chunks],
        api_key=api_key,
        embedding_model=embedding_model,
    )

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    index = {
        "version": RAG_INDEX_VERSION,
        "embedding_model": embedding_model.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": knowledge_base_fingerprint(
            include_demo=include_demo
        ),
        "include_demo": include_demo,
        "document_count": len(
            read_knowledge_documents(include_demo=include_demo)
        ),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    temporary_path = RAG_INDEX_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(index, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(RAG_INDEX_PATH)

    return index


def load_rag_index() -> dict[str, Any] | None:
    if not RAG_INDEX_PATH.exists():
        return None

    try:
        index = json.loads(RAG_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if index.get("version") != RAG_INDEX_VERSION:
        return None

    if not isinstance(index.get("chunks"), list):
        return None

    return index


def rag_index_is_current(
    index: dict[str, Any] | None,
    embedding_model: str,
    include_demo: bool,
) -> bool:
    if not index:
        return False

    return (
        index.get("embedding_model") == embedding_model.strip()
        and bool(index.get("include_demo")) == bool(include_demo)
        and index.get("fingerprint")
        == knowledge_base_fingerprint(include_demo=include_demo)
    )


def ensure_rag_index(
    api_key: str,
    embedding_model: str,
    include_demo: bool,
) -> tuple[dict[str, Any], bool]:
    index = load_rag_index()

    if rag_index_is_current(
        index=index,
        embedding_model=embedding_model,
        include_demo=include_demo,
    ):
        return index, False

    return (
        build_rag_index(
            api_key=api_key,
            embedding_model=embedding_model,
            include_demo=include_demo,
        ),
        True,
    )


def cosine_similarity(
    left: list[float] | np.ndarray,
    right: list[float] | np.ndarray,
) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)

    if left_array.shape != right_array.shape:
        return 0.0

    denominator = float(
        np.linalg.norm(left_array) * np.linalg.norm(right_array)
    )
    if denominator == 0.0:
        return 0.0

    return float(np.dot(left_array, right_array) / denominator)


def lexical_similarity(question: str, text: str) -> float:
    question_tokens = tokenize_korean_text(question)
    text_tokens = tokenize_korean_text(text)

    if not question_tokens or not text_tokens:
        return 0.0

    overlap = len(question_tokens & text_tokens)
    return overlap / max(len(question_tokens), 1)


def retrieve_knowledge_evidence(
    question: str,
    api_key: str,
    embedding_model: str,
    include_demo: bool,
    top_k: int = 4,
    minimum_score: float = 0.20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index, rebuilt = ensure_rag_index(
        api_key=api_key,
        embedding_model=embedding_model,
        include_demo=include_demo,
    )

    query_vector = create_embeddings(
        texts=[question],
        api_key=api_key,
        embedding_model=embedding_model,
    )[0]

    scored: list[dict[str, Any]] = []

    for chunk in index.get("chunks", []):
        vector_score = cosine_similarity(
            query_vector,
            chunk.get("embedding", []),
        )
        keyword_score = lexical_similarity(question, chunk.get("text", ""))
        hybrid_score = 0.90 * vector_score + 0.10 * keyword_score

        scored.append(
            {
                "kind": "knowledge",
                "title": chunk.get("metadata", {}).get("title")
                or chunk.get("metadata", {}).get("filename")
                or "지식문서",
                "content": chunk.get("text", ""),
                "score": round(float(hybrid_score), 4),
                "vector_score": round(float(vector_score), 4),
                "keyword_score": round(float(keyword_score), 4),
                "chunk_id": chunk.get("chunk_id"),
                "metadata": chunk.get("metadata", {}),
            }
        )

    scored.sort(
        key=lambda item: (
            item["score"],
            item["vector_score"],
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    for item in scored:
        if item["score"] < minimum_score:
            continue

        compact = re.sub(r"\s+", " ", item["content"]).strip()
        if compact in seen_texts:
            continue

        seen_texts.add(compact)
        selected.append(item)

        if len(selected) >= top_k:
            break

    info = {
        "rebuilt": rebuilt,
        "created_at": index.get("created_at"),
        "embedding_model": index.get("embedding_model"),
        "document_count": index.get("document_count", 0),
        "chunk_count": index.get("chunk_count", 0),
    }

    return selected, info


def build_personal_evidence_records(
    normalized: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not normalized:
        return []

    evidence: list[dict[str, Any]] = []
    deceased = normalized.get("deceased", {})

    evidence.append(
        {
            "kind": "personal",
            "title": "피상속인 기본정보",
            "content": (
                f"성명={deceased.get('name')}, "
                f"생년월일={deceased.get('birth_date')}, "
                f"조회기준일={deceased.get('reference_date')}, "
                f"조회신청일={deceased.get('request_date')}"
            ),
            "metadata": {"record_type": "deceased"},
        }
    )

    for index, item in enumerate(
        normalized.get("financial_items", [])
    ):
        evidence.append(
            {
                "kind": "personal",
                "title": (
                    f"금융항목 {index + 1}: "
                    f"{item.get('item_name') or '항목명 없음'}"
                ),
                "content": (
                    f"분류={item.get('category')}, "
                    f"항목={item.get('item_name')}, "
                    f"기관={item.get('institution')}, "
                    f"금액={item.get('amount_won')}원, "
                    f"기일종류={item.get('event_type')}, "
                    f"기일={item.get('event_date')}, "
                    f"상태={item.get('status')}, "
                    f"비고={item.get('note')}, "
                    f"우선순위={item.get('priority')}, "
                    f"검토필요={item.get('needs_review')}, "
                    f"검토사유={item.get('review_reason')}"
                ),
                "metadata": {
                    "record_type": "financial_item",
                    "item_index": index,
                    "category": item.get("category"),
                    "needs_review": safe_bool(
                        item.get("needs_review")
                    ),
                },
            }
        )

    for index, item in enumerate(
        normalized.get("required_documents", [])
    ):
        evidence.append(
            {
                "kind": "personal",
                "title": f"필요서류 {index + 1}: {item.get('document_name')}",
                "content": (
                    f"서류명={item.get('document_name')}, "
                    f"제출여부={item.get('submission_status')}, "
                    f"비고={item.get('note')}"
                ),
                "metadata": {
                    "record_type": "required_document",
                    "document_index": index,
                },
            }
        )

    for index, notice in enumerate(normalized.get("notices", [])):
        evidence.append(
            {
                "kind": "personal",
                "title": f"문서 안내사항 {index + 1}",
                "content": str(notice),
                "metadata": {
                    "record_type": "notice",
                    "notice_index": index,
                },
            }
        )

    return evidence


def retrieve_personal_evidence(
    question: str,
    normalized: dict[str, Any] | None,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    records = build_personal_evidence_records(normalized)
    if not records:
        return []

    question_tokens = tokenize_korean_text(question)
    urgent_question = any(
        keyword in question for keyword in ["우선", "먼저", "급", "오늘"]
    )

    scored: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        record_text = f"{record['title']} {record['content']}"
        overlap = len(
            question_tokens & tokenize_korean_text(record_text)
        )
        bonus = 0.0

        if urgent_question and (
            "우선순위=긴급" in record_text
            or "우선순위=높음" in record_text
        ):
            bonus += 4.0

        category = record.get("metadata", {}).get("category")
        if category and category in question:
            bonus += 3.0

        if "서류" in question and record.get("metadata", {}).get(
            "record_type"
        ) == "required_document":
            bonus += 3.0

        if any(word in question for word in ["이상", "오류", "확인"]):
            if record.get("metadata", {}).get("needs_review"):
                bonus += 2.5

        score = float(overlap) + bonus
        scored.append(
            {
                **record,
                "score": round(score, 4),
                "source_order": index,
            }
        )

    scored.sort(
        key=lambda item: (item["score"], -item["source_order"]),
        reverse=True,
    )

    selected = [item for item in scored[:top_k] if item["score"] > 0]

    if selected:
        return selected

    # 질문과 직접 겹치는 단어가 없으면 긴급·높음 항목을 우선 제공한다.
    priority_records = [
        item
        for item in scored
        if "우선순위=긴급" in item["content"]
        or "우선순위=높음" in item["content"]
    ]
    return (priority_records or scored)[:top_k]


def classify_rag_question(
    question: str,
    has_personal_data: bool,
) -> str:
    personal_markers = [
        "내 ",
        "제 ",
        "우리 ",
        "조회된",
        "현재 내역",
        "내역",
        "금액",
        "몇 원",
        "언제",
        "어느 기관",
        "어느 은행",
        "먼저 확인",
        "우선순위",
        "오늘 할 일",
        "이상치",
        "검토 필요",
    ]
    knowledge_markers = [
        "절차",
        "방법",
        "어떻게",
        "필요서류",
        "준비서류",
        "주의사항",
        "신청",
        "처리",
        "의미",
        "왜",
        "무엇인가",
        "일반적으로",
        "상속채무",
        "자동이체를",
        "보험을",
    ]

    lowered = f" {question.lower()} "
    personal = any(marker in lowered for marker in personal_markers)
    knowledge = any(marker in lowered for marker in knowledge_markers)

    if personal and knowledge:
        return "MIXED"
    if personal:
        return "PERSONAL"
    if knowledge:
        return "KNOWLEDGE"
    return "MIXED" if has_personal_data else "KNOWLEDGE"


def retrieve_rag_context(
    question: str,
    normalized: dict[str, Any] | None,
    api_key: str,
    embedding_model: str,
    include_demo: bool,
    personal_top_k: int,
    knowledge_top_k: int,
    minimum_score: float,
) -> dict[str, Any]:
    route = classify_rag_question(
        question=question,
        has_personal_data=bool(normalized),
    )

    personal_evidence: list[dict[str, Any]] = []
    knowledge_evidence: list[dict[str, Any]] = []
    index_info: dict[str, Any] = {}

    if route in {"PERSONAL", "MIXED"}:
        personal_evidence = retrieve_personal_evidence(
            question=question,
            normalized=normalized,
            top_k=personal_top_k,
        )

    if route in {"KNOWLEDGE", "MIXED"}:
        knowledge_evidence, index_info = retrieve_knowledge_evidence(
            question=question,
            api_key=api_key,
            embedding_model=embedding_model,
            include_demo=include_demo,
            top_k=knowledge_top_k,
            minimum_score=minimum_score,
        )

    return {
        "route": route,
        "personal_evidence": personal_evidence,
        "knowledge_evidence": knowledge_evidence,
        "index_info": index_info,
    }


def format_rag_prompt_context(rag_context: dict[str, Any]) -> str:
    sections: list[str] = []

    personal_lines = []
    for index, item in enumerate(
        rag_context.get("personal_evidence", []),
        start=1,
    ):
        personal_lines.append(
            f"[개인 {index}] {item['title']}\n{item['content']}"
        )

    if personal_lines:
        sections.append(
            "[사용자 개인 금융정보]\n" + "\n\n".join(personal_lines)
        )

    knowledge_lines = []
    for index, item in enumerate(
        rag_context.get("knowledge_evidence", []),
        start=1,
    ):
        metadata = item.get("metadata", {})
        location_parts = []
        if metadata.get("section"):
            location_parts.append(f"섹션={metadata['section']}")
        if metadata.get("page"):
            location_parts.append(f"페이지={metadata['page']}")
        location = ", ".join(location_parts) or "위치정보 없음"

        knowledge_lines.append(
            f"[문서 {index}] {item['title']} ({location})\n"
            f"{item['content']}"
        )

    if knowledge_lines:
        sections.append(
            "[상속업무 지식문서]\n" + "\n\n".join(knowledge_lines)
        )

    return "\n\n".join(sections)


def answer_with_rag(
    question: str,
    normalized: dict[str, Any] | None,
    api_key: str,
    model: str,
    embedding_model: str,
    include_demo: bool,
    personal_top_k: int,
    knowledge_top_k: int,
    minimum_score: float,
) -> tuple[str, dict[str, Any]]:
    rag_context = retrieve_rag_context(
        question=question,
        normalized=normalized,
        api_key=api_key,
        embedding_model=embedding_model,
        include_demo=include_demo,
        personal_top_k=personal_top_k,
        knowledge_top_k=knowledge_top_k,
        minimum_score=minimum_score,
    )

    evidence_context = format_rag_prompt_context(rag_context)

    if not evidence_context.strip():
        return (
            "검색된 근거가 없어 이 질문에 답할 수 없습니다. "
            "개인 금융정보를 먼저 추출하거나 관련 지식문서를 "
            "knowledge_base 폴더에 추가한 뒤 인덱스를 갱신하세요.",
            rag_context,
        )

    response = get_openai_client(api_key).responses.create(
        model=model.strip(),
        store=False,
        instructions=(
            "당신은 상속 금융업무 안내 보조 AI다. "
            "제공된 근거에 포함된 사실만 사용한다. "
            "개인 금융정보를 근거로 한 문장에는 [개인 1] 형식의 "
            "인용을 붙이고, 지식문서를 근거로 한 문장에는 "
            "[문서 1] 형식의 인용을 붙인다. "
            "검색 근거가 부족하면 추측하지 말고 확인할 수 없다고 "
            "명시한다. 사용자 확인이 끝나지 않은 항목은 미확정 "
            "정보라고 표시한다. 상속 승인, 포기, 한정승인 또는 "
            "법률적 권리를 단정하지 않는다. 답변은 한국어로 한다."
        ),
        input=(
            f"질문 분류: {rag_context['route']}\n"
            f"사용자 질문:\n{question}\n\n"
            f"검색된 근거:\n{evidence_context}"
        ),
        max_output_tokens=1200,
    )

    answer = (
        response.output_text.strip()
        if response.output_text
        else "답변을 생성하지 못했습니다."
    )

    return answer, rag_context


def flatten_rag_evidence(
    rag_context: dict[str, Any],
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []

    for index, item in enumerate(
        rag_context.get("personal_evidence", []),
        start=1,
    ):
        flattened.append({**item, "citation": f"개인 {index}"})

    for index, item in enumerate(
        rag_context.get("knowledge_evidence", []),
        start=1,
    ):
        flattened.append({**item, "citation": f"문서 {index}"})

    return flattened


def render_rag_evidence(rag_context: dict[str, Any]) -> None:
    route_labels = {
        "PERSONAL": "개인 금융정보 검색",
        "KNOWLEDGE": "지식문서 벡터 검색",
        "MIXED": "개인정보 + 지식문서 혼합 검색",
    }
    route = rag_context.get("route", "-")
    st.caption(f"질문 라우팅: {route_labels.get(route, route)}")

    flattened = flatten_rag_evidence(rag_context)
    if not flattened:
        st.info("사용된 근거가 없습니다.")
        return

    for item in flattened:
        citation = item.get("citation", "근거")
        title = item.get("title", "근거")
        score = item.get("score")
        metadata = item.get("metadata", {})

        heading = f"**[{citation}] {title}**"
        if item.get("kind") == "knowledge" and score is not None:
            heading += f" · 검색점수 {float(score):.3f}"

        st.markdown(heading)
        st.write(item.get("content", ""))

        if item.get("kind") == "knowledge":
            source_parts = [
                metadata.get("source"),
                metadata.get("filename"),
            ]
            if metadata.get("section"):
                source_parts.append(f"섹션: {metadata['section']}")
            if metadata.get("page"):
                source_parts.append(f"페이지: {metadata['page']}")
            source_text = " · ".join(
                str(part) for part in source_parts if part
            )
            if source_text:
                st.caption(source_text)


def rag_index_status(
    embedding_model: str,
    include_demo: bool,
) -> dict[str, Any]:
    ensure_default_knowledge_base()
    index = load_rag_index()
    current = rag_index_is_current(
        index=index,
        embedding_model=embedding_model,
        include_demo=include_demo,
    )

    return {
        "exists": index is not None,
        "current": current,
        "document_count": index.get("document_count", 0) if index else 0,
        "chunk_count": index.get("chunk_count", 0) if index else 0,
        "created_at": index.get("created_at") if index else None,
        "embedding_model": index.get("embedding_model") if index else None,
        "knowledge_file_count": len(
            list_knowledge_files(include_demo=include_demo)
        ),
    }


ensure_default_knowledge_base()


# =========================================================
# 12. 사이드바
# =========================================================

with st.sidebar:
    st.header("⚙️ 실행 설정")

    api_key_input = st.text_input(
        "OpenAI API 키",
        type="password",
        help=(
            ".env에 OPENAI_API_KEY가 있으면 비워 두어도 됩니다. "
            "RapidOCR와 규칙 검증에는 API 키가 필요하지 않습니다."
        ),
    )

    api_key = resolve_api_key(api_key_input)

    if api_key:
        st.success("OpenAI API 키가 설정되었습니다.")
    else:
        st.info(
            "LLM 정보 추출과 상담 기능에는 API 키가 필요합니다."
        )

    model_name = st.text_input(
        "OpenAI 모델",
        value=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="계정에서 사용할 수 있는 모델명을 입력하세요.",
    )

    st.divider()
    st.subheader("RapidOCR 설정")

    render_dpi = st.select_slider(
        "PDF 변환 해상도",
        options=[200, 250, 300, 350, 400],
        value=300,
    )

    max_pages = st.number_input(
        "최대 처리 페이지",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
    )

    minimum_ocr_score = st.slider(
        "최소 OCR 신뢰도",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
    )

    st.divider()
    st.subheader("RAG 설정")

    embedding_model_name = st.text_input(
        "임베딩 모델",
        value=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        ),
        help="지식문서와 질문을 벡터로 변환할 모델입니다.",
    )

    include_demo_knowledge = st.checkbox(
        "데모 지식문서 포함",
        value=True,
    )

    rag_knowledge_top_k = st.slider(
        "지식문서 검색 개수",
        min_value=1,
        max_value=8,
        value=4,
    )

    rag_personal_top_k = st.slider(
        "개인정보 검색 개수",
        min_value=1,
        max_value=8,
        value=4,
    )

    rag_minimum_score = st.slider(
        "최소 벡터 검색점수",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        help=(
            "너무 높이면 근거가 검색되지 않을 수 있습니다. "
            "데모에서는 0.15~0.30 정도가 적절합니다."
        ),
    )

    current_rag_status = rag_index_status(
        embedding_model=embedding_model_name,
        include_demo=include_demo_knowledge,
    )

    if current_rag_status["current"]:
        st.success(
            "RAG 인덱스 사용 가능 · "
            f"문서 {current_rag_status['document_count']}개 / "
            f"청크 {current_rag_status['chunk_count']}개"
        )
    elif current_rag_status["exists"]:
        st.warning(
            "지식문서 또는 임베딩 설정이 변경되어 "
            "인덱스 갱신이 필요합니다."
        )
    else:
        st.info("아직 RAG 인덱스가 생성되지 않았습니다.")

    if st.button(
        "RAG 인덱스 생성·갱신",
        use_container_width=True,
    ):
        if not api_key:
            st.error("RAG 인덱스 생성에는 OpenAI API 키가 필요합니다.")
        else:
            try:
                with st.spinner("지식문서를 임베딩하고 있습니다..."):
                    built_index = build_rag_index(
                        api_key=api_key,
                        embedding_model=embedding_model_name,
                        include_demo=include_demo_knowledge,
                    )
                st.session_state["rag_index_info"] = {
                    "created_at": built_index.get("created_at"),
                    "document_count": built_index.get("document_count"),
                    "chunk_count": built_index.get("chunk_count"),
                }
                st.success(
                    "RAG 인덱스 생성 완료 · "
                    f"문서 {built_index['document_count']}개 / "
                    f"청크 {built_index['chunk_count']}개"
                )
                st.rerun()
            except Exception as error:
                st.error(f"RAG 인덱스 생성 오류: {error}")
                st.exception(error)

    st.caption(f"지식문서 폴더: {KNOWLEDGE_BASE_DIR}")

    st.divider()
    st.subheader("데모")

    if st.button(
        "데모 OCR 텍스트 불러오기",
        use_container_width=True,
    ):
        st.session_state["ocr_text_editor"] = DEMO_OCR_TEXT
        st.session_state["ocr_result_info"] = {
            "source": "데모 OCR 텍스트"
        }
        st.rerun()

    if st.button(
        "데모 구조화 결과 불러오기",
        use_container_width=True,
    ):
        load_structured_result(build_demo_extracted_result())
        st.rerun()

    if st.button("전체 초기화", use_container_width=True):
        for key, default_value in SESSION_DEFAULTS.items():
            if isinstance(default_value, dict):
                st.session_state[key] = {}
            elif isinstance(default_value, list):
                st.session_state[key] = []
            else:
                st.session_state[key] = default_value

        st.rerun()


# =========================================================
# 13. 상단 화면 및 권한 확인
# =========================================================

st.markdown(
    '<div class="main-title">상속 금융비서</div>',
    unsafe_allow_html=True,
)
st.markdown(
    (
        '<div class="sub-title">'
        "상속 금융조회 결과를 인식하고 확인할 업무와 "
        "필요서류를 정리하는 데모"
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.info(
    "본 서비스는 상속권이나 법률적 결론을 결정하지 않습니다. "
    "실제 신청과 처리는 관련 기관의 확인이 필요합니다."
)

st.subheader("1. 권한 확인")

auth_col1, auth_col2, auth_col3 = st.columns(3)

with auth_col1:
    identity_verified = st.checkbox(
        "본인인증 완료",
        value=True,
    )

with auth_col2:
    deceased_verified = st.checkbox(
        "피상속인 정보 확인",
        value=True,
    )

with auth_col3:
    relationship_verified = st.checkbox(
        "상속관계 확인",
        value=True,
    )

if all(
    [
        identity_verified,
        deceased_verified,
        relationship_verified,
    ]
):
    st.success("데모상 권한 확인 단계가 완료되었습니다.")
else:
    st.warning(
        "실제 서비스에서는 권한 확인이 끝난 사용자만 "
        "문서 분석 결과를 사용할 수 있어야 합니다."
    )


# =========================================================
# 14. 탭
# =========================================================

(
    tab_document,
    tab_detail,
    tab_dashboard,
    tab_tasks,
    tab_documents,
    tab_schedule,
    tab_chat,
) = st.tabs(
    [
        "📑 문서 인식",
        "📋 상세정보·이상치",
        "📊 대시보드",
        "✅ 오늘 할 일",
        "📄 필요서류",
        "🗓 일정표",
        "💬 AI 상담",
    ]
)


# =========================================================
# 15. 문서 인식
# =========================================================

with tab_document:
    st.header("2. 데이터 수집 및 문서 인식")

    uploaded_document = st.file_uploader(
        "PDF, 이미지 또는 텍스트 파일을 업로드하세요.",
        type=[
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "webp",
            "txt",
        ],
    )

    button_col1, button_col2 = st.columns(2)

    with button_col1:
        run_ocr_button = st.button(
            "RapidOCR 문서 인식",
            type="primary",
            use_container_width=True,
        )

    with button_col2:
        load_text_button = st.button(
            "TXT 파일 직접 불러오기",
            use_container_width=True,
        )

    if run_ocr_button:
        if uploaded_document is None:
            st.warning("먼저 PDF 또는 이미지 파일을 업로드하세요.")
        elif uploaded_document.name.lower().endswith(".txt"):
            st.warning(
                "TXT 파일은 'TXT 파일 직접 불러오기'를 사용하세요."
            )
        else:
            try:
                with st.spinner(
                    "PDF를 이미지로 변환하고 RapidOCR로 "
                    "인식하고 있습니다..."
                ):
                    images = uploaded_file_to_images(
                        file_bytes=uploaded_document.getvalue(),
                        filename=uploaded_document.name,
                        dpi=int(render_dpi),
                        max_pages=int(max_pages),
                    )

                    ocr_text, result_info = run_rapidocr(
                        images=images,
                        min_score=float(minimum_ocr_score),
                    )

                st.session_state["ocr_text_editor"] = ocr_text
                st.session_state["ocr_result_info"] = result_info
                st.success("RapidOCR 문서 인식이 완료되었습니다.")
                st.rerun()
            except Exception as error:
                st.error(f"RapidOCR 실행 오류: {error}")
                st.exception(error)

    if load_text_button:
        if uploaded_document is None:
            st.warning("먼저 TXT 파일을 업로드하세요.")
        elif not uploaded_document.name.lower().endswith(".txt"):
            st.warning("TXT 파일만 직접 불러올 수 있습니다.")
        else:
            raw_bytes = uploaded_document.getvalue()
            decoded_text = None

            for encoding in [
                "utf-8-sig",
                "utf-8",
                "cp949",
                "euc-kr",
            ]:
                try:
                    decoded_text = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if decoded_text is None:
                st.error("텍스트 파일 인코딩을 해석하지 못했습니다.")
            else:
                st.session_state["ocr_text_editor"] = decoded_text
                st.session_state["ocr_result_info"] = {
                    "source": uploaded_document.name,
                    "method": "TXT 직접 입력",
                }
                st.success("텍스트 파일을 불러왔습니다.")
                st.rerun()

    result_info = st.session_state.get("ocr_result_info", {})

    if result_info:
        info_col1, info_col2, info_col3 = st.columns(3)

        info_col1.metric(
            "처리 페이지",
            result_info.get("page_count", "-"),
        )

        confidence = result_info.get("average_confidence")
        info_col2.metric(
            "평균 신뢰도",
            (
                f"{confidence:.3f}"
                if isinstance(confidence, (int, float))
                else "-"
            ),
        )

        info_col3.metric(
            "저신뢰 항목",
            result_info.get("low_confidence_count", "-"),
        )

    st.subheader("OCR 텍스트 확인 및 수정")

    st.text_area(
        "기관명, 금액, 날짜를 원문과 비교해 수정하세요.",
        key="ocr_text_editor",
        height=470,
    )

    extract_button = st.button(
        "LLM 정보 추출 + 정규화 + 규칙 이상 탐지",
        type="primary",
        use_container_width=True,
        disabled=not bool(api_key),
    )

    if not api_key:
        st.caption(
            "이 버튼은 OpenAI API 키가 있어야 사용할 수 있습니다. "
            "규칙 이상 탐지 자체는 LLM 추출이 끝난 데이터에 대해 "
            "Python에서 실행됩니다."
        )

    if extract_button:
        ocr_text = st.session_state.get("ocr_text_editor", "")

        if not ocr_text.strip():
            st.warning("먼저 OCR 텍스트를 입력하세요.")
        else:
            try:
                with st.spinner(
                    "LLM 정보 추출 후 규칙 기반 이상 탐지를 "
                    "수행하고 있습니다..."
                ):
                    extracted = extract_information_with_llm(
                        ocr_text=ocr_text,
                        api_key=api_key,
                        model=model_name,
                    )

                    load_structured_result(extracted)

                st.success(
                    "정보 추출, 정규화, 규칙 이상 탐지가 "
                    "완료되었습니다."
                )
                st.rerun()
            except Exception as error:
                st.error(f"정보 추출 오류: {error}")
                st.exception(error)

    normalized_result = st.session_state.get("normalized_result")

    if normalized_result is not None:
        st.divider()
        st.subheader("추출 결과 미리보기")

        preview_col1, preview_col2 = st.columns(2)

        with preview_col1:
            st.markdown("**문서정보**")
            st.json(normalized_result.get("document", {}))

        with preview_col2:
            st.markdown("**피상속인 정보**")
            st.json(normalized_result.get("deceased", {}))

        anomaly_df = st.session_state.get("anomaly_df")

        if anomaly_df is not None and not anomaly_df.empty:
            st.warning(
                f"규칙 기반 이상치가 {len(anomaly_df)}건 "
                "탐지되었습니다. 상세정보·이상치 탭에서 확인하세요."
            )
        else:
            st.success("규칙 기반 이상치가 탐지되지 않았습니다.")


# =========================================================
# 16. 상세정보 및 규칙 이상치
# =========================================================

with tab_detail:
    st.header("정규화된 금융거래 및 규칙 이상치")

    finance_df = st.session_state.get("extracted_finance_df")

    if finance_df is None or finance_df.empty:
        st.info("구조화된 금융거래 데이터가 없습니다.")
    else:
        edited_finance_df = st.data_editor(
            finance_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="finance_editor",
            column_config={
                "category": st.column_config.SelectboxColumn(
                    "분류",
                    options=ALLOWED_CATEGORIES,
                ),
                "item_name": "항목",
                "institution": "기관",
                "amount_text": "원문 금액",
                "amount_won": st.column_config.NumberColumn(
                    "정규화 금액(원)",
                    min_value=0,
                    format="%d",
                ),
                "event_type": "기일 종류",
                "event_date": "기일",
                "status": "상태",
                "note": "비고",
                "priority": "우선순위",
                "priority_score": "우선순위 점수",
                "priority_reason": "우선순위 근거",
                "days_remaining": "남은 일수",
                "needs_review": "사용자 확인 필요",
                "review_reason": "검토 사유",
                "anomaly_score": "이상치 점수",
                "anomaly_count": "이상치 개수",
                "source_text": "원문 근거",
            },
            disabled=[
                "priority",
                "priority_score",
                "priority_reason",
                "days_remaining",
                "anomaly_score",
                "anomaly_count",
            ],
        )

        st.session_state["extracted_finance_df"] = edited_finance_df

        if st.button(
            "수정 데이터 규칙 재검증",
            type="primary",
            use_container_width=True,
        ):
            try:
                revalidate_edited_finance_data()
                st.success("규칙 재검증이 완료되었습니다.")
                st.rerun()
            except Exception as error:
                st.error(f"재검증 오류: {error}")
                st.exception(error)

        st.divider()
        st.subheader("규칙 기반 이상 탐지 결과")

        anomaly_df = st.session_state.get("anomaly_df")

        if anomaly_df is None or anomaly_df.empty:
            st.success("현재 탐지된 규칙 이상치가 없습니다.")
        else:
            high_count = int(
                (anomaly_df["severity"] == "high").sum()
            )
            review_count = int(
                anomaly_df["requires_review"].fillna(False).sum()
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("전체 이상치", len(anomaly_df))
            metric_col2.metric("높은 심각도", high_count)
            metric_col3.metric("사용자 확인 필요", review_count)

            st.dataframe(
                anomaly_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "item_index": "항목 번호(0부터 시작)",
                    "anomaly_label": "이상 유형",
                    "anomaly_type": "이상 코드",
                    "field": "필드",
                    "severity": "심각도",
                    "reason": "탐지 사유",
                    "source_value": "원문 값",
                    "extracted_value": "추출 값",
                    "confidence": "규칙 신뢰도",
                    "anomaly_score": "이상치 점수",
                    "requires_review": "확인 필요",
                    "detector": "탐지기",
                },
            )

            st.caption(
                "item_index가 비어 있으면 금융 항목이 아니라 "
                "문서 기본정보에서 발견된 이상치입니다."
            )

        download_col1, download_col2, download_col3 = (
            st.columns(3)
        )

        with download_col1:
            st.download_button(
                "금융거래 CSV",
                data=dataframe_to_csv_bytes(
                    st.session_state["extracted_finance_df"]
                ),
                file_name="normalized_financial_items.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with download_col2:
            current_anomaly_df = st.session_state.get("anomaly_df")

            if current_anomaly_df is not None:
                st.download_button(
                    "이상치 CSV",
                    data=dataframe_to_csv_bytes(
                        current_anomaly_df
                    ),
                    file_name="rule_anomalies.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with download_col3:
            normalized_result = st.session_state.get(
                "normalized_result"
            )

            if normalized_result:
                st.download_button(
                    "전체 JSON",
                    data=safe_json_dumps(
                        normalized_result
                    ).encode("utf-8"),
                    file_name="normalized_extraction.json",
                    mime="application/json",
                    use_container_width=True,
                )


# =========================================================
# 17. 대시보드
# =========================================================

with tab_dashboard:
    st.header("금융거래 대시보드")

    finance_df = st.session_state.get("extracted_finance_df")
    normalized_result = st.session_state.get("normalized_result")

    if finance_df is None or finance_df.empty:
        st.info(
            "문서 인식 탭에서 정보 추출을 실행하거나 "
            "데모 구조화 결과를 불러오세요."
        )
    else:
        working_df = finance_df.copy()
        working_df["amount_won"] = pd.to_numeric(
            working_df["amount_won"],
            errors="coerce",
        ).fillna(0)

        assets = working_df[
            working_df["category"].isin(["예금", "휴면예금"])
        ]["amount_won"].sum()

        liabilities = working_df[
            working_df["category"].isin(["대출", "카드"])
        ]["amount_won"].sum()

        recurring = working_df[
            working_df["category"].isin(["자동이체", "보험"])
        ]["amount_won"].sum()

        review_count = int(
            working_df["needs_review"].fillna(False).sum()
        )

        metric_col1, metric_col2, metric_col3, metric_col4 = (
            st.columns(4)
        )

        metric_col1.metric("확인된 자산", f"{int(assets):,}원")
        metric_col2.metric(
            "확인된 채무·카드",
            f"{int(liabilities):,}원",
        )
        metric_col3.metric(
            "정기 납부 항목",
            f"{int(recurring):,}원",
        )
        metric_col4.metric("검토 필요 항목", f"{review_count}건")

        category_summary = (
            working_df.groupby("category", as_index=False)["amount_won"]
            .sum()
            .sort_values("amount_won", ascending=False)
            .set_index("category")
        )

        st.subheader("분류별 금액")
        st.bar_chart(category_summary)

        st.subheader("우선 확인 항목")

        priority_order = {
            "긴급": 0,
            "높음": 1,
            "보통": 2,
            "낮음": 3,
        }

        priority_view = (
            working_df.assign(
                priority_order=working_df["priority"].map(
                    priority_order
                )
            )
            .sort_values(
                ["priority_order", "priority_score"],
                ascending=[True, False],
            )
            [
                [
                    "priority",
                    "category",
                    "item_name",
                    "amount_won",
                    "event_date",
                    "needs_review",
                    "note",
                    "priority_reason",
                ]
            ]
        )

        st.dataframe(
            priority_view,
            use_container_width=True,
            hide_index=True,
        )

        if normalized_result:
            st.caption(
                "우선순위 기준일: "
                f"{normalized_result.get('base_date', '-')}"
            )


# =========================================================
# 18. 오늘 할 일
# =========================================================

with tab_tasks:
    st.header("오늘 우선 확인할 일")

    finance_df = st.session_state.get("extracted_finance_df")

    if finance_df is None or finance_df.empty:
        st.info("우선순위를 계산할 데이터가 없습니다.")
    else:
        task_df = finance_df.copy()
        task_df["priority_score"] = pd.to_numeric(
            task_df["priority_score"],
            errors="coerce",
        ).fillna(0)
        task_df["days_remaining_numeric"] = pd.to_numeric(
            task_df["days_remaining"],
            errors="coerce",
        ).fillna(9999)

        task_df = task_df[
            task_df["priority"].isin(["긴급", "높음"])
            | (task_df["days_remaining_numeric"] <= 7)
            | task_df["needs_review"].fillna(False)
        ].sort_values("priority_score", ascending=False)

        if task_df.empty:
            st.success("현재 우선 확인할 항목이 없습니다.")
        else:
            for _, row in task_df.iterrows():
                with st.expander(
                    (
                        f"{row.get('priority', '확인')} · "
                        f"{row.get('item_name', '항목 확인')}"
                    ),
                    expanded=True,
                ):
                    st.write(
                        f"**기관:** {row.get('institution') or '-'}"
                    )

                    amount = normalize_amount(row.get("amount_won"))
                    st.write(
                        f"**금액:** "
                        f"{amount:,}원"
                        if amount is not None
                        else "**금액:** 확인 필요"
                    )
                    st.write(
                        f"**기일:** {row.get('event_date') or '-'}"
                    )
                    st.write(
                        f"**확인사항:** {row.get('note') or '원문 확인'}"
                    )

                    if safe_bool(row.get("needs_review")):
                        st.warning(
                            f"추출 검토 필요: "
                            f"{row.get('review_reason') or '원문 비교 필요'}"
                        )

                    st.caption(row.get("priority_reason") or "")


# =========================================================
# 19. 필요서류
# =========================================================

with tab_documents:
    st.header("제출 필요서류")

    documents_df = st.session_state.get("required_documents_df")

    if documents_df is None or documents_df.empty:
        st.info("문서에서 추출된 필요서류가 없습니다.")
    else:
        documents_df = documents_df.copy()

        if "준비완료" not in documents_df.columns:
            documents_df.insert(0, "준비완료", False)

        edited_documents_df = st.data_editor(
            documents_df,
            use_container_width=True,
            hide_index=True,
            key="required_documents_editor",
            column_config={
                "준비완료": st.column_config.CheckboxColumn(
                    "준비완료"
                ),
                "document_name": "서류명",
                "submission_status": "제출 여부",
                "note": "비고",
                "source_text": "원문 근거",
            },
        )

        st.session_state["required_documents_df"] = (
            edited_documents_df
        )

        completed = int(
            edited_documents_df["준비완료"].fillna(False).sum()
        )
        total = len(edited_documents_df)

        st.progress(completed / total if total else 0)
        st.caption(f"{completed} / {total}개 서류 준비 완료")


# =========================================================
# 20. 일정표
# =========================================================

with tab_schedule:
    st.header("기일 일정표")

    finance_df = st.session_state.get("extracted_finance_df")

    if finance_df is None or finance_df.empty:
        st.info("기일이 포함된 데이터가 없습니다.")
    else:
        schedule_df = finance_df[
            finance_df["event_date"].notna()
            & (finance_df["event_date"].astype(str).str.strip() != "")
        ].copy()

        if schedule_df.empty:
            st.info("추출된 기일 정보가 없습니다.")
        else:
            schedule_df["event_date_parsed"] = pd.to_datetime(
                schedule_df["event_date"],
                errors="coerce",
            )
            schedule_df = schedule_df.sort_values(
                "event_date_parsed"
            )

            st.dataframe(
                schedule_df[
                    [
                        "event_date",
                        "event_type",
                        "category",
                        "item_name",
                        "institution",
                        "amount_won",
                        "priority",
                        "needs_review",
                        "note",
                    ]
                ].rename(
                    columns={
                        "event_date": "날짜",
                        "event_type": "기일 종류",
                        "category": "분류",
                        "item_name": "항목",
                        "institution": "기관",
                        "amount_won": "금액(원)",
                        "priority": "우선순위",
                        "needs_review": "검토 필요",
                        "note": "확인사항",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


# =========================================================
# 21. 임베딩 RAG 기반 AI 상담
# =========================================================

with tab_chat:
    st.header("임베딩 RAG 기반 AI 상담")

    normalized_result = st.session_state.get("normalized_result")

    st.caption(
        "개인 금융정보는 구조화 검색하고, 상속업무 지식문서는 "
        "OpenAI 임베딩과 로컬 벡터 인덱스로 검색합니다."
    )

    with st.expander("RAG 지식문서 관리", expanded=False):
        st.write(
            "TXT, Markdown, PDF 문서를 knowledge_base 폴더에 "
            "저장한 뒤 RAG 인덱스를 갱신합니다."
        )

        uploaded_knowledge_files = st.file_uploader(
            "지식문서 추가",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            key="rag_knowledge_uploader",
        )

        if st.button(
            "업로드한 지식문서 저장",
            disabled=not uploaded_knowledge_files,
            use_container_width=True,
        ):
            try:
                saved_files = save_uploaded_knowledge_files(
                    uploaded_knowledge_files or []
                )
                if saved_files:
                    st.success(
                        f"{len(saved_files)}개 문서를 저장했습니다. "
                        "사이드바에서 RAG 인덱스를 갱신하세요."
                    )
                else:
                    st.warning("저장할 수 있는 문서가 없습니다.")
            except Exception as error:
                st.error(f"지식문서 저장 오류: {error}")
                st.exception(error)

        knowledge_paths = list_knowledge_files(
            include_demo=include_demo_knowledge
        )
        st.markdown(f"**현재 사용 대상 문서: {len(knowledge_paths)}개**")
        for path in knowledge_paths:
            st.caption(str(path.relative_to(KNOWLEDGE_BASE_DIR)))

        status = rag_index_status(
            embedding_model=embedding_model_name,
            include_demo=include_demo_knowledge,
        )
        if status["current"]:
            st.success(
                "인덱스 최신 상태 · "
                f"문서 {status['document_count']}개 / "
                f"청크 {status['chunk_count']}개"
            )
        else:
            st.warning(
                "인덱스가 없거나 오래되었습니다. 질문 시 자동으로 "
                "생성되지만, 사이드바에서 미리 생성하는 것을 권장합니다."
            )

    if not api_key:
        st.warning(
            "RAG 검색용 임베딩과 답변 생성을 위해 OpenAI API 키가 "
            "필요합니다."
        )
    else:
        if normalized_result is None:
            st.info(
                "개인 금융정보가 아직 없습니다. 일반 상속 절차 질문은 "
                "지식문서 RAG로 답변할 수 있습니다."
            )

        for chat_item in st.session_state["chat_history"]:
            with st.chat_message(chat_item["role"]):
                st.markdown(chat_item["content"])

                rag_context = chat_item.get("rag_context")
                if rag_context:
                    with st.expander("사용한 RAG 근거"):
                        render_rag_evidence(rag_context)
                elif chat_item.get("evidence"):
                    # 이전 버전의 채팅 기록과 호환한다.
                    with st.expander("사용한 근거"):
                        for index, evidence in enumerate(
                            chat_item["evidence"],
                            start=1,
                        ):
                            st.markdown(
                                f"**근거 {index}**  \n{evidence}"
                            )

        question = st.chat_input(
            "예: 내 대출이 있는데 어떤 절차를 먼저 확인해야 하나요?"
        )

        if question:
            st.session_state["chat_history"].append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            try:
                with st.chat_message("assistant"):
                    with st.spinner(
                        "개인정보와 지식문서를 검색하고 답변을 "
                        "생성하고 있습니다..."
                    ):
                        answer, rag_context = answer_with_rag(
                            question=question,
                            normalized=normalized_result,
                            api_key=api_key,
                            model=model_name,
                            embedding_model=embedding_model_name,
                            include_demo=include_demo_knowledge,
                            personal_top_k=rag_personal_top_k,
                            knowledge_top_k=rag_knowledge_top_k,
                            minimum_score=rag_minimum_score,
                        )

                    st.markdown(answer)

                    with st.expander("사용한 RAG 근거", expanded=True):
                        render_rag_evidence(rag_context)

                    index_info = rag_context.get("index_info", {})
                    if index_info.get("rebuilt"):
                        st.caption(
                            "지식문서 변경을 감지하여 RAG 인덱스를 "
                            "자동으로 갱신했습니다."
                        )

                st.session_state["rag_last_context"] = rag_context
                st.session_state["chat_history"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "rag_context": rag_context,
                    }
                )
            except Exception as error:
                st.error(f"RAG 상담 오류: {error}")
                st.exception(error)

