# nlp/sentiment_engine.py
import functools
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from underthesea import word_tokenize

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

SLANG_MAP = {
    "rat": "rất",
    "hok": "không",
    "ko": "không",
    "k": "không",
    "dc": "được",
    "vs": "với",
    "dep": "đẹp",
    "xau": "xấu",
}


# =====================
# 1. TIỀN XỬ LÝ TIẾNG VIỆT
# =====================

def normalize_slang(text: str) -> str:
    """Thay viết tắt bằng từ gốc."""
    tokens = text.split()
    new_tokens = []
    for t in tokens:
        key = t.lower()
        new_tokens.append(SLANG_MAP.get(key, t))
    return " ".join(new_tokens)


def preprocess_text(raw_text: str) -> str:
    """
    Chuẩn hóa tiếng Việt:
    - Xóa khoảng trắng
    - lower case
    - sửa viết tắt
    - tách từ
    - giới hạn độ dài
    """
    text = raw_text.strip()
    if not text:
        return text

    text = text.lower()
    text = normalize_slang(text)

    try:
        text = word_tokenize(text, format="text")
    except Exception:
        pass

    return text[:200]  # giới hạn 200 ký tự


# =====================
# 2. TẢI PIPELINE TRANSFORMER
# =====================

@functools.lru_cache(maxsize=1)
def get_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    clf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return clf


def map_label(label: str) -> str:
    """Map nhãn model → 3 nhãn chính."""
    raw = label.upper().strip()

    if "STAR" in raw:
        try:
            stars = int(raw.split()[0])
        except:
            stars = 3

        if stars <= 2:
            return "NEGATIVE"
        if stars == 3:
            return "NEUTRAL"
        return "POSITIVE"

    for s in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        if s in raw:
            return s

    return "NEUTRAL"


# =====================
# 3. PHÂN LOẠI CẢM XÚC
# =====================

def classify_sentiment(raw_text: str) -> dict:
    """Trả về {text, sentiment, score}"""

    if not raw_text or len(raw_text.strip()) < 5:
        raise ValueError("Câu quá ngắn, vui lòng nhập ít nhất 5 ký tự.")

    preprocessed = preprocess_text(raw_text)

    clf = get_pipeline()
    result = clf(preprocessed)[0]

    label = map_label(result["label"])
    score = float(result["score"])

    if score < 0.5:
        label = "NEUTRAL"

    return {
        "text": raw_text,
        "sentiment": label,
        "score": score,
    }
