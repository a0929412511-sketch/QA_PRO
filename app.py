import csv
import os
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="美肌相談室", page_icon="🌷", layout="centered")

DATA_PATH = Path(__file__).with_name("QA.csv")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.32"))
LOW_CONFIDENCE_REPLY = (
    "目前的常見問題中沒有找到足夠相近的答案。您可以換個方式描述，"
    "或聯絡診所客服；若涉及個人健康狀況或療程適用性，請由合格醫師評估。"
)
CATEGORY_LABELS = {
    "booking": "預約與諮詢",
    "services": "療程資訊",
    "eligibility": "適合評估",
    "pricing": "費用與付款",
    "precare": "術前準備",
    "aftercare": "術後照護",
    "general": "一般服務",
}


def load_faqs():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到問答資料：{DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"question", "answer"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("QA.csv 必須包含 question 和 answer 欄位，且至少有一筆資料。")

    faqs = [
        {
            "category": (row.get("category") or "general").strip(),
            "question": (row.get("question") or "").strip(),
            "answer": (row.get("answer") or "").strip(),
        }
        for row in rows
        if (row.get("question") or "").strip() and (row.get("answer") or "").strip()
    ]
    if not faqs:
        raise ValueError("QA.csv 沒有可用的問答資料。")
    return faqs


def faq_text(faq):
    # Match the input text used to generate the vectors in qa.sql.
    return (
        f"Category: {faq['category']}\n"
        f"Question: {faq['question']}\n"
        f"Answer: {faq['answer']}"
    )


@st.cache_resource(show_spinner=False)
def get_search_index():
    faqs = load_faqs()
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        [faq_text(faq) for faq in faqs],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return model, faqs, np.asarray(vectors, dtype=np.float32)


def find_answer(question, model, faqs, faq_vectors, candidate_indices):
    query_vector = model.encode(
        [f"Question: {question}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    similarities = faq_vectors[candidate_indices] @ query_vector
    best_local_index = int(np.argmax(similarities))
    best_faq = faqs[candidate_indices[best_local_index]]

    if float(similarities[best_local_index]) < SIMILARITY_THRESHOLD:
        return LOW_CONFIDENCE_REPLY

    return f"{best_faq['answer']}\n\n**參考問題：** {best_faq['question']}"


def submit_example(question):
    st.session_state.pending_question = question


def clear_conversation():
    st.session_state.messages = []


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@400;500;600;700&display=swap');
    .stApp {
        background:
            radial-gradient(ellipse at 10% 4%, rgba(231, 190, 193, .28), transparent 30%),
            radial-gradient(ellipse at 92% 24%, rgba(221, 191, 142, .20), transparent 25%),
            #fbf5ec;
        color: #3f2930;
        font-family: 'Noto Serif TC', 'Songti TC', serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f3e5d9 0%, #f8eee4 100%);
        border-right: 1px solid #ddc49f;
    }
    h1, h2, h3, p, label { font-family: 'Noto Serif TC', 'Songti TC', serif; }
    .hero-card {
        text-align: center;
        padding: 1.7rem 1.2rem 1.35rem;
        margin: .4rem 0 1.35rem;
        border: 1px solid #d5b77f;
        border-radius: 24px;
        background: linear-gradient(145deg, rgba(255,252,246,.96), rgba(250,237,229,.94));
        box-shadow: 0 8px 24px rgba(111, 61, 68, .08), inset 0 0 0 5px rgba(213,183,127,.12);
    }
    .hero-kicker { color: #9b6e56; letter-spacing: .22em; font-size: .78rem; }
    .hero-title { color: #76394d; font-size: 2rem; font-weight: 700; margin: .35rem 0; }
    .hero-copy { color: #705b57; font-size: .96rem; margin: 0; }
    .hero-ornament { color: #b58b55; letter-spacing: .45em; margin-top: .75rem; }
    [data-testid="stChatMessage"] {
        background: rgba(255, 252, 246, .9);
        border: 1px solid #e8d7bd;
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(92, 53, 60, .045);
    }
    div.stButton > button {
        color: #73384c;
        background: #fffaf3;
        border: 1px solid #d4b27c;
        border-radius: 16px;
        transition: all .18s ease;
    }
    div.stButton > button:hover {
        color: #fffaf3;
        background: #8b4b5b;
        border-color: #8b4b5b;
    }
    [data-testid="stChatInput"] textarea {
        background: #fffdf8;
        border-color: #d6bb91;
        border-radius: 16px;
    }
    </style>
    <div class="hero-card">
        <div class="hero-kicker">✦ BEAUTY SALON FAQ ✦</div>
        <div class="hero-title">🌷 美肌相談室 🌷</div>
        <p class="hero-copy">以一點古典浪漫，陪你找到安心的療程資訊</p>
        <div class="hero-ornament">❧　✧　❧</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("回覆根據診所常見問題提供；個人療程適用性與醫療建議請諮詢合格醫師。")

with st.spinner("正在載入語意模型與常見問答索引…首次啟動可能需要一些時間。"):
    try:
        model, faqs, faq_vectors = get_search_index()
    except Exception as exc:
        st.error(f"無法載入問答服務：{exc}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

available_codes = {faq["category"] for faq in faqs}
category_codes = [code for code in CATEGORY_LABELS if code in available_codes]
category_codes.extend(sorted(available_codes - set(category_codes)))
category_options = ["所有類別"] + [CATEGORY_LABELS.get(code, code) for code in category_codes]
label_to_code = {CATEGORY_LABELS.get(code, code): code for code in category_codes}

with st.sidebar:
    st.markdown("### ❧ 問題分類 ❧")
    selected_label = st.selectbox("依主題瀏覽與搜尋", category_options, label_visibility="collapsed")
    selected_code = label_to_code.get(selected_label)
    if selected_code:
        candidate_indices = [i for i, faq in enumerate(faqs) if faq["category"] == selected_code]
    else:
        candidate_indices = list(range(len(faqs)))
    st.caption(f"目前分類共 {len(candidate_indices)} 題")
    st.markdown("---")
    st.markdown("##### ✿ 諮詢小提醒")
    st.caption("線上客服提供一般 FAQ。若有不適、病史或療程安全疑問，請直接聯絡診所或醫師。")
    if st.session_state.messages:
        st.button("清除對話", on_click=clear_conversation, use_container_width=True)

with st.expander("常見問題靈感", expanded=not st.session_state.messages):
    for start in range(0, min(6, len(candidate_indices)), 2):
        columns = st.columns(2)
        for column, faq_index in zip(columns, candidate_indices[start : start + 2]):
            faq = faqs[faq_index]
            column.button(
                faq["question"],
                key=f"example_{selected_label}_{faq_index}",
                use_container_width=True,
                on_click=submit_example,
                args=(faq["question"],),
            )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("請輸入您的問題，例如：療程後可以化妝嗎？")
if prompt is None:
    prompt = st.session_state.pop("pending_question", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在搜尋最相關的常見問答…"):
            response = find_answer(prompt, model, faqs, faq_vectors, candidate_indices)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
