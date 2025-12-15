import re
import os
import sys
import uuid
import time
import threading

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

import streamlit as st
import markdown as md 
from openai import OpenAI

from pypdf import PdfReader
import boto3
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


# Configuration constants
DEFAULT_BASE_URL = "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"
DEFAULT_MODEL = "openai.gpt-oss-120b-1:0"
DEFAULT_NER_MODEL = "nazar0202/xlm-roberta-base-ner"
MAX_CONTEXT_CHARS = 12000
MAX_PER_FILE_CHARS = 4000
TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"}
OCR_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}

UPLOAD_ROOT = Path("uploads")
UPLOAD_S3_BUCKET = os.getenv("UPLOAD_S3_BUCKET", "legal-ocr-input")
DOWNLOAD_S3_BUCKET = os.getenv("DOWNLOAD_S3_BUCKET", "legal-ocr-output")



def readable_size(path: Path) -> str:
    size_kb = path.stat().st_size / 1024
    return f"{size_kb:.1f} KB"


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    api_key = os.getenv("BEDROCK_KEY")
    if not api_key:
        raise RuntimeError(
            "BEDROCK_KEY environment variable must be set before launching the app."
        )
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(base_url=base_url, api_key=api_key)

@lru_cache(maxsize=1)
def get_rag_assistant():
    rag_dir = Path(__file__).parent / "services" / "rag-pipeline"
    if str(rag_dir) not in sys.path:
        sys.path.append(str(rag_dir))
    from final_assistant import FinalLegalAssistantRAG  # type: ignore
    return FinalLegalAssistantRAG()


@lru_cache(maxsize=1)
def get_ner_pipeline():
    model_name = os.getenv("NER_MODEL_NAME", DEFAULT_NER_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="checkpoint-160")
    model = AutoModelForTokenClassification.from_pretrained(model_name, subfolder="checkpoint-160")
    device = "cuda" if torch.cuda.is_available() else -1
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )


def anonymize_text(text: str) -> Tuple[str, List[str]]:
    """Replace detected entities with their labels, returning the new text and stats."""
    if not text.strip():
        return "", []

    ner = get_ner_pipeline()
    print("Running NER for anonymization...")
    predictions = ner(text)

    print(f"NER predictions: {predictions}")
    if not predictions:
        return text, []

    # Sort by start index to rebuild text in order
    predictions = sorted(predictions, key=lambda p: p["start"])
    result_parts: List[str] = []
    last_idx = 0
    counts: dict[str, int] = {}

    for pred in predictions:
        start, end = int(pred["start"]), int(pred["end"])
        label = (pred.get("entity_group") or pred.get("entity") or "ENT").upper()
        result_parts.append(text[last_idx:start])
        result_parts.append(f"[{label}]")
        last_idx = end
        counts[label] = counts.get(label, 0) + 1

    result_parts.append(text[last_idx:])
    anonymized = "".join(result_parts)

    stats = [f"{label}: {count}" for label, count in sorted(counts.items())]
    return anonymized, stats


def read_text_file(path: Path) -> Tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text, f"текст витягнуто ({readable_size(path)})"
    except Exception as exc:
        return "", f"не вдалося прочитати текстовий файл: {exc}"


def read_pdf(path: Path) -> Tuple[str, str]:
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        page_count = len(reader.pages)
        pages_word = "сторінка" if page_count == 1 else "сторінок" if page_count >= 5 else "сторінки"
        pages_info = f"{page_count} {pages_word}"
        return text, f"текст витягнуто з PDF ({pages_info}, {readable_size(path)})"
    except Exception as exc:
        return "", f"не вдалося обробити PDF: {exc}"


def read_docx(path: Path) -> Tuple[str, str]:
    """Extract text from DOCX files when python-docx is available."""
    try:
        import docx  # type: ignore
    except ImportError:
        return "", (
            f"DOCX завантажено ({readable_size(path)}); встановіть 'python-docx', щоб автоматично витягувати текст."
        )

    try:
        document = docx.Document(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
        return text, f"текст витягнуто з DOCX ({readable_size(path)})"
    except Exception as exc:
        return "", f"не вдалося обробити DOCX: {exc}"


def process_ocr(path: Path) -> Tuple[str, List[str]]:
    s3_client = boto3.client("s3")
    new_file_name = f"{uuid.uuid4()}"

    s3_client.upload_file(str(path), UPLOAD_S3_BUCKET, new_file_name)

    bucket_name = DOWNLOAD_S3_BUCKET
    file_name = f"{new_file_name}.txt"
    s3_client = boto3.client("s3")

    # Long polling to wait for the OCR result
    timeout = 600
    poll_interval = 15
    elapsed_time = 0

    try:
        while elapsed_time < timeout:
            try:
                obj = s3_client.get_object(Bucket=bucket_name, Key="results/" + file_name)
                ocr_text = obj["Body"].read().decode("utf-8")

                print(ocr_text)
                print("OCR text retrieved, starting anonymization...")

                anonymized_text, stats = anonymize_text(ocr_text)
                print(anonymized_text)
                return anonymized_text, stats
            except s3_client.exceptions.NoSuchKey:
                time.sleep(poll_interval)
                elapsed_time += poll_interval

        raise TimeoutError("OCR result not available within the timeout period.")
    except Exception as exc:
        return f"Error during OCR processing: {exc}", []


# Async OCR runner
def start_async_ocr(path: Path, state: dict):
    def worker():
        try:
            text, stats = process_ocr(path)
            state["ocr_result"] = text
            state["ocr_anon_stats"] = stats
            state["ocr_status"] = "done"
        except Exception as exc:
            state["ocr_result"] = f"OCR failed: {exc}"
            state["ocr_status"] = "failed"
            state["ocr_anon_stats"] = []

    state["ocr_status"] = "running"
    state["ocr_result"] = ""

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()



def extract_file_context(path: Path) -> Tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix in IMAGE_EXTENSIONS:
        return "", (
            f"зображення завантажено ({readable_size(path)}); додайте підказку з описом його вмісту."
        )
    return "", f"{suffix or 'файл'} завантажено ({readable_size(path)}); витягнення не підтримується."


def merge_contexts(state: dict) -> str:
    parts = [state.get("base_context", ""), state.get("ocr_context", "")]
    return "\n\n".join(part for part in parts if part).strip()


def compose_context_display(state: dict) -> str:
    sections = [state.get("file_summary_md", ""), state.get("ocr_summary_md", "")]
    sections = [s for s in sections if s and s.strip()]
    return "\n\n".join(sections) if sections else "Контекст ще не завантажено."


def summarize_context(
    sections: List[str], summaries: List[str], title: str = "### Завантажений контекст"
) -> Tuple[str, str]:
    combined_context = "\n\n".join(sections).strip()
    if len(combined_context) > MAX_CONTEXT_CHARS:
        combined_context = combined_context[:MAX_CONTEXT_CHARS] + "\n...[контекст урізано]"

    summary_lines = "\n".join(f"- {line}" for line in summaries)
    context_md = f"{title}\n{summary_lines}" if summary_lines else "Контекст ще не завантажено."
    return combined_context, context_md


def process_files(
    uploaded_files: Optional[List[str]], state: Optional[dict]
) -> Tuple[dict, str, str]:
    state = state or {
        "context": "",
        "history": [],
        "files": [],
        "ocr_files": [],
        "ocr_anon_stats": [],
        "base_anon_stats": [],
    }
    if not uploaded_files:
        state.update(
            {
                "context": "",
                "files": [],
                "base_context": "",
                "file_summary_md": "",
                "base_anon_stats": [],
            }
        )
        state["context"] = merge_contexts(state)
        context_md = compose_context_display(state)
        return state, context_md, "Контекст очищено."

    extracted_sections: List[str] = []
    summary_lines: List[str] = []
    anonymization_summaries: List[str] = []
    for file_path in uploaded_files:
        path = Path(file_path)
        text, summary = extract_file_context(path)
        if text:
            anonymized_text, stats = anonymize_text(text)
            snippet = anonymized_text[:MAX_PER_FILE_CHARS]
            if len(anonymized_text) > MAX_PER_FILE_CHARS:
                snippet += "\n...[урізано]"
            extracted_sections.append(f"Джерело: {path.name}\n{snippet}")
            if stats:
                anonymization_summaries.append(f"{path.name}: {', '.join(stats)}")
            stats_note = f" | анонімізація: {', '.join(stats)}" if stats else ""
        else:
            stats_note = ""
        summary_lines.append(f"**{path.name}** — {summary}{stats_note}")

    combined_context, context_md = summarize_context(
        extracted_sections, summary_lines, title="### OCR контекст"
    )
    state.update(
        {
            "base_context": combined_context,
            "context": merge_contexts(state),
            "files": [Path(p).name for p in uploaded_files],
            "file_summary_md": context_md,
            "base_anon_stats": anonymization_summaries,
        }
    )
    file_count = len(uploaded_files)
    file_word = "файл" if file_count == 1 else "файли" if 2 <= file_count <= 4 else "файлів"
    status = f"Завантажено {file_count} {file_word}."
    combined_display = compose_context_display(state)
    return state, combined_display, status


def process_ocr_files(
    uploaded_files: Optional[List[str]], state: Optional[dict]
) -> Tuple[dict, str, str, str]:
    state = state or {
        "context": "",
        "history": [],
        "files": [],
        "ocr_files": [],
        "base_context": "",
        "ocr_anon_stats": [],
        "base_anon_stats": [],
    }

    if not uploaded_files:
        state.update({"ocr_context": "", "ocr_files": [], "ocr_summary_md": "", "ocr_anon_stats": []})
        state["context"] = merge_contexts(state)
        context_md = compose_context_display(state)
        return state, context_md, "OCR контекст очищено.", ""

    extracted_sections: List[str] = []
    summary_lines: List[str] = []
    processed_files = 0
    anonymization_summaries: List[str] = []

    for file_path in uploaded_files:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in OCR_EXTENSIONS:
            summary_lines.append(f"**{path.name}** — формат не підтримується для OCR.")
            continue

        text, stats = process_ocr(path)
        extracted_sections.append(f"Джерело: {path.name}\n{text}")
        stats_note = f" | анонімізація: {', '.join(stats)}" if stats else ""
        summary_lines.append(f"**{path.name}** — OCR чернетка ({readable_size(path)}){stats_note}")
        if stats:
            anonymization_summaries.append(f"{path.name}: {', '.join(stats)}")
        processed_files += 1

    combined_context, context_md = summarize_context(extracted_sections, summary_lines)
    state.update(
        {
            "ocr_context": combined_context,
            "ocr_files": [Path(p).name for p in uploaded_files],
            "ocr_summary_md": context_md,
            "context": merge_contexts(state),
            "ocr_anon_stats": anonymization_summaries,
        }
    )

    status = (
        f"OCR підготовлено для {processed_files} файл(ів)."
        if processed_files
        else "Файли для OCR не оброблено."
    )
    combined_display = compose_context_display(state)
    return state, combined_display, status, combined_context


def build_messages(state: dict, user_message: str) -> List[dict]:
    system_prompt = (
        "You are Legal Helper, a virtual legal research assistant. "
        "Provide general insights, outline options, and suggest next steps, "
        "but never offer definitive legal advice or claim to replace a lawyer. "
        "Always include a gentle reminder that the user should consult a qualified attorney."
    )
    messages: List[dict] = [{"role": "system", "content": system_prompt}]

    context = (state or {}).get("context", "").strip()
    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Supporting materials supplied by the user:\n"
                    f"{context}"
                ),
            }
        )

    for user_text, assistant_text in (state or {}).get("history", []):
        messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": user_message})
    return messages


def fetch_completion(messages: List[dict]) -> str:
    client = get_client()
    model_name = os.getenv("LEGAL_HELPER_MODEL", DEFAULT_MODEL)
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    message = completion.choices[0].message
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    content = (content or "").strip()

    content = re.sub(
        r"<reasoning>.*?</reasoning>",
        "",
        content,
        flags=re.DOTALL
    ).strip()

    return content


def initial_state() -> dict:
    return {
        "context": "",
        "history": [],
        "files": [],
        "ocr_files": [],
        "ocr_context": "",
        "base_context": "",
        "file_summary_md": "",
        "ocr_summary_md": "",
        "ocr_anon_stats": [],
        "base_anon_stats": [],
        "use_rag": False,
    }


def generate_response(user_message: str, state: Optional[dict]) -> Tuple[dict, str]:
    user_message = (user_message or "").strip()
    if not user_message:
        return state or initial_state(), "Введіть запитання."

    state = state or initial_state()
    state_history = state.get("history", [])
    state_history.append((user_message, ""))

    try:
        use_rag = (state or {}).get("use_rag")
        if use_rag:
            rag_query = user_message
            if state.get("context"):
                rag_query = f"{user_message}\n\nДодатковий контекст користувача:\n{state['context']}"
            assistant = get_rag_assistant()
            assistant_response = assistant.run_pipeline(rag_query)
        else:
            messages = build_messages(state, user_message)
            assistant_response = fetch_completion(messages)
    except Exception as exc:
        state_history[-1] = (user_message, "⚠️ Не вдалося підключитися до мовної моделі.")
        state["history"] = state_history
        return state, f"Помилка виклику моделі: {exc}"

    state_history[-1] = (user_message, assistant_response)
    state["history"] = state_history
    return state, "Відповідь згенеровано."


def clear_conversation() -> Tuple[dict, str, str]:
    cleared_state = initial_state()
    return cleared_state, "Розмову очищено.", "Контекст ще не завантажено."



# Streamlit UI
def save_uploaded_files(uploaded_files: List[Any], subdir: str) -> List[str]:
    if not uploaded_files:
        return []

    target_dir = UPLOAD_ROOT / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    for f in uploaded_files:
        path = target_dir / f.name
        # Overwrite if exists (fine for this use case)
        path.write_bytes(f.read())
        saved_paths.append(str(path))
    return saved_paths


def build_chat_html(
    history_pairs: List[Tuple[str, Optional[str]]],
    chat_height: int = 600,
) -> str:
    chat_history_html = ""

    for user_text, assistant_text in history_pairs:
        u_html = md.markdown(user_text or "") if user_text else ""
        a_html = md.markdown(assistant_text or "") if assistant_text else ""

        if user_text:
            chat_history_html += (
                '<div style="display:flex; justify-content:flex-end; '
                'margin-bottom:0.75rem;">'
                '  <div style="max-width:85%; text-align:left;">'
                '    <div style="font-size:0.8rem; color:#bbbbbb; text-align:right; '
                '                margin-bottom:0.15rem;">'
                '      🧑‍💼 Користувач:'
                '    </div>'
                '    <div style="padding:0.45rem 0.7rem; border-radius:12px; '
                '                border-bottom-right-radius:0; '
                '                background-color:#262730;">'
                f'      {u_html}'
                '    </div>'
                '  </div>'
                '</div>'
            )

        if assistant_text:
            chat_history_html += (
                '<div style="display:flex; justify-content:flex-start; '
                'margin-bottom:0.75rem;">'
                '  <div style="max-width:85%; text-align:left;">'
                '    <div style="font-size:0.8rem; color:#bbbbbb; margin-bottom:0.15rem;">'
                '      🤖 Помічник:'
                '    </div>'
                '    <div style="padding:0.45rem 0.7rem; border-radius:12px; '
                '                border-bottom-left-radius:0; '
                '                background-color:#1e3a5f;">'
                f'      {a_html}'
                '    </div>'
                '  </div>'
                '</div>'
            )

    if not chat_history_html:
        inner = "<span style='color:#888;'>Ще немає повідомлень.</span>"
    else:
        inner = chat_history_html

    outer_html = (
        f'<div style="height:{chat_height}px; overflow-y:auto; '
        'padding:0.75rem 0.5rem; border:1px solid #444; '
        'border-radius:8px; background-color:#0e1117;">'
        f'{inner}'
        '</div>'
    )
    return outer_html



def main():
    st.set_page_config(page_title="Юридичний помічник", layout="wide")

    if "app_state" not in st.session_state:
        st.session_state.app_state = initial_state()
    if "status_msg" not in st.session_state:
        st.session_state.status_msg = ""
    if "context_display" not in st.session_state:
        st.session_state.context_display = "Контекст ще не завантажено."
    if "ocr_preview_text" not in st.session_state:
        st.session_state.ocr_preview_text = ""

    with st.sidebar:
        st.header("Документи")

        base_files = st.file_uploader(
            "Завантажте документи, фото або скани",
            accept_multiple_files=True,
            help="Підтримуються TXT, MD, PDF, DOCX, зображення.",
        )

        if st.button("Оновити контекст з файлів"):
            if base_files:
                file_paths = save_uploaded_files(base_files, subdir="base")
                state, ctx_display, status = process_files(file_paths, st.session_state.app_state)
            else:
                state, ctx_display, status = process_files([], st.session_state.app_state)
            st.session_state.app_state = state
            st.session_state.context_display = ctx_display
            st.session_state.status_msg = status

        st.markdown("---")
        st.subheader("OCR з PDF та зображень")

        ocr_files = st.file_uploader(
            "Додайте файли для OCR (PDF або зображення)",
            accept_multiple_files=True,
            type=["pdf", "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif"],
        )

        if st.button("Запустити OCR (демо)"):
            if ocr_files:
                ocr_paths = save_uploaded_files(ocr_files, subdir="ocr")
                ocr_path = Path(ocr_paths[0])  # take first file, same as your logic
                start_async_ocr(ocr_path, st.session_state.app_state)
                st.session_state.status_msg = f"OCR запущено для {ocr_path.name}"
            else:
                st.session_state.status_msg = "Немає файлів для OCR"


                st.markdown(
                    "_Цей OCR блок поки що використовує демо-функцію. Після інтеграції з реальним сервісом тут "
                    "відобразиться розпізнаний текст._"
                )

        # Auto-refresh while OCR runs
        if st.session_state.app_state.get("ocr_status") == "running":
            st.info("OCR виконується… будь ласка, зачекайте.")
            st.rerun()
        elif st.session_state.app_state.get("ocr_status") == "done":
            st.success("OCR завершено!")
        elif st.session_state.app_state.get("ocr_status") == "failed":
            st.error(st.session_state.app_state.get("ocr_result"))


    col_ctx, col_chat = st.columns([1, 2])

    with col_ctx:
        st.title("Юридичний помічник")
        st.markdown(
            "Завантажте допоміжні матеріали й спілкуйтеся з AI-асистентом для юридичних досліджень.\n\n"
            "**Увага:** асистент надає лише інформаційні рекомендації й **не замінює адвоката**."
        )

        st.subheader("Поточний контекст")
        st.markdown(st.session_state.context_display)
        base_anon_stats = st.session_state.app_state.get("base_anon_stats") or []
        if base_anon_stats:
            st.markdown("**Анонімізація файлів (NER):** " + "; ".join(base_anon_stats))

        st.subheader("Чернетка тексту після OCR")
        
        st.session_state.ocr_preview_text_box = st.session_state.app_state.get("ocr_result", "")
        st.text_area(
            "OCR текст",
            height=200,
            key="ocr_preview_text_box",
        )
        anon_stats = st.session_state.app_state.get("ocr_anon_stats") or []
        if anon_stats:
            st.markdown("**Анонімізація (NER):** " + "; ".join(anon_stats))


    with col_chat:
        st.subheader("Розмова з Юридичним помічником")

        if st.button("Очистити розмову"):
            new_state, status, ctx_display = clear_conversation()
            st.session_state.app_state = new_state
            st.session_state.status_msg = status
            st.session_state.context_display = ctx_display
            st.session_state.ocr_preview_text = ""
            st.rerun()

        use_rag_toggle = st.checkbox(
            "Використовувати RAG (ККУ + судова практика)",
            value=st.session_state.app_state.get("use_rag", False),
            help="Увімкніть, щоб відповідати на основі бази знань RAG про ККУ та рішення ВС.",
        )
        st.session_state.app_state["use_rag"] = use_rag_toggle

        chat_container = st.container()
        user_message = st.chat_input("Опишіть свою ситуацію або запитання...")

        with chat_container:
            chat_height = 600
            history = st.session_state.app_state.get("history", [])
            chat_placeholder = st.empty()


            if user_message:
                pending_pairs: List[Tuple[str, Optional[str]]] = history + [
                    (user_message, None)
                ]
                pending_html = build_chat_html(pending_pairs, chat_height=chat_height)
                chat_placeholder.markdown(pending_html, unsafe_allow_html=True)
            else:
                current_html = build_chat_html(
                    [(u, a) for (u, a) in history], chat_height=chat_height
                )
                chat_placeholder.markdown(current_html, unsafe_allow_html=True)

            if user_message:
                state, status = generate_response(user_message, st.session_state.app_state)
                st.session_state.app_state = state
                st.session_state.status_msg = status

                final_history = st.session_state.app_state.get("history", [])
                final_html = build_chat_html(
                    [(u, a) for (u, a) in final_history], chat_height=chat_height
                )
                chat_placeholder.markdown(final_html, unsafe_allow_html=True)


    if st.session_state.status_msg:
        st.info(st.session_state.status_msg)


if __name__ == "__main__":
    main()
