import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

import gradio as gr
from openai import OpenAI

# Default configuration values can be overridden with environment variables.
DEFAULT_BASE_URL = "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"
DEFAULT_MODEL = "openai.gpt-oss-120b-1:0"
MAX_CONTEXT_CHARS = 12000
MAX_PER_FILE_CHARS = 4000
TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"}


def readable_size(path: Path) -> str:
    size_kb = path.stat().st_size / 1024
    return f"{size_kb:.1f} KB"


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Create a cached OpenAI client configured for the Legal Helper app."""
    api_key = os.getenv("BEDROCK_KEY")
    if not api_key:
        raise RuntimeError(
            "BEDROCK_KEY environment variable must be set before launching the app."
        )
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(base_url=base_url, api_key=api_key)


def read_text_file(path: Path) -> Tuple[str, str]:
    """Read UTF-8-ish text files with graceful fallback."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text, f"text extracted ({readable_size(path)})"
    except Exception as exc:
        return "", f"unable to read text file: {exc}"


def read_pdf(path: Path) -> Tuple[str, str]:
    """Extract text from PDFs when pypdf is available."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", (
            f"PDF uploaded ({readable_size(path)}); install 'pypdf' to auto-extract text."
        )

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        pages_info = f"{len(reader.pages)} page{'s' if len(reader.pages) != 1 else ''}"
        return text, f"text extracted from PDF ({pages_info}, {readable_size(path)})"
    except Exception as exc:
        return "", f"failed to parse PDF: {exc}"


def read_docx(path: Path) -> Tuple[str, str]:
    """Extract text from DOCX files when python-docx is available."""
    try:
        import docx  # type: ignore
    except ImportError:
        return "", (
            f"DOCX uploaded ({readable_size(path)}); install 'python-docx' to auto-extract text."
        )

    try:
        document = docx.Document(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
        return text, f"text extracted from DOCX ({readable_size(path)})"
    except Exception as exc:
        return "", f"failed to parse DOCX: {exc}"


def extract_file_context(path: Path) -> Tuple[str, str]:
    """Return extracted text (if any) and a human-friendly summary."""
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix in IMAGE_EXTENSIONS:
        return "", (
            f"image uploaded ({readable_size(path)}); add a prompt describing its contents."
        )
    return "", f"{suffix or 'file'} uploaded ({readable_size(path)}); extraction not supported."


def summarize_context(sections: List[str], summaries: List[str]) -> Tuple[str, str]:
    """Combine extracted text and create a Markdown-friendly summary."""
    combined_context = "\n\n".join(sections).strip()
    if len(combined_context) > MAX_CONTEXT_CHARS:
        combined_context = combined_context[:MAX_CONTEXT_CHARS] + "\n...[context truncated]"

    summary_lines = "\n".join(f"- {line}" for line in summaries)
    context_md = (
        "### Loaded Context\n" + summary_lines if summary_lines else "No context loaded yet."
    )
    return combined_context, context_md


def process_files(
    uploaded_files: Optional[List[str]], state: Optional[dict]
) -> Tuple[dict, str, str]:
    """Handle file uploads, extract available text, and update session state."""
    state = state or {"context": "", "history": [], "files": []}
    if not uploaded_files:
        state.update({"context": "", "files": []})
        return state, "No context loaded yet.", "Context cleared."

    extracted_sections: List[str] = []
    summary_lines: List[str] = []
    for file_path in uploaded_files:
        path = Path(file_path)
        text, summary = extract_file_context(path)
        if text:
            snippet = text[:MAX_PER_FILE_CHARS]
            if len(text) > MAX_PER_FILE_CHARS:
                snippet += "\n...[truncated]"
            extracted_sections.append(f"Source: {path.name}\n{snippet}")
        summary_lines.append(f"**{path.name}** — {summary}")

    combined_context, context_md = summarize_context(extracted_sections, summary_lines)
    state.update(
        {"context": combined_context, "files": [Path(p).name for p in uploaded_files]}
    )
    status = (
        f"Loaded {len(uploaded_files)} file{'s' if len(uploaded_files) != 1 else ''}."
    )
    return state, context_md, status


def build_messages(state: dict, user_message: str) -> List[dict]:
    """Prepare the message list for the OpenAI chat completion call."""
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
    """Request a response from the configured LLM."""
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
    return (content or "").strip()


def generate_response(
    user_message: str, chat_history: List[Tuple[str, str]], state: Optional[dict]
) -> Tuple[List[Tuple[str, str]], dict, Any, str]:
    """Send the user's prompt to the LLM and return the updated conversation."""
    user_message = (user_message or "").strip()
    if not user_message:
        return chat_history, state or {}, gr.update(value=user_message), "Please enter a question."

    chat_history = chat_history or []
    state = state or {"context": "", "history": [], "files": []}

    chat_history.append((user_message, ""))

    try:
        messages = build_messages(state, user_message)
        assistant_response = fetch_completion(messages)
    except Exception as exc:  # pragma: no cover - network failure path
        chat_history[-1] = (user_message, "⚠️ Unable to reach the language model.")
        return chat_history, state, gr.update(value=user_message), f"Model call failed: {exc}"

    chat_history[-1] = (user_message, assistant_response)
    state_history = state.get("history", [])
    state_history.append((user_message, assistant_response))
    state["history"] = state_history
    return chat_history, state, gr.update(value=""), "Response generated."


def clear_conversation(state: Optional[dict]) -> Tuple[List[Tuple[str, str]], dict, Any, str, str]:
    """Reset conversation history and uploaded context."""
    cleared_state = {"context": "", "history": [], "files": []}
    return [], cleared_state, gr.update(value=""), "Conversation cleared.", "No context loaded yet."


def build_interface() -> gr.Blocks:
    """Create the Gradio UI for the Legal Helper app."""
    layout_css = """
    #legal-helper {
        max-width: 800px;      /* was 600px */
        margin: 0 auto;
    }

    .gr-chatbot {
        height: 800px !important;  /* increased chat height */
    }
    """ 

    with gr.Blocks(title="Legal Helper", css=layout_css, elem_id="legal-helper") as demo:
        gr.Markdown(
            "# Legal Helper\n"
            "Upload supporting materials and chat with an AI legal research assistant.\n"
            "The assistant provides informational guidance only and is not a substitute for a lawyer."
        )

        state = gr.State({"context": "", "history": [], "files": []})

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload documents, photos, or scans",
                    file_count="multiple",
                    type="filepath",
                )
            with gr.Column(scale=1):
                context_display = gr.Markdown("No context loaded yet.")

        chatbot = gr.Chatbot(label="Legal Helper Conversation", height=560)
        prompt = gr.Textbox(
            label="Ask a legal question (informational guidance only)",
            placeholder="Describe your situation or question...",
            lines=3,
        )

        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear Conversation")

        status = gr.Markdown("")

        file_input.upload(
            fn=process_files,
            inputs=[file_input, state],
            outputs=[state, context_display, status],
        )

        send_button.click(
            fn=generate_response,
            inputs=[prompt, chatbot, state],
            outputs=[chatbot, state, prompt, status],
        )

        prompt.submit(
            fn=generate_response,
            inputs=[prompt, chatbot, state],
            outputs=[chatbot, state, prompt, status],
        )

        clear_button.click(
            fn=clear_conversation,
            inputs=[state],
            outputs=[chatbot, state, prompt, status, context_display],
        )

    return demo


if __name__ == "__main__":  # pragma: no cover - Gradio launch path
    demo = build_interface()
    demo.launch(server_name="0.0.0.0")
