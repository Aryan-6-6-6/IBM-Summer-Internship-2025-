import os
import fitz
from docx import Document
from openai import OpenAI, AuthenticationError
import gradio as gr
from dotenv import load_dotenv

# Load environment variables (for local development)
load_dotenv()  # looks for .env
load_dotenv("marnelife.env", override=False)  # also supports your existing file

api_key = os.environ.get("GITHUB_AI_TOKEN") or os.environ.get("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("Missing GitHub AI token. Put GITHUB_AI_TOKEN in .env or marnelife.env and restart the app.")

# Configuration
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=api_key
)

SYSTEM_PROMPT = """You are a warm, knowledgeable marine life assistant. 
You only answer questions about marine biology, underwater ecosystems, and water-based pets. 
If asked about other topics, kindly redirect the user to marine-related themes. 
Do not explicitly state that you are an AI."""

def extract_text(file_obj):
    if not file_obj:
        return ""

    try:
        path = file_obj.name
        if path.endswith(".pdf"):
            with fitz.open(path) as doc:
                return "\n".join(page.get_text() for page in doc)
        if path.endswith(".docx"):
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        if path.endswith((".txt", ".md", ".csv", ".json")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        # For images or other file types, keep the upload available without blocking the chat UI.
        return f"Uploaded file: {os.path.basename(path)}\nFile type: {os.path.splitext(path)[1].lower() or 'unknown'}"
    except Exception as e:
        return f"Error reading file: {e}"


def chat_interface(message, history, file):
    file_content = extract_text(file)
    if file_content:
        context = f"Uploaded context:\n{file_content}\n\nUser question: {message}"
    else:
        context = message

    try:
        stream = client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": context}],
            temperature=0.7,
            stream=True,
        )

        partial_message = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                partial_message += chunk.choices[0].delta.content
                yield partial_message

    except AuthenticationError:
        yield (
            "I’m unable to answer right now because the AI token is invalid or expired. "
            "Please update GITHUB_AI_TOKEN in marnelife.env and restart the app."
        )
    except Exception as exc:
        yield f"Sorry, the chatbot hit an error: {exc}"

# Build UI
with gr.Blocks() as demo:
    gr.Markdown("# 🌊 Marine Life AI Assistant")
    gr.Markdown("Upload a file or image if you want extra context, then ask your marine-life question.")
    file_upload = gr.File(
        label="Upload file or image (optional)",
        file_count="single"
    )
    gr.ChatInterface(
        fn=chat_interface,
        additional_inputs=[file_upload],
        title="Marine Life Assistant"
    )


demo.launch()