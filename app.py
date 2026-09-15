import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# SQLITE WORKAROUND
# Needed for Chroma on some Hugging Face Spaces environments
# ============================================================
try:
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass


import gradio as gr
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("OpenAI API key not found in .env")
    sys.exit(1)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
)

# ============================================================
# CONFIG
# ============================================================

DOCS_DIR = "data"
DB_DIR = "./chroma_db"

WELCOME_MESSAGE = (
    "Hello, I'm Isabelle Bidou's Online Minister. You may ask me questions "
    "about my experience, skills, availability, eligibility and hobbies, "
    "etc. You can chat with me in multiple languages."
)


# ============================================================
# EMBEDDINGS
# ============================================================

print("Loading embedding model...", flush=True)
if not OPENAI_API_KEY:
    print("The API key is missing. Please configure API_KEY in your Space's Secrets.")
    sys.exit(1) 

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.", flush=True)


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_documents():
    """Load TXT, PDF and DOCX documents from the data directory."""

    os.makedirs(DOCS_DIR, exist_ok=True)

    documents = []

    files = os.listdir(DOCS_DIR)

    if not files:
        print(f"No files found in '{DOCS_DIR}/'", flush=True)

    for filename in files:

        file_path = os.path.join(DOCS_DIR, filename)

        if not os.path.isfile(file_path):
            continue

        # -------------------------
        # TXT
        # -------------------------
        if filename.lower().endswith(".txt"):
            print(f"Loading text file: {filename}", flush=True)

            try:
                with open(
                    file_path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    text = f.read()

                if text.strip():
                    from langchain_core.documents import Document

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename
                            },
                        )
                    )

            except Exception as e:
                print(
                    f"Error loading {filename}: {e}",
                    flush=True,
                )

        # -------------------------
        # PDF
        # -------------------------
        elif filename.lower().endswith(".pdf"):
            print(f"Loading PDF: {filename}", flush=True)
            import pypdf
            reader = pypdf.PdfReader(file_path)

            try:
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        from langchain_core.documents import Document
                        documents.append(Document(page_content=text, metadata={"source": filename, "page": i}))

            except Exception as e:
                print(
                    f"Error loading {filename}: {e}",
                    flush=True,
                )

        # -------------------------
        # DOCX
        # -------------------------
        elif filename.lower().endswith(".docx"):
            print(
                f"Loading Word document: {filename}",
                flush=True,
            )

            try:
                import docx2txt
                text = docx2txt.process(file_path)
                if text.strip():
                    from langchain_core.documents import Document
                    documents.append(Document(page_content=text, metadata={"source": filename}))

            except Exception as e:
                print(
                    f"Error loading {filename}: {e}",
                    flush=True,
                )

    print(
        f"Loaded {len(documents)} documents.",
        flush=True,
    )

    return documents


# ============================================================
# VECTOR DATABASE
# ============================================================

def build_vectorstore():
    """
    Load the existing Chroma database if it exists.
    Otherwise create it from documents in data/.
    """

    # --------------------------------------------------------
    # EXISTING DATABASE
    # --------------------------------------------------------

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print(
            "Existing Chroma database found. Loading it...",
            flush=True,
        )

        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )

        print(
            "Chroma database loaded.",
            flush=True,
        )

        return vectorstore

    # --------------------------------------------------------
    # BUILD DATABASE
    # --------------------------------------------------------

    print(
        "No existing Chroma database found.",
        flush=True,
    )

    print(
        "Building vector database from documents...",
        flush=True,
    )

    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "No PDF or DOCX documents were found in "
            f"'{DOCS_DIR}/'. Please add at least one document."
        )

    # --------------------------------------------------------
    # SPLIT DOCUMENTS
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    print(
        f"Created {len(chunks)} text chunks.",
        flush=True,
    )

    # --------------------------------------------------------
    # CREATE CHROMA
    # --------------------------------------------------------

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    print(
        "Chroma database created successfully.",
        flush=True,
    )

    return vectorstore


# ============================================================
# INITIALIZE VECTORSTORE
# ============================================================

try:
    vectorstore = build_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

except Exception as e:
    print(
        f"ERROR INITIALIZING VECTOR DATABASE: {e}",
        flush=True,
    )

    vectorstore = None
    retriever = None


# ============================================================
# CHAT HISTORY
# ============================================================

def format_history(history):
    """Convert Gradio chat history into readable text."""

    if not history:
        return ""

    lines = []

    for message in history[-8:]:

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role and content:
            lines.append(
                f"{role}: {content}"
            )

    return "\n".join(lines)


# ============================================================
# QUESTION ANSWERING
# ============================================================

def answer_question(query, history):

    if history is None:
        history = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ]

    if not query or not query.strip():
        return "", history

    # --------------------------------------------------------
    # Check vector database
    # --------------------------------------------------------

    if retriever is None:
        answer = (
            "I'm sorry, my knowledge base is currently "
            "unavailable. Please try again later."
        )

        history.append(
            {
                "role": "user",
                "content": query,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return "", history

    try:

        # ----------------------------------------------------
        # Retrieve relevant documents
        # ----------------------------------------------------

        docs = retriever.invoke(query)

        context_parts = []

        for doc in docs:
            if doc.page_content:
                context_parts.append(
                    doc.page_content
                )

        context = "\n\n".join(context_parts)

        # ----------------------------------------------------
        # Conversation history
        # ----------------------------------------------------

        history_text = format_history(history)

        # ----------------------------------------------------
        # System prompt
        # ----------------------------------------------------

        system_prompt = """
You are Isabelle Bidou's personal recruitment assistant.

Your job is to answer questions about Isabelle Bidou,
including:

- professional experience
- skills
- education
- languages
- hobbies
- availability
- eligibility
- career
- personal interests relevant to recruitment

IMPORTANT RULES:

1. Use ONLY the information contained in the provided context.
2. Do not invent facts about Isabelle.
3. If the answer is not contained in the context, say:
   "I'm not sure about that."
4. Speak as Isabelle using "I", "my", and "me".
5. Be helpful, natural and professional.
6. Answer in the same language as the user's question whenever possible.
7. Keep answers concise unless the user asks for more detail.
"""

        # ----------------------------------------------------
        # User prompt
        # ----------------------------------------------------

        user_prompt = f"""
Previous conversation:
{history_text}

Relevant information about Isabelle:
{context}

Current question:
{query}

Answer as Isabelle:
"""

        answer  = llm.invoke(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        )
        if hasattr(answer, 'content'):
            final_text = answer.content
        else:
            final_text = str(answer)

    except Exception as e:

        print(
            f"ERROR ANSWERING QUESTION: {e}",
            flush=True,
        )

        answer = (
            "I'm sorry, something went wrong while "
            "processing your question."
        )

    # --------------------------------------------------------
    # Update conversation history
    # --------------------------------------------------------

    history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": final_text,
        }
    )

    return "", history


# ============================================================
# CLEAR CHAT
# ============================================================

def clear_chat():
    return [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]


# ============================================================
# AVATARS
# ============================================================

guest_img = os.path.join(
    DOCS_DIR,
    "Guest.jpg",
)

isabelle_img = os.path.join(
    DOCS_DIR,
    "isabelle_bidou.png",
)

avatars = None

if (
    os.path.exists(guest_img)
    and os.path.exists(isabelle_img)
):
    avatars = [
        guest_img,
        isabelle_img,
    ]


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="Isabelle Bidou's Personal Minister"
) as demo:

    gr.Markdown(
        "# Isabelle Bidou's Personal Minister"
    )

    chatbot = gr.Chatbot(
        value=[
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ],
        avatar_images=avatars,
        height=500,
    )

    msg = gr.Textbox(
        placeholder="Ask a question...",
        label="Your question",
        lines=2,
    )

    with gr.Row():

        send = gr.Button(
            "Send",
            variant="primary",
        )

        clear = gr.Button(
            "Clear",
        )

    # Press Enter
    msg.submit(
        answer_question,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
    )

    # Click Send
    send.click(
        answer_question,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
    )

    # Clear
    clear.click(
        clear_chat,
        inputs=None,
        outputs=chatbot,
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv("PORT", "7860")
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
