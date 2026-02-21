import re
from typing import Optional

import streamlit as st
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1200)
SUMMARY_MAX_CHARS = 1400


def clean(text: str) -> str:
    return text.strip().strip('"').strip("'")


def first_url(text: str) -> Optional[str]:
    m = re.search(r"(https?://[^\s)]+)", text)
    return m.group(1) if m else None


def _strip_wrapper_labels(text: str) -> str:
    text = text.strip()
    summary_match = re.search(r"(?is)\bsummary:\s*(.+)$", text)
    if summary_match:
        return summary_match.group(1).strip()
    return text


def _trim_to_sentence(text: str, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars].rstrip()
    boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if boundary > 120:
        return cut[: boundary + 1].strip()
    return cut + "..."


def get_topic_summary(topic: str) -> str:
    topic = clean(topic)
    if not topic:
        return ""

    try:
        page = wiki.wiki_client.page(topic)
        direct = getattr(page, "summary", "") or ""
        direct = direct.strip()
        if direct:
            return _trim_to_sentence(direct)
    except Exception:
        pass

    raw = wiki.run(topic)
    cleaned = _strip_wrapper_labels(raw)
    return _trim_to_sentence(cleaned)


@tool
def wiki_summary(topic: str) -> str:
    """Short Wikipedia summary for a topic."""
    topic = clean(topic)
    if not topic:
        return "Please provide a valid topic."
    try:
        summary = get_topic_summary(topic)
        return summary or f"Could not find a summary for '{topic}'."
    except Exception as e:
        return f"Could not fetch summary for '{topic}': {e}"


@tool
def wiki_image(topic: str) -> str:
    """Return one Wikipedia image URL for a topic or 'No image found'."""
    topic = clean(topic)
    if not topic:
        return "No image found"

    try:
        page = wiki.wiki_client.page(topic)
        images = page.images or []
    except Exception:
        return "No image found"

    for img in images:
        low = img.lower()
        if low.endswith((".jpg", ".jpeg", ".png", ".webp")):
            return img

    return images[0] if images else "No image found"


@tool
def wiki_compare(a: str, b: str) -> str:
    """Compare two topics using short Wikipedia summaries (markdown)."""
    a, b = clean(a), clean(b)
    if not a or not b:
        return "Please provide two valid topics to compare."

    sa = get_topic_summary(a)
    sb = get_topic_summary(b)

    return (
        f"## {a} vs {b}\n\n"
        "Use the two summaries below to produce a brief comparison with:\n"
        "1. Similarities\n"
        "2. Key differences\n"
        "3. When to choose one over the other (if applicable)\n\n"
        f"### {a} summary\n{sa}\n\n"
        f"### {b} summary\n{sb}\n"
    )


SYSTEM_PROMPT = (
    "You are a Wikipedia multi-tool assistant.\n"
    "You MUST choose exactly one tool per user request.\n"
    "Rules:\n"
    "1) If user intent is comparison (e.g., 'compare X and Y' or 'X vs Y') -> call wiki_compare.\n"
    "2) If user asks for an image/photo/picture -> call wiki_image.\n"
    "3) Otherwise -> call wiki_summary.\n"
    "Always return complete sentences.\n"
    "Never output raw labels like 'Page:' or 'Summary:'.\n"
    "If comparison was requested, provide an actual comparison, not just two pasted summaries.\n"
    "Return concise markdown."
)


def build_agent(api_key: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0.4,
    )
    tools = [wiki_summary, wiki_image, wiki_compare]
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


def main() -> None:
    st.set_page_config(page_title="Wikipedia Multi-Tool Agent", layout="centered")
    st.title("Wikipedia Multi-Tool Agent")

    st.markdown(
        "- Tell me about Albert Einstein\n"
        "- Show me a picture of the Eiffel Tower\n"
        "- Lionel Messi vs Cristiano Ronaldo"
    )

    with st.sidebar:
        api_key = st.text_input("GOOGLE_API_KEY", type="password")

    user_query = st.text_input("Enter your question")

    if st.button("Run"):
        if not api_key:
            st.error("Please provide GOOGLE_API_KEY in the sidebar.")
            return
        if not user_query.strip():
            st.error("Please enter a query.")
            return

        agent = build_agent(api_key)
        result = agent.invoke({"messages": [HumanMessage(content=user_query)]})
        raw = result["messages"][-1].content
        if isinstance(raw, list):
            answer = " ".join(p.get("text", "") for p in raw if isinstance(p, dict))
        else:
            answer = str(raw)

        st.markdown(answer)

        url = first_url(answer)
        if url and url != "No image found":
            st.image(url)


if __name__ == "__main__":
    main()
