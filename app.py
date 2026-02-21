import re
from typing import Optional

import streamlit as st
from langchain.agents import create_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


# Wikipedia wrapper uses the `wikipedia` python package under the hood. :contentReference[oaicite:2]{index=2}
wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1200)


def clean(text: str) -> str:
    return text.strip().strip('"').strip("'")


def normalize_compare(query: str) -> str:
    q = query.strip()

    # "X vs Y"
    m = re.search(r"(?i)^\s*(.+?)\s+vs\.?\s+(.+?)\s*$", q)
    if m:
        return f"Compare {clean(m.group(1))} and {clean(m.group(2))}"

    # "compare X and Y"
    m = re.search(r"(?i)^\s*compare\s+(.+?)\s+and\s+(.+?)\s*$", q)
    if m:
        return f"Compare {clean(m.group(1))} and {clean(m.group(2))}"

    return q


def first_url(text: str) -> Optional[str]:
    m = re.search(r"(https?://[^\s)]+)", text)
    return m.group(1) if m else None


@tool
def wiki_summary(topic: str) -> str:
    """Short Wikipedia summary for a topic."""
    topic = clean(topic)
    if not topic:
        return "Please provide a valid topic."
    try:
        return wiki.run(topic)
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

    sa = wiki.run(a)
    sb = wiki.run(b)

    return (
        f"## {a} vs {b}\n\n"
        f"### {a}\n{sa}\n\n"
        f"### {b}\n{sb}\n"
    )


def extract_text(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No answer produced."

    last = messages[-1]
    content = getattr(last, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        joined = "".join(parts).strip()
        return joined or "No answer produced."
    return str(content) if content else "No answer produced."


def build_executor(api_key: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,   # langchain-google-genai supports api_key + env var. :contentReference[oaicite:3]{index=3}
        temperature=0,
    )

    tools = [wiki_summary, wiki_image, wiki_compare]

    system = (
        "You are a Wikipedia multi-tool assistant.\n"
        "You MUST choose exactly one tool per user request.\n"
        "Rules:\n"
        "1) If user intent is comparison (e.g., 'compare X and Y' or 'X vs Y') -> call wiki_compare.\n"
        "2) If user asks for an image/photo/picture -> call wiki_image.\n"
        "3) Otherwise -> call wiki_summary.\n"
        "Return concise markdown."
    )

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system,
        debug=False,
    )


def main() -> None:
    st.set_page_config(page_title="Wikipedia Multi-Tool Agent", layout="centered")
    st.title("Wikipedia Multi-Tool Agent")

    st.markdown(
        "- Tell me about Ada Lovelace\n"
        "- Show me a picture of the Eiffel Tower\n"
        "- Compare Python and JavaScript\n"
        "- Messi vs Ronaldo"
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

        executor = build_executor(api_key)
        normalized = normalize_compare(user_query)

        result = executor.invoke(
            {"messages": [{"role": "user", "content": normalized}]}
        )
        answer = extract_text(result)

        st.markdown(answer)

        url = first_url(answer)
        if url and url != "No image found":
            st.image(url)


if __name__ == "__main__":
    main()
