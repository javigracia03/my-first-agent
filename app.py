import re
from typing import Optional

import streamlit as st
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    # Prefer langchain_classic as requested.
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    # Backward-compatible fallback if langchain_classic isn't installed yet.
    from langchain.agents import AgentExecutor
    try:
        from langchain.agents import create_tool_calling_agent
    except ImportError:
        from langchain.agents.tool_calling_agent.base import create_tool_calling_agent


wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1200)


def _clean_query(text: str) -> str:
    return text.strip().strip('"').strip("'")


def _extract_compare_pair(query: str) -> Optional[tuple[str, str]]:
    q = query.strip()
    m = re.search(r"(?i)compare\s+(.+?)\s+and\s+(.+)", q)
    if m:
        return _clean_query(m.group(1)), _clean_query(m.group(2))

    m = re.search(r"(?i)(.+?)\s+vs\.?\s+(.+)", q)
    if m:
        return _clean_query(m.group(1)), _clean_query(m.group(2))

    return None


@tool
def wiki_summary(topic: str) -> str:
    """Return a short Wikipedia summary for a topic."""
    topic = _clean_query(topic)
    if not topic:
        return "Please provide a valid topic."

    try:
        return wiki.run(topic)
    except Exception as e:
        return f"Could not fetch summary for '{topic}': {e}"


@tool
def wiki_image(topic: str) -> str:
    """Return one Wikipedia image URL for a topic or 'No image found'."""
    topic = _clean_query(topic)
    if not topic:
        return "No image found"

    try:
        page = wiki.wiki_client.page(topic)
    except Exception:
        return "No image found"

    images = page.images or []
    if not images:
        return "No image found"

    for img in images:
        lower = img.lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
            return img

    return images[0] if images else "No image found"


@tool
def wiki_compare(a: str, b: str) -> str:
    """Compare two topics using short Wikipedia summaries and return markdown."""
    a = _clean_query(a)
    b = _clean_query(b)
    if not a or not b:
        return "Please provide two valid topics to compare."

    sa = wiki_summary.invoke({"topic": a})
    sb = wiki_summary.invoke({"topic": b})

    return (
        f"## Comparison: {a} vs {b}\n\n"
        f"### {a}\n{sa}\n\n"
        f"### {b}\n{sb}\n"
    )


def build_agent(api_key: str) -> AgentExecutor:
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    tools = [wiki_summary, wiki_image, wiki_compare]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Wikipedia multi-tool assistant. "
                "Always choose exactly one tool per user request. "
                "Rules: "
                "1) If query includes comparison intent (like 'compare X and Y' or 'X vs Y'), call wiki_compare. "
                "2) If query asks for photo/image/picture, call wiki_image. "
                "3) Otherwise call wiki_summary. "
                "Return concise markdown.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(model, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


def parse_compare_query(query: str) -> str:
    pair = _extract_compare_pair(query)
    if not pair:
        return query
    return f"Compare {pair[0]} and {pair[1]}"


def extract_first_url(text: str) -> Optional[str]:
    m = re.search(r"(https?://[^\s)]+)", text)
    return m.group(1) if m else None


def main() -> None:
    st.set_page_config(page_title="Wikipedia Multi-Tool Agent")
    st.title("Wikipedia Multi-Tool Agent")
    st.markdown(
        "**Things you can ask:**\n"
        "- \"Tell me about Ada Lovelace\"\n"
        "- \"Show me a picture of the Eiffel Tower\"\n"
        "- \"Compare Python and JavaScript\"\n"
        "- \"Messi vs Ronaldo\""
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

        executor = build_agent(api_key)

        normalized_query = parse_compare_query(user_query)
        result = executor.invoke({"input": normalized_query})
        final_answer = result.get("output", "No answer produced.")

        st.markdown(final_answer)

        image_url = extract_first_url(final_answer)
        if image_url and image_url != "No image found":
            st.image(image_url)


if __name__ == "__main__":
    main()
