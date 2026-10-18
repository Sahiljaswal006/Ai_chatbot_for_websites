import streamlit as st

from chatbot import (
    scrape_website,
    read_llm_txt,
    ask,
    API_KEY
)

from rag import build_vector_store

st.set_page_config(
    page_title="TechAgastyaan Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 TechAgastyaan Assistant")
st.caption("RAG Powered Chatbot")

if not API_KEY:
    st.error("GROQ_API_KEY not found")
    st.stop()


@st.cache_resource
def initialize_knowledge():

    llm_content = read_llm_txt()

    website_content = scrape_website()

    combined_text = f"""
    {llm_content}

    {website_content}
    """

    build_vector_store(combined_text)

    return website_content, llm_content


with st.spinner("Building RAG knowledge base..."):

    website_content, llm_content = initialize_knowledge()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input(
    "Ask anything about TechAgastyaan..."
)

if prompt:

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    history = st.session_state.messages[:-1]

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            answer = ask(
                prompt,
                history
            )

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


with st.sidebar:

    st.header("Knowledge Base")

    st.write(
        f"Website Content: {len(website_content):,} chars"
    )

    st.write(
        f"Extra Notes: {len(llm_content):,} chars"
    )

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()