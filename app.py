

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import get_rag_response


# Page configuration
st.set_page_config(
    page_title="Advanced Intent-Based RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# Custom styling
st.markdown("""
<style>
.stApp {
    background: #f5f6fc;
}

.main-title {
    color: #4338ca;
    font-size: 30px;
    font-weight: bold;
}

.subtitle {
    color: #64748b;
    font-size: 15px;
}

.stChatMessage {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("🤖 RAG Assistant")
    st.markdown("---")

    st.subheader("About")
    st.write(
    "Advanced Intent-Based RAG Chatbot "
    "using LangChain, Groq and Supabase."
)
    

    st.markdown("---")
    st.subheader("Features")

    st.write("🧠 Intent Classification")
    st.write("📚 PDF Document Retrieval")
    st.write("💬 Conversation History")
    st.write("⚡ AI-powered Responses")

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Main heading
st.markdown(
    '<div class="main-title">'
    'Advanced Intent-Based RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your company policy document.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("intent"):
            st.caption("Detected Intent: " + message["intent"])


# User input
user_input = st.chat_input("Ask your question here...")


if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare conversation history
    chat_history = []

    for message in st.session_state.messages:

        if message["role"] == "user":
            chat_history.append(
                HumanMessage(content=message["content"])
            )

        else:
            chat_history.append(
                AIMessage(content=message["content"])
            )

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                response, intent, sources = get_rag_response(
                    user_input,
                    chat_history
                )

                st.markdown(response)
                st.caption("Detected Intent: " + intent)

                # Display document sources
                if sources:

                    with st.expander("View document sources"):

                        for source in sources:
                            st.write(
                                source.get(
                                    "source",
                                    "Company Policy PDF"
                                )
                            )

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "intent": intent,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}")
