"""Streamlit UI: login, chat, sidebar, source citations."""
import streamlit as st
from dotenv import load_dotenv
import os

from config import USERS, ROLE_DISPLAY
from rag_chain import RAGPipeline

# Load env
load_dotenv()

st.set_page_config(page_title="Finzo NexusChat", page_icon="🏦")

def login_page():
    st.title("Finzo NexusChat Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username in USERS and USERS[username]["password"] == password:
                st.session_state["user"] = USERS[username]
                st.rerun()
            else:
                st.error("Invalid username or password")
                
    st.markdown("### Demo Credentials")
    st.markdown("""
    | Username | Role | Department |
    |---|---|---|
    | Lewis | CXO | All |
    | Max | HR | HR & General |
    | Kimi | Finance | Finance & General |
    | George | Marketing | Marketing & General |
    | Charles | Engineering | Engineering & General |
    | Lando | Employee | General |
    """)
    st.info("Password for all users: password123")

def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main_app():
    user = st.session_state["user"]
    role = user["role"]
    emoji, display_role = ROLE_DISPLAY[role]
    
    # Init RAG
    @st.cache_resource
    def get_pipeline():
        return RAGPipeline()
        
    try:
        pipeline = get_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize AI components. Check your API keys in .env. Error: {e}")
        return
        
    init_chat_history()
    
    # Sidebar
    with st.sidebar:
        st.title("🏦 Finzo NexusChat")
        st.markdown(f"**Name:** {user['name']}")
        st.markdown(f"**Title:** {user['title']}")
        st.markdown(f"**Role:** {emoji} {display_role}")
        
        if st.button("Logout"):
            del st.session_state["user"]
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()
            
        st.divider()
        st.markdown("### Recent Queries")
        recent_msgs = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
        for msg in recent_msgs[-5:]:
            st.caption(f"💬 {msg[:30]}...")

    # Main Chat Area
    st.title("Finzo NexusChat")
    
    # Role-specific greeting
    allowed_deps = "All data" if role == "cxo_level" else f"{display_role} and General data" if role != "employee" else "General data"
    st.info(f"Welcome {user['name']}, you have access to {allowed_deps}.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📄 Sources"):
                    for src in message["sources"]:
                        st.markdown(f"- {src['source']} (Tag: `{src['department']}`)")

    # Chat input
    if prompt := st.chat_input("Ask about Finzo internal knowledge..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and show response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = pipeline.query(prompt, role)
                    answer = result["answer"]
                    sources = result["sources"]
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("📄 Sources"):
                            for src in sources:
                                st.markdown(f"- {src['source']} (Tag: `{src['department']}`)")
                                
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

def app():
    if "user" not in st.session_state:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    app()
