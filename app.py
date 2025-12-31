"""
STREAMLIT UI (app.py)
This file is the user interface, linking the user to the RAG engine.
"""

import streamlit as st
from rag_engine import run_rag_pipeline
from conversation_manager import create_conversation_chain
from model_manager import get_model_manager

# --- Page Configuration ---
st.set_page_config(page_title="AI Law Firm Site Generator", layout="wide")

# --- Session State Initialization ---
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = create_conversation_chain()
if "conversation_mode" not in st.session_state:
    st.session_state.conversation_mode = True
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- Helper Functions ---
def render_html_preview(html_content: str) -> None:
    """Display HTML preview in iframe."""
    cleaned = html_content.replace("\\n", "\n")
    full_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body style="margin:0;padding:1rem;background:#f9fafb;">
    <div style="max-width:1200px;margin:0 auto;">{cleaned}</div>
</body>
</html>"""
    st.components.v1.html(full_page, height=600, scrolling=True)


def render_proposals(result: dict) -> None:
    """Display design proposals or error messages."""
    if result.get("is_greeting"):
        st.info(result["message"])
    elif result.get("is_dangerous"):
        st.error(f"🚫 {result['error']}")
    elif "error" in result:
        st.error(f"❌ {result['error']}")
    elif "message" in result:
        st.warning(result["message"])
    elif "proposals" in result:
        proposals = result["proposals"]
        st.success(f"✅ Generated {len(proposals)} design proposals!")

        for tab, proposal in zip(st.tabs([f"🎨 {p['design_name']}" for p in proposals]), proposals):
            with tab:
                st.markdown(f"**📝 {proposal['narrative_summary']}**")
                st.divider()

                # Component IDs
                col1, col2, col3 = st.columns(3)
                col1.metric("Hero", proposal["hero_id"])
                col2.metric("Services", proposal["services_id"])
                col3.metric("Contact", proposal["contact_id"])

                # HTML Preview
                if proposal.get("full_html"):
                    with st.expander("👁️ View Live Preview", expanded=True):
                        render_html_preview(proposal["full_html"])


# --- Header ---
st.title("⚖️ AI Law Firm Site Designer")
st.caption("Transform your vision into professional law firm websites")

# --- Controls ---
col1, col2 = st.columns([4, 1])
with col1:
    st.session_state.conversation_mode = st.checkbox(
        "💬 Conversation Mode",
        value=st.session_state.conversation_mode,
        help="Enable follow-up questions that reference previous designs"
    )
with col2:
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.conversation_chain.clear()
        st.session_state.chat_history = []
        st.session_state.prompt_history = []
        st.rerun()

st.divider()

# --- Chat History ---
if st.session_state.conversation_mode and st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                render_proposals(msg["content"])

# --- Input Section ---
with st.form("chat_form", clear_on_submit=True):
    user_prompt = st.text_input(
        "Your message:",
        placeholder="e.g., Create a modern website for criminal defense attorney...",
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button("Send 🚀", use_container_width=True)

# --- Handle Submission ---
if submit_button and user_prompt:
    st.session_state.prompt_history.append(user_prompt)

    if st.session_state.conversation_mode:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    with st.spinner("🔍 Processing..."):
        conversation_chain = st.session_state.conversation_chain if st.session_state.conversation_mode else None
        result = run_rag_pipeline(user_prompt, conversation_chain)

    if st.session_state.conversation_mode:
        st.session_state.chat_history.append({"role": "assistant", "content": result})
        st.rerun()
    else:
        with st.chat_message("assistant"):
            render_proposals(result)

# --- Sidebar ---
with st.sidebar:
    st.header("📜 Recent Prompts")
    if st.session_state.prompt_history:
        for i, prompt in enumerate(reversed(st.session_state.prompt_history[-5:]), 1):
            with st.container():
                st.caption(f"#{i}")
                st.text(prompt[:60] + ("..." if len(prompt) > 60 else ""))
    else:
        st.info("No prompts yet")

    st.divider()

    st.header("🤖 AI Model")
    model_info = get_model_manager().get_model_info()
    st.text(f"{model_info['name']}")
    st.caption("Auto-switch enabled")

    st.divider()

    st.header("ℹ️ How It Works")
    st.markdown("""
1. **Safety Check** - Filters harmful inputs
2. **Validate** - Law firm relevance
3. **AI Search** - Find best components
4. **Generate** - 3 unique proposals
5. **Preview** - Live HTML mockups
""")
