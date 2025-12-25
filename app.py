"""
STREAMLIT UI (app.py)
This file is the user interface, linking the user to the RAG engine.
"""

import streamlit as st
from rag_engine import run_rag_pipeline, UNMATCHED_PROMPT_MESSAGE
from conversation_manager import create_conversation_chain
from model_manager import get_model_manager

# --- Page Configuration ---
st.set_page_config(page_title="AI Law Firm Site Generator", layout="wide")

# Custom CSS for better styling with chat interface
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    /* Fix input at bottom */
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #e5e7eb;
        z-index: 1000;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []
if "rag_result" not in st.session_state:
    st.session_state.rag_result = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = create_conversation_chain()
if "conversation_mode" not in st.session_state:
    st.session_state.conversation_mode = True  # Enable by default
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store chat messages for display
if "input_key" not in st.session_state:
    st.session_state.input_key = 0  # Key to force input field reset

# --- Header ---
st.title("⚖️ AI Law Firm Site Designer")
st.markdown("### Transform your vision into professional law firm websites")

# --- Conversation Mode Toggle ---
col_mode, col_clear, col_help = st.columns([3, 1, 1])
with col_mode:
    st.session_state.conversation_mode = st.checkbox(
        "💬 Conversation Mode",
        value=st.session_state.conversation_mode,
        help="Enable follow-up questions that reference previous designs"
    )
with col_clear:
    if st.button("🔄 Clear Chat"):
        st.session_state.conversation_chain.clear()
        st.session_state.rag_result = None
        st.session_state.chat_history = []
        st.session_state.prompt_history = []
        st.success("✅ Chat cleared!")
        st.rerun()
with col_help:
    with st.popover("💡 Help"):
        st.markdown("""
        **Example prompts:**
        - "Modern aggressive site for criminal defense"
        - "Traditional estate planning website"
        - "Professional corporate law firm"
        """)

st.markdown("---")

# --- Display Chat History ---
if st.session_state.conversation_mode and st.session_state.chat_history:
    st.markdown("### 💬 Conversation")

    # Chat messages container (scrollable)
    chat_container = st.container()

    with chat_container:
        # Display all messages
        for idx, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                # User message (right-aligned)
                st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 1.5rem;">
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    color: white; padding: 1rem 1.5rem; border-radius: 1rem 1rem 0 1rem;
                                    max-width: 70%; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                            <div style="font-size: 0.75rem; opacity: 0.9; margin-bottom: 0.25rem;">You</div>
                            <div style="font-size: 0.95rem; line-height: 1.5;">{msg["content"]}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            elif msg["role"] == "assistant":
                # Assistant message (left-aligned)
                result = msg["content"]

                # Handle different response types
                if result.get("is_greeting", False):
                    message_text = result["message"]
                    st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 1.5rem;">
                            <div style="background: #e3f2fd; color: #1976d2; padding: 1rem 1.5rem;
                                        border-radius: 1rem 1rem 1rem 0; max-width: 80%;
                                        box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 3px solid #2196f3;">
                                <div style="font-size: 0.75rem; color: #1565c0; margin-bottom: 0.25rem;">AI Assistant</div>
                                <div style="font-size: 0.95rem; line-height: 1.5; white-space: pre-wrap;">{message_text}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                elif result.get("is_dangerous", False):
                    st.error(f"🚫 {result['error']}")

                elif "error" in result:
                    st.error(f"❌ {result['error']}")

                elif "message" in result:
                    st.warning(result["message"])

                elif "proposals" in result:
                    # Show assistant header
                    num_proposals = len(result["proposals"])
                    st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 1rem;">
                            <div style="background: #f0f4f8; color: #1f2937; padding: 1rem 1.5rem;
                                        border-radius: 1rem 1rem 1rem 0; max-width: 80%;
                                        box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 3px solid #667eea;">
                                <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.25rem;">AI Assistant</div>
                                <div style="font-size: 0.95rem; line-height: 1.5;">✅ Generated {num_proposals} design proposals</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Display proposals in tabs
                    proposals = result["proposals"]
                    tab_labels = [f"🎨 {p['design_name']}" for p in proposals]
                    tabs = st.tabs(tab_labels)

                    for tab_idx, (tab, proposal) in enumerate(zip(tabs, proposals)):
                        with tab:
                            st.markdown(f"**📝 Description:** {proposal['narrative_summary']}")
                            st.markdown("---")

                            # Component IDs
                            cols = st.columns(3)
                            with cols[0]:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <h4 style="margin:0; font-size: 0.9rem;">Hero</h4>
                                        <p style="margin:0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{proposal["hero_id"]}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            with cols[1]:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <h4 style="margin:0; font-size: 0.9rem;">Services</h4>
                                        <p style="margin:0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{proposal["services_id"]}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            with cols[2]:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <h4 style="margin:0; font-size: 0.9rem;">Contact</h4>
                                        <p style="margin:0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{proposal["contact_id"]}</p>
                                    </div>
                                """, unsafe_allow_html=True)

                            # HTML Preview
                            if "full_html" in proposal and proposal["full_html"]:
                                with st.expander("👁️ View Live Preview", expanded=False):
                                    cleaned_html = proposal["full_html"].replace("\\n", "\n")
                                    full_page = f"""
                                    <!DOCTYPE html>
                                    <html lang="en">
                                    <head>
                                        <meta charset="UTF-8">
                                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                        <script src="https://cdn.tailwindcss.com"></script>
                                        <style>
                                            body {{
                                                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                                                margin: 0; padding: 1rem; background: #f9fafb;
                                            }}
                                            .container {{ max-width: 1200px; margin: 0 auto; }}
                                        </style>
                                    </head>
                                    <body>
                                        <div class="container">{cleaned_html}</div>
                                    </body>
                                    </html>
                                    """
                                    st.components.v1.html(full_page, height=600, scrolling=True)

                    st.markdown("<br>", unsafe_allow_html=True)

elif st.session_state.rag_result:
    # Non-conversation mode - show single result
    result = st.session_state.rag_result

    if result.get("is_greeting", False):
        st.info(result["message"])
    elif result.get("is_dangerous", False):
        st.error(f"🚫 {result['error']}")
    elif "error" in result:
        st.error(f"❌ {result['error']}")
    elif "message" in result:
        st.warning(result["message"])
    elif "proposals" in result:
        proposals = result["proposals"]
        if proposals:
            st.success(f"✅ Generated {len(proposals)} design proposals!")

            tab_labels = [f"🎨 {p['design_name']}" for p in proposals]
            tabs = st.tabs(tab_labels)

            for tab, proposal in zip(tabs, proposals):
                with tab:
                    st.markdown(f"**📝 Description:** {proposal['narrative_summary']}")
                    st.markdown("---")

                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="margin:0; font-size: 0.9rem;">Hero</h4>
                                <p style="margin:0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{proposal["hero_id"]}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="margin:0; font-size: 0.9rem;">Services</h4>
                                <p style="margin:0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{proposal["services_id"]}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    with cols[2]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="margin:0; font-size: 0.9rem;">Contact</h4>
                                <p style="margin:0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{proposal["contact_id"]}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    if "full_html" in proposal and proposal["full_html"]:
                        with st.expander("👁️ View Live Preview", expanded=True):
                            cleaned_html = proposal["full_html"].replace("\\n", "\n")
                            full_page = f"""
                            <!DOCTYPE html>
                            <html lang="en">
                            <head>
                                <meta charset="UTF-8">
                                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                <script src="https://cdn.tailwindcss.com"></script>
                                <style>
                                    body {{
                                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                                        margin: 0; padding: 1rem; background: #f9fafb;
                                    }}
                                    .container {{ max-width: 1200px; margin: 0 auto; }}
                                </style>
                            </head>
                            <body>
                                <div class="container">{cleaned_html}</div>
                            </body>
                            </html>
                            """
                            st.components.v1.html(full_page, height=600, scrolling=True)

# --- Input Section at Bottom (Fixed) ---
st.markdown("---")
st.markdown("### 💬 Ask a question")

# Create a form to handle submission
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])

    with col1:
        user_prompt = st.text_input(
            "Your message:",
            placeholder="e.g., Create a modern website for criminal defense attorney...",
            key=f"user_input_{st.session_state.input_key}",
            label_visibility="collapsed"
        )

    with col2:
        submit_button = st.form_submit_button("Send 🚀", use_container_width=True)

# Handle form submission
if submit_button and user_prompt:
    # Increment key to reset input
    st.session_state.input_key += 1

    # Store prompt
    st.session_state.prompt_history.append(user_prompt)

    # Add user message to chat history
    if st.session_state.conversation_mode:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_prompt
        })

    # Run pipeline
    with st.spinner("🔍 Processing..."):
        conversation_chain = st.session_state.conversation_chain if st.session_state.conversation_mode else None
        st.session_state.rag_result = run_rag_pipeline(user_prompt, conversation_chain)

    # Add assistant response to chat history
    if st.session_state.conversation_mode and st.session_state.rag_result:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": st.session_state.rag_result
        })

    # Rerun to show new messages
    st.rerun()

# --- Sidebar ---
with st.sidebar:
    # Conversation stats
    if st.session_state.conversation_mode and st.session_state.conversation_chain.has_previous_context():
        st.markdown("### 💬 Stats")
        stats = st.session_state.conversation_chain.get_stats()
        st.markdown(f"""
        - **Turns:** {stats['total_turns']}
        - **Messages:** {stats['total_turns'] * 2}
        """)
        st.markdown("---")

    # Recent prompts
    st.markdown("### 📜 Recent")
    if st.session_state.prompt_history:
        for i, prompt in enumerate(reversed(st.session_state.prompt_history[-5:]), 1):
            if prompt:
                st.markdown(f"""
                    <div style="background: white; padding: 0.75rem; border-radius: 0.5rem;
                                margin-bottom: 0.75rem; border-left: 3px solid #667eea;">
                        <small style="color: #6b7280;">#{i}</small>
                        <p style="margin: 0.25rem 0 0 0; color: #1f2937; font-size: 0.875rem;">{prompt[:60]}{"..." if len(prompt) > 60 else ""}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No prompts yet")

    st.markdown("---")

    # Model info
    st.markdown("### 🤖 AI Model")
    model_manager = get_model_manager()
    model_info = model_manager.get_model_info()
    stats = model_manager.get_stats()

    st.markdown(f"""
    **Current:** {model_info['name']}
    **Available:** {stats['active_models']} models
    **Auto-switch:** Enabled
    """)

    st.markdown("---")

    # How it works
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
    1. **Safety Check** - Filters harmful inputs
    2. **Validate** - Law firm relevance
    3. **AI Search** - Find best components
    4. **Generate** - 3 unique proposals
    5. **Preview** - Live HTML mockups
    """)
