"""
STREAMLIT UI (app.py)
This file is the user interface, linking the user to the RAG engine.
"""

import streamlit as st
from rag_engine import run_rag_pipeline, UNMATCHED_PROMPT_MESSAGE

# --- Page Configuration ---
st.set_page_config(page_title="AI Law Firm Site Generator", layout="wide")

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
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
    .proposal-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("⚖️ AI Law Firm Site Designer")
st.markdown("### Transform your vision into professional law firm websites")
st.caption(
    "Describe the type of law practice and design style you want, and we'll generate 3 unique website proposals with live previews."
)
st.markdown("---")

# --- Session State Initialization ---
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []
if "rag_result" not in st.session_state:
    st.session_state.rag_result = None

# --- User Input ---
col1, col2 = st.columns([3, 1])
with col1:
    user_prompt = st.text_input(
        "Describe the desired website:",
        placeholder="e.g., Aggressive site for criminal defense with quick contact form.",
        key="user_prompt",
        label_visibility="collapsed",
    )
with col2:
    generate_button = st.button(
        "🚀 Generate Design", type="primary", disabled=not user_prompt
    )

with st.expander("💡 Need inspiration? See example prompts", expanded=False):
    st.markdown("""
    **Good examples:**
    - "Modern and aggressive site for criminal defense attorney"
    - "Traditional estate planning website with trustworthy design"
    - "Professional corporate law firm with clean, minimalist design"
    - "Conservative family law practice with warm, approachable tone"
    - "Fast-response immigration law site with multilingual support feel"
    
    **What to include:**
    - ✅ Type of law practice (criminal, estate, corporate, family, immigration, etc.)
    - ✅ Design tone (modern, traditional, aggressive, conservative, professional)
    - ✅ Target audience or special features
    
    **Won't work:**
    - ❌ General questions like "What is law?"
    - ❌ Non-legal businesses (restaurants, retail, etc.)
    - ❌ Unrelated topics
    """)

# --- Generate Button ---
if generate_button:
    with st.spinner("🔍 Validating your request..."):
        # Store prompt first
        st.session_state.prompt_history.append(user_prompt)

        # Run the pipeline
        st.session_state.rag_result = run_rag_pipeline(user_prompt)

        # Show success message if we got proposals
        if st.session_state.rag_result and "proposals" in st.session_state.rag_result:
            st.success("✅ Validation passed! Generating designs...")

# --- Display Results ---
if st.session_state.rag_result:
    result = st.session_state.rag_result

    # Handle error cases
    if "error" in result:
        st.error(f"❌ {result['error']}")
    # Handle unmatched prompt case (query validation failed)
    elif "message" in result:
        st.warning(result["message"])
        if "validation_reason" in result:
            st.info(f"**Reason:** {result['validation_reason']}")
        if "details" in result:
            st.info(f"**Details:** {result['details']}")
    # Handle successful response
    elif "proposals" in result:
        proposals = result["proposals"]

        if not proposals:
            st.warning("⚠️ No matching designs found. Please try a different prompt.")
        else:
            st.success(f"✅ Generated {len(proposals)} design proposals!")

            # Display each proposal
            for i, proposal in enumerate(proposals, 1):
                with st.expander(
                    f"🎨 Design Proposal {i}: {proposal['design_name']}",
                    expanded=(i == 1),
                ):
                    st.markdown(f"**📝 Description:** {proposal['narrative_summary']}")
                    st.markdown("---")

                    # Show component IDs in colored cards
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h4 style="margin:0; font-size: 0.9rem;">Hero Section</h4>
                                <p style="margin:0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{proposal["hero_id"]}</p>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with cols[1]:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h4 style="margin:0; font-size: 0.9rem;">Services Section</h4>
                                <p style="margin:0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{proposal["services_id"]}</p>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with cols[2]:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h4 style="margin:0; font-size: 0.9rem;">Contact Section</h4>
                                <p style="margin:0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{proposal["contact_id"]}</p>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Show HTML preview if available
                    if "full_html" in proposal and proposal["full_html"]:
                        # Clean up the HTML by replacing literal \n with actual newlines
                        cleaned_html = proposal["full_html"].replace("\\n", "\n")

                        # Wrap in proper HTML document with Tailwind CDN
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
                                    margin: 0;
                                    padding: 1rem;
                                    background: #f9fafb;
                                }}
                                .container {{
                                    max-width: 1200px;
                                    margin: 0 auto;
                                }}
                            </style>
                        </head>
                        <body>
                            <div class="container">
                                {cleaned_html}
                            </div>
                        </body>
                        </html>
                        """

                        st.markdown("### 👁️ Live Preview")
                        st.components.v1.html(full_page, height=700, scrolling=True)
                    else:
                        st.warning("⚠️ No preview available for this design.")
    else:
        st.error("❌ Unexpected response format from the RAG pipeline.")

# --- Display Prompt History ---
with st.sidebar:
    st.markdown("### 📜 Recent Prompts")
    st.markdown("---")

    if st.session_state.prompt_history:
        for i, prompt in enumerate(
            reversed(st.session_state.prompt_history[-5:]), 1
        ):  # Show only last 5
            if prompt:  # Only display non-empty prompts
                st.markdown(
                    f"""
                    <div style="background: white; padding: 0.75rem; border-radius: 0.5rem; 
                                margin-bottom: 0.75rem; border-left: 3px solid #667eea;">
                        <small style="color: #6b7280;">Prompt {i}</small>
                        <p style="margin: 0.25rem 0 0 0; color: #1f2937; font-size: 0.875rem;">{prompt[:100]}{"..." if len(prompt) > 100 else ""}</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No prompts yet. Start by entering a query above!")

    st.markdown("---")
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
    1. **Validate** - Checks if your query is law-firm related
    2. **Match** - Finds relevant design components
    3. **Generate** - Creates 3 unique proposals
    4. **Preview** - Shows live HTML mockups
    """)

    st.markdown("---")
    st.markdown("### 🎯 Best Results")
    st.markdown("""
    - Be specific about practice area
    - Mention desired tone/style
    - Include target audience
    - Use clear, descriptive language
    """)
