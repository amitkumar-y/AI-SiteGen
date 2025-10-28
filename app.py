# -----------------------------------------------------------------------------
# STREAMLIT UI (app.py)
# This file is the user interface, linking the user to the RAG engine.
# -----------------------------------------------------------------------------
import streamlit as st
from rag_engine import run_rag_pipeline, UNMATCHED_PROMPT_MESSAGE, DesignProposal

# --- 1. Page Configuration ---
# Sets up the basic look and feel of the app
st.set_page_config(
    page_title="AI Law Firm Site Generator",
    layout="wide"
)
st.title("⚖️ AI Law Firm Site Designer")
st.markdown("---")
st.info("Enter a prompt describing the type of law firm and the required design tone (e.g., 'Modern site for criminal defense' or 'Traditional site for estate planning').")

# --- 2. Input and Generation Logic ---

# Initialize session state to store previous results (important for persistence)
if 'proposals' not in st.session_state:
    st.session_state.proposals = None
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []

# Input field is mandatory (Check A.2)
user_prompt = st.text_input(
    "Describe the desired website:",
    placeholder="e.g., Aggressive site for criminal defense with quick contact form.",
    key="user_prompt"
)

# Button validation (Check A.3, A.4)
if st.button("Generate Design", type="primary", disabled=not user_prompt):
    # Only run the RAG pipeline if the prompt is not empty
    with st.spinner("Analyzing request and generating proposals..."):
        # Log the prompt for history (helpful for debugging/QA)
        st.session_state.prompt_history.append(user_prompt)
        
        # Call the core RAG logic (Check E.3: Latency starts here)
        result = run_rag_pipeline(user_prompt)
        
        # Store the result in session state
        st.session_state.proposals = result

# --- 3. Output Display (Handling Success, Deflection, and Errors) ---

if st.session_state.proposals is not None:
    proposals = st.session_state.proposals

    # Case 1: Deflection/Unmatched Request (Check D.3, B.5)
    if isinstance(proposals, str) and proposals == UNMATCHED_PROMPT_MESSAGE:
        st.warning(proposals)
        st.error("Please rephrase your request to focus on law firm type and design requirements.")

    # Case 2: API or Internal RAG Engine Error (Check E.1)
    elif isinstance(proposals, str):
        st.error(f"An internal error occurred: {proposals}")
        st.code("Check the console or Groq API key.", language="text")

    # Case 3: Successful Generation (Check C.1, C.4)
    elif isinstance(proposals, list) and all(isinstance(p, DesignProposal) for p in proposals):
        st.subheader("✅ Top Design Proposals Generated (RAG-Verified)")
        
        # Use columns to present the three options side-by-side
        cols = st.columns(len(proposals))
        
        for i, proposal in enumerate(proposals):
            with cols[i]:
                # Display the proposal name and summary
                st.markdown(f"### Option {i+1}: {proposal.design_name}")
                st.markdown(f"**Summary:** *{proposal.narrative_summary}*")
                
                # Use st.expander for component breakdown
                with st.expander("Component Breakdown"):
                    st.json({
                        "Hero ID": proposal.hero_id,
                        "Services ID": proposal.services_id,
                        "Contact ID": proposal.contact_id
                    })

                # Display the Mock HTML Preview
                st.subheader("Mock Preview")
                
                # The raw HTML snippet is displayed using st.components.v1.html
                # This renders the mock Tailwind CSS/HTML (Check C.3)
                st.components.v1.html(
                    f"""
                    <script src="https://cdn.tailwindcss.com"></script>
                    <div class="space-y-6">
                        <div class="text-xl font-bold p-2 bg-blue-100 rounded-md">Full Page Mockup:</div>
                        {proposal.full_html} 
                    </div>
                    """, 
                    height=650
                )
