"""
RAG CORE ENGINE (rag_engine.py)
This file orchestrates the RAG pipeline using separate modules for loading and retrieval.
"""

import os
import json
from typing import List, Optional, Dict, Any
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from data_loader import load_knowledge_base
from retriever import ComponentRetriever
from safety_filters import SafetyFilter
from conversation_manager import ConversationChain

# --- Initialization & Security ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

# Groq Client Initialization
client = Groq()
MODEL_NAME = "llama-3.1-8b-instant"

# Safety Filter Initialization
safety_filter = SafetyFilter()

# --- Constants ---
UNMATCHED_PROMPT_MESSAGE = "⚠️ This system is specialized for **Law Firm Website Design** only.\n\nYour request doesn't appear to be related to creating a law firm website. Please provide details about:\n\n✅ **Type of law practice** (e.g., criminal defense, estate planning, corporate law, family law)\n✅ **Design tone/style** (e.g., modern, traditional, aggressive, conservative)\n✅ **Target audience** (e.g., individuals, corporations, families)\n\n**Example prompts:**\n- 'Modern and aggressive site for criminal defense attorney'\n- 'Traditional and trustworthy site for estate planning firm'\n- 'Professional corporate law firm with clean design'"


# --- Data Models ---
class DesignProposal(BaseModel):
    """Defines the structure for a single, complete website design proposal."""

    design_name: str = Field(description="A marketing name for the proposed design.")
    narrative_summary: str = Field(
        description="A 2-3 sentence summary explaining the design rationale based ONLY on the context provided."
    )
    hero_id: str = Field(
        description="The ID (e.g., 'H-01') of the chosen Hero component."
    )
    services_id: str = Field(
        description="The ID (e.g., 'S-01') of the chosen Services component."
    )
    contact_id: str = Field(
        description="The ID (e.g., 'C-01') of the chosen Contact component."
    )
    full_html: Optional[str] = Field(
        default=None, description="Complete HTML mockup assembled from components."
    )


class DesignOutput(BaseModel):
    """Defines the final, expected structured output."""

    proposals: List[DesignProposal] = Field(
        description="A list of exactly 3 distinct website design proposals."
    )


class QueryValidation(BaseModel):
    """Validation response for user queries."""

    is_valid: bool = Field(
        description="Whether the query is related to law firm website design"
    )
    reason: str = Field(description="Brief explanation of why it's valid or invalid")


def clean_html(html_str: str) -> str:
    """
    Cleans HTML snippets by converting escaped characters to actual characters.

    Args:
        html_str: Raw HTML string from CSV

    Returns:
        Cleaned HTML string
    """
    if not html_str:
        return ""

    # Replace escaped newlines with actual newlines
    cleaned = html_str.replace("\\n", "\n")

    # Remove any extra whitespace
    cleaned = cleaned.strip()

    return cleaned


def validate_query(user_prompt: str) -> Dict[str, Any]:
    """
    Validates if the user query is related to law firm website design.

    Args:
        user_prompt: The user's input query

    Returns:
        Dict with 'is_valid' (bool) and 'reason' (str)
    """
    try:
        validation_prompt = f"""You are a query validation system for a law firm website design tool.

Analyze this user query and determine if it's related to creating/designing a law firm website.

User Query: "{user_prompt}"

A query is VALID if it mentions:
- Any type of legal practice (criminal defense, estate planning, corporate law, family law, immigration, etc.)
- Website design for lawyers/attorneys/law firms
- Legal services website
- Law office website

A query is INVALID if it:
- Asks unrelated questions (jokes, general questions, math, etc.)
- Is about non-legal businesses
- Contains offensive content
- Is completely off-topic

Return ONLY valid JSON in this format:
{{
    "is_valid": true or false,
    "reason": "Brief explanation"
}}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a query validator. Respond only with valid JSON.",
                },
                {"role": "user", "content": validation_prompt},
            ],
            temperature=0.3,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Error validating query: {e}")
        # If validation fails, assume it might be valid to avoid false negatives
        return {"is_valid": True, "reason": "Validation system error"}


# --- Core RAG Pipeline ---
def generate_design_proposals(
    user_prompt: str, components: List[Dict[str, Any]]
) -> Optional[DesignOutput]:
    """
    Augments the prompt with context and generates structured proposals via Groq.

    Args:
        user_prompt: The user's input prompt
        components: List of relevant components from the retriever

    Returns:
        DesignOutput containing generated proposals or None if generation fails
    """
    try:
        # Format the context for the prompt
        context_str = "\n".join(
            [
                f"ID: {item['id']} | Type: {item['feature']} | Keywords: {item.get('keywords', '')} | Tone: {item.get('tone', '')}"
                for item in components
            ]
        )

        # Create the prompt with context
        prompt = f"""You are an expert web designer specializing in law firm websites. Based on the available components and user request, generate 3 distinct website design proposals.

Available Components:
{context_str}

User Request: {user_prompt}

For each proposal:
1. Choose ONE Hero component (H-XX) that matches the tone
2. Choose ONE Services component (S-XX) that fits the practice area
3. Choose ONE Contact component (C-XX) that matches urgency level
4. Create a unique design name that reflects the tone and purpose
5. Write a 2-3 sentence narrative explaining why this combination works for the user's needs

RULES:
- Each proposal must use DIFFERENT combinations of components
- Match the tone from the user's request (modern/traditional/aggressive/conservative)
- Match the practice area (criminal, estate, corporate, etc.)
- The narrative should reference specific keywords from the components

Return ONLY valid JSON with this exact structure:
{{
    "proposals": [
        {{
            "design_name": "Design Name Here",
            "narrative_summary": "2-3 sentence explanation referencing the components chosen.",
            "hero_id": "H-01",
            "services_id": "S-02",
            "contact_id": "C-02"
        }},
        {{
            "design_name": "Different Design Name",
            "narrative_summary": "Different explanation.",
            "hero_id": "H-02",
            "services_id": "S-01",
            "contact_id": "C-01"
        }},
        {{
            "design_name": "Third Design Name",
            "narrative_summary": "Third explanation.",
            "hero_id": "H-01",
            "services_id": "S-01",
            "contact_id": "C-02"
        }}
    ]
}}
"""

        # Call Groq API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional web design consultant specializing in law firm websites. Always respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        # Parse and validate the response
        result = json.loads(response.choices[0].message.content)

        # Validate that we have proposals
        if "proposals" not in result or not result["proposals"]:
            print("Error: No proposals in response")
            return None

        # Get HTML for each proposal and clean it
        for proposal in result["proposals"]:
            # Find matching components
            hero_html = next(
                (
                    clean_html(c["html"])
                    for c in components
                    if c["id"] == proposal.get("hero_id")
                ),
                "",
            )
            services_html = next(
                (
                    clean_html(c["html"])
                    for c in components
                    if c["id"] == proposal.get("services_id")
                ),
                "",
            )
            contact_html = next(
                (
                    clean_html(c["html"])
                    for c in components
                    if c["id"] == proposal.get("contact_id")
                ),
                "",
            )

            # If any component is missing, use a placeholder
            if not hero_html:
                hero_html = "<div class='bg-gray-800 text-white p-8 rounded-lg'><h2>Hero Section Placeholder</h2></div>"
            if not services_html:
                services_html = "<div class='bg-white p-8 rounded-lg'><h2>Services Section Placeholder</h2></div>"
            if not contact_html:
                contact_html = "<div class='bg-gray-100 p-8 rounded-lg'><h2>Contact Section Placeholder</h2></div>"

            # Assemble the full HTML with proper spacing
            proposal["full_html"] = f"""
                <div class="space-y-8 p-4">
                    {hero_html}
                    {services_html}
                    {contact_html}
                </div>
            """

        return DesignOutput(**result)

    except Exception as e:
        print(f"Error generating design proposals: {e}")
        if "response" in locals():
            print(f"Response content: {response.choices[0].message.content}")
        return None


def run_rag_pipeline(
    user_prompt: str, conversation_chain: Optional[ConversationChain] = None
) -> Dict[str, Any]:
    """
    Runs the full RAG pipeline:
    1. Pre-filters input for safety (dangerous patterns, greetings)
    2. Checks for conversation context and follow-ups
    3. Validates the query is law-firm related
    4. Loads the knowledge base
    5. Retrieves relevant components
    6. Generates design proposals

    Args:
        user_prompt: User's input query
        conversation_chain: Optional conversation chain for context

    Returns:
        Dict containing either the proposals or an error message
    """
    try:
        # Handle conversation context if provided
        enhanced_prompt = user_prompt
        if conversation_chain:
            # Build contextual query for follow-ups
            enhanced_prompt = conversation_chain.build_contextual_query(user_prompt)

            # Store user message in conversation
            conversation_chain.add_message("user", user_prompt)

        # Step 0: Safety filter check (pre-RAG filtering)
        safety_check = safety_filter.validate_input(user_prompt)

        # Handle greetings
        if safety_check["is_greeting"]:
            return {
                "is_greeting": True,
                "message": safety_check["message"]
            }

        # Block dangerous inputs
        if safety_check["is_dangerous"]:
            return {
                "is_dangerous": True,
                "error": safety_check["message"]
            }

        # If not safe to continue, stop here
        if not safety_check["should_continue"]:
            return {
                "error": "Input validation failed. Please try a different request."
            }

        # Step 1: Validate the query
        validation = validate_query(user_prompt)

        if not validation.get("is_valid", False):
            return {
                "message": UNMATCHED_PROMPT_MESSAGE,
                "validation_reason": validation.get(
                    "reason", "Query not related to law firm websites"
                ),
            }

        # Step 2: Load data
        kb_path = os.path.join(os.path.dirname(__file__), "design_db.csv")
        df = load_knowledge_base(kb_path)

        if df.empty:
            return {
                "error": "Failed to load knowledge base. Please check the data file."
            }

        # Step 3: Retrieve relevant components (use enhanced prompt for better context)
        retriever = ComponentRetriever(df)
        components = retriever.get_relevant_components(enhanced_prompt, top_n=6)

        if not components or len(components) < 3:
            return {
                "message": UNMATCHED_PROMPT_MESSAGE,
                "details": "Could not find enough matching components for your request.",
            }

        # Step 4: Generate design proposals
        result = generate_design_proposals(user_prompt, components)

        if not result:
            return {
                "error": "Failed to generate design proposals. Please try rephrasing your request."
            }

        # Validate we have 3 proposals
        if len(result.proposals) < 3:
            return {
                "error": "System generated fewer than 3 proposals. Please try again."
            }

        # Store assistant response in conversation
        proposals_list = [p.model_dump() for p in result.proposals]
        if conversation_chain:
            conversation_chain.add_message(
                "assistant",
                f"Generated {len(proposals_list)} design proposals",
                metadata={"proposals": proposals_list}
            )

        return {
            "proposals": proposals_list,
            "is_follow_up": conversation_chain.detect_follow_up(user_prompt) if conversation_chain else False
        }

    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}. Please try again."}
