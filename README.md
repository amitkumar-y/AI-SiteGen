# AI SiteGen

A RAG-based web application that generates law firm website design proposals using AI.

## Overview

This tool uses Retrieval-Augmented Generation (RAG) to suggest website designs for law firms based on natural language descriptions. It matches user requirements with pre-defined design components and uses Groq's LLM to generate coherent proposals.

## Features

- Natural language input for describing law firm websites
- Generates 3 design proposals per request
- Live HTML previews using Tailwind CSS
- Fuzzy matching for component retrieval
- Structured output validation with Pydantic

## Installation

Install dependencies:

1. Clone the repository:
   ```bash
   git clone [https://github.com/amitkumar-y/AI-SiteGen.git](https://github.com/amitkumar-y/AI-SiteGen.git)
   cd AI-SiteGenerator
   
2. Install dependencies:

   `pip install -r requirements.txt`

3. Create a .env file with your Groq API key
 GROQ_API_KEY=your_api_key_here

4. Run the application:
streamlit run app.py
---------------------------

Usage
Enter a description like: "Modern site for criminal defense" or "Traditional estate planning site"
Click "Generate Design"
Review the 3 generated proposals with HTML previews

-------------------
Project Structure


├── app.py              # Streamlit UI

├── rag_engine.py       # RAG pipeline and generation logic

├── design_db.csv       # Design component knowledge base

└── requirements.txt    # Python dependencies


-----------

Tech Stack
Frontend: Streamlit
LLM: Groq (Llama 3 8B)
Retrieval: RapidFuzz for fuzzy matching
Validation: Pydantic for structured outputs

   

   
