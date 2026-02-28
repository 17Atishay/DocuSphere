# llm_handler.py
# Handles all Gemini API interactions
# Takes retrieved context chunks from Endee + user question
# and generates a grounded, accurate final answer

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Model to use
MODEL = "gemini-2.0-flash"  # Fast, capable, generous free tier


def build_prompt(question: str, context_chunks: List[Dict], mode: str) -> str:
    """
    Build a well-structured prompt for Gemini.
    context_chunks: list of metadata dicts from Endee query results
    mode: "document" or "research"
    """
    # Extract text from metadata dicts returned by Endee
    context_texts = [chunk.get("text", "") for chunk in context_chunks if chunk.get("text")]

    if not context_texts:
        return f"Answer this question to the best of your ability: {question}"

    # Join context chunks into one block
    context_block = "\n\n---\n\n".join(context_texts)

    if mode == "document":
        source = context_chunks[0].get("source", "the uploaded document")
        prompt = f"""You are an intelligent document assistant. 
A user has uploaded a document called '{source}' and is asking questions about it.
Answer ONLY based on the context provided below. 
If the answer is not found in the context, say "I couldn't find that in the document."

CONTEXT FROM DOCUMENT:
{context_block}

USER QUESTION:
{question}

ANSWER:"""

    elif mode == "research":
        topic = context_chunks[0].get("topic", "the researched topic")
        prompt = f"""You are an intelligent research assistant.
The following context was gathered from Wikipedia and web search about '{topic}'.
Answer the user's question ONLY based on the context provided below.
If the answer is not in the context, say "I couldn't find that in the research data."
Always mention key facts and be comprehensive in your answer.

RESEARCH CONTEXT:
{context_block}

USER QUESTION:
{question}

ANSWER:"""

    else:
        prompt = f"""Answer the following question based on the context below.

CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:"""

    return prompt


def get_answer(question: str, context_chunks: List[Dict], mode: str) -> str:
    """
    Main function — takes question + retrieved Endee chunks
    and returns Gemini's grounded answer.
    """
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not found. Please check your .env file."

    prompt = build_prompt(question, context_chunks, mode)

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                # max_output_tokens=1024
                max_output_tokens=512
            )
        )
        return response.text

    except Exception as e:
        return f"Error generating answer: {str(e)}"


def get_summary(text: str) -> str:
    """
    Bonus feature — summarize an entire document or research content.
    Called when user clicks 'Summarize' button in the UI.
    """
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not found."

    prompt = f"""Please provide a comprehensive yet concise summary of the following content.
Structure your summary with:
- Main topic/theme
- Key points (in bullet form)
- Important conclusions or findings

CONTENT:
{text[:8000]}

SUMMARY:"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024
            )
        )
        return response.text

    except Exception as e:
        return f"Error generating summary: {str(e)}"