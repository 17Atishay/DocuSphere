# llm_handler.py
# Handles all Groq API interactions
# Fast inference, generous free tier

import os
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Best free model on Groq — fast and capable
MODEL = "llama-3.3-70b-versatile"


def build_prompt(question: str, context_chunks: List[Dict], mode: str) -> str:
    """
    Build a well-structured prompt.
    context_chunks: list of metadata dicts from Endee query results
    mode: "document" or "research"
    """
    context_texts = [chunk.get("text", "") for chunk in context_chunks if chunk.get("text")]

    if not context_texts:
        return question

    context_block = "\n\n---\n\n".join(context_texts)

    if mode == "document":
        source = context_chunks[0].get("source", "the uploaded document")
        system_prompt = f"""You are an intelligent document assistant.
A user has uploaded a document called '{source}' and is asking questions about it.
Answer ONLY based on the context provided.
If the answer is not found in the context, say "I couldn't find that in the document." """

    elif mode == "research":
        topic = context_chunks[0].get("topic", "the researched topic")
        system_prompt = f"""You are an intelligent research assistant.
The following context was gathered from Wikipedia and web search about '{topic}'.
Answer ONLY based on the context provided.
If the answer is not in the context, say "I couldn't find that in the research data."
Always mention key facts and be comprehensive."""

    else:
        system_prompt = "Answer the following question based on the context provided."

    user_prompt = f"""CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:"""

    return system_prompt, user_prompt


def get_answer(question: str, context_chunks: List[Dict], mode: str) -> str:
    """
    Main function — takes question + retrieved Endee chunks
    and returns Groq's grounded answer.
    """
    if not GROQ_API_KEY:
        return "Error: Groq API key not found. Please check your .env file."

    system_prompt, user_prompt = build_prompt(question, context_chunks, mode)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating answer: {str(e)}"


def get_summary(text: str) -> str:
    """
    Summarize entire document or research content.
    """
    if not GROQ_API_KEY:
        return "Error: Groq API key not found."

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful summarization assistant."
                },
                {
                    "role": "user",
                    "content": f"""Please provide a comprehensive yet concise summary structured with:
- Main topic/theme
- Key points (in bullet form)
- Important conclusions or findings

CONTENT:
{text[:8000]}

SUMMARY:"""
                }
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating summary: {str(e)}"