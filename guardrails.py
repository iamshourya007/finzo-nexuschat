"""Guardrails for PII masking and scope detection."""
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os

# Regex patterns for PII
EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_REGEX = r"\b\d{10}\b"
AADHAAR_REGEX = r"\b\d{12}\b"

def mask_pii(text: str) -> str:
    """Mask PII in the given text."""
    if not isinstance(text, str):
        return text
        
    masked = re.sub(EMAIL_REGEX, "[EMAIL REDACTED]", text)
    masked = re.sub(PHONE_REGEX, "[PHONE REDACTED]", masked)
    masked = re.sub(AADHAAR_REGEX, "[ID REDACTED]", masked)
    return masked

def check_scope(query: str) -> bool:
    """
    LLM pre-check to ensure the query is about company internal knowledge.
    Returns True if in scope, False otherwise.
    """
    try:
        # Fallback if API key is not set to avoid breaking locally if not configured yet
        if not os.getenv("GROQ_API_KEY"):
            return True
            
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        
        prompt = PromptTemplate.from_template(
            "You are a routing assistant for a company internal knowledge base.\n"
            "Your job is to determine if the user's query is related to company internal knowledge "
            "(e.g., HR, finance, marketing, policies, employee data).\n"
            "If it is related, answer 'YES'. If it is a general knowledge question, code generation, "
            "or unrelated topic, answer 'NO'.\n\n"
            "User Query: {query}\n"
            "Answer (YES or NO):"
        )
        
        chain = prompt | llm
        result = chain.invoke({"query": query}).content.strip().upper()
        
        return "YES" in result
    except Exception as e:
        # On error, default to letting it pass and let RAG handle it
        return True
