"""LangChain RAG pipeline with RBAC retriever."""
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from rbac import get_role_filter
from hr_analytics import is_hr_analytics_query, query_hr_data
from guardrails import mask_pii, check_scope

class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings
        )
        
        self.prompt = PromptTemplate.from_template(
            "You are an internal knowledge assistant for Finzo Technologies.\n"
            "Use the following pieces of retrieved context to answer the question.\n"
            "If you don't know the answer based on the context, say that you don't know.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, user_query: str, role: str) -> dict:
        """
        Execute RAG query with RBAC and routing.
        Returns dict with "answer" and "sources".
        """
        # Scope Guard
        if not check_scope(user_query):
            return {
                "answer": "I can only answer questions about finzo's internal company knowledge.",
                "sources": []
            }
            
        # Mask PII in query just in case (though we mainly mask response)
        masked_query = mask_pii(user_query)

        # Check HR analytics routing
        if is_hr_analytics_query(masked_query, role):
            raw_answer = query_hr_data(masked_query)
            masked_answer = mask_pii(raw_answer)
            return {
                "answer": masked_answer,
                "sources": [{"source": "hr_data.csv", "department": "hr"}]
            }

        # Standard RAG
        filter_dict = get_role_filter(role)
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"filter": filter_dict, "k": 4}
        )
        
        # Retrieve docs
        docs = retriever.invoke(masked_query)
        
        # Build chain
        chain = (
            {"context": lambda x: self.format_docs(docs), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        raw_answer = chain.invoke(masked_query)
        masked_answer = mask_pii(raw_answer)
        
        sources = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            dept = doc.metadata.get("department", "Unknown")
            source_dict = {"source": src, "department": dept}
            if source_dict not in sources:
                sources.append(source_dict)
                
        return {
            "answer": masked_answer,
            "sources": sources
        }
