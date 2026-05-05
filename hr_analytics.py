"""Pandas-powered HR CSV query handler."""
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

HR_DATA_PATH = "data/hr/hr_data.csv"

def is_hr_analytics_query(query: str, role: str) -> bool:
    """Determine if query should be routed to HR analytics."""
    if role not in ["hr", "cxo_level"]:
        return False
        
    keywords = ["how many", "average", "highest", "lowest", "count", "salary", "rating", "employees in", "performance", "department has the most"]
    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords)

def query_hr_data(query: str) -> str:
    """Query HR CSV data using Pandas and summarize with LLM."""
    if not os.path.exists(HR_DATA_PATH):
        return "HR data file not found."
        
    df = pd.read_csv(HR_DATA_PATH)
    
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        
        # Ask LLM to generate pandas expression
        prompt = PromptTemplate.from_template(
            "You are a pandas data analyst. You have a DataFrame 'df' with columns:\n"
            "{columns}\n\n"
            "Write ONLY a valid python expression using 'df' that answers the user's query. "
            "Do NOT assign to variables. Just return the expression that evaluates to the answer.\n"
            "Examples:\n"
            "df['salary'].mean()\n"
            "len(df[df['department'] == 'finance'])\n"
            "df.loc[df['performance_rating'].idxmax()]['full_name']\n"
            "df['department'].value_counts().idxmax()\n\n"
            "User query: {query}\n"
            "Expression:"
        )
        
        columns = ", ".join(df.columns)
        chain = prompt | llm
        
        expression = chain.invoke({"columns": columns, "query": query}).content.strip()
        
        # Clean markdown
        expression = expression.replace("```python", "").replace("```", "").strip()
        
        # Evaluate
        result = eval(expression, {"pd": pd, "df": df})
        
        # Summarize
        summary_prompt = PromptTemplate.from_template(
            "The user asked: {query}\n"
            "The data analysis returned: {result}\n"
            "Provide a natural language answer explaining this result."
        )
        summary_chain = summary_prompt | llm
        summary = summary_chain.invoke({"query": query, "result": str(result)}).content
        
        return summary
        
    except Exception as e:
        return f"I couldn't analyze the HR data for that query."
