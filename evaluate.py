"""RAGAS evaluation with 10 test cases."""
import os
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from rag_chain import RAGPipeline

# Initialize RAG Pipeline
pipeline = RAGPipeline()

# Define test cases (Question, Role, Ground Truth)
TEST_CASES = [
    {
        "question": "What is the maternity leave entitlement?",
        "role": "employee",
        "ground_truth": "Maternity leave entitlement is 6 months of paid leave according to the employee handbook."
    },
    {
        "question": "How do I submit an expense reimbursement?",
        "role": "employee",
        "ground_truth": "You can submit an expense reimbursement by filling out the expense form on the HR portal and getting manager approval."
    },
    {
        "question": "How many employees are in the Finance department?",
        "role": "hr",
        "ground_truth": "There are several employees in the Finance department based on the HR data."
    },
    {
        "question": "What is the average attendance percentage across all employees?",
        "role": "hr",
        "ground_truth": "The average attendance percentage is calculated across all active employees."
    },
    {
        "question": "What was Q4 2024 gross margin?",
        "role": "finance",
        "ground_truth": "The Q4 2024 gross margin was detailed in the quarterly financial report."
    },
    {
        "question": "What is the cash flow trend in 2024?",
        "role": "finance",
        "ground_truth": "The cash flow trend has been positive throughout 2024."
    },
    {
        "question": "What were the Q1 2024 marketing campaign highlights?",
        "role": "marketing",
        "ground_truth": "The Q1 2024 marketing campaign highlights included the successful launch of the new product line."
    },
    {
        "question": "What is the annual marketing spend for 2024?",
        "role": "marketing",
        "ground_truth": "The annual marketing spend for 2024 is documented in the annual marketing report."
    },
    {
        "question": "Which department has the most employees?",
        "role": "cxo_level",
        "ground_truth": "The engineering department has the most employees."
    },
    {
        "question": "Compare Q3 revenue with Q3 marketing spend.",
        "role": "cxo_level",
        "ground_truth": "Q3 revenue significantly outpaced Q3 marketing spend, showing a high ROI."
    }
]

def run_evaluation():
    print("Starting RAGAS evaluation on 10 test cases...")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    eval_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    for idx, tc in enumerate(TEST_CASES):
        print(f"Evaluating Case {idx+1}/10: {tc['question']}")
        
        result = pipeline.query(tc["question"], tc["role"])
        
        if result["sources"] and result["sources"][0]["source"] == "hr_data.csv":
            retrieved_contexts = ["Data queried directly from HR CSV using Pandas."]
        else:
            from rbac import get_role_filter
            retriever = pipeline.vectorstore.as_retriever(search_kwargs={"filter": get_role_filter(tc["role"]), "k": 4})
            docs = retriever.invoke(tc["question"])
            retrieved_contexts = [doc.page_content for doc in docs]
            
        if not retrieved_contexts:
            retrieved_contexts = ["No relevant context found."]
            
        questions.append(tc["question"])
        answers.append(result["answer"])
        contexts.append(retrieved_contexts)
        ground_truths.append(tc["ground_truth"]) 
        
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    print("\nRunning RAGAS Metrics...")
    
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    print("\n=== Evaluation Results ===")
    print(result)
    
    df = result.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()
