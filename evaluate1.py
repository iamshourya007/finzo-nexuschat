"""RAGAS evaluation with 15 grounded test cases — all ground truths verified against source files."""
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

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES — all ground truths verified directly from source data files
# Sources:
#   employee   → employee_handbook.md
#   hr         → hr_data.csv  (100 rows, pandas-powered)
#   finance    → quarterly_financial_report.md + financial_summary.md
#   marketing  → marketing_report_q1/q2/q3_2024.md, market_report_q4_2024.md,
#                marketing_report_2024.md
#   engineering→ engineering_master_doc.md
#   cxo_level  → all of the above
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [

    # ── EMPLOYEE ROLE (employee_handbook.md) ─────────────────────────────────
    {
        "question": "What is the maternity leave entitlement?",
        "role": "employee",
        "ground_truth": (
            "Maternity leave entitlement is 26 weeks of paid leave for the first two children, "
            "and 12 weeks for subsequent children, as per the Maternity Benefit Act."
        )
    },
    {
        "question": "How do I submit an expense reimbursement?",
        "role": "employee",
        "ground_truth": (
            "Submit expense claims via HRMS with original bills within 30 days of incurring the expense. "
            "Claims are reviewed and approved by the reporting manager, and reimbursement is processed "
            "with the next payroll cycle. Required documentation includes original bills/invoices, "
            "travel tickets/boarding passes, and approval emails if pre-approval was required."
        )
    },
    {
        "question": "How many days of sick leave are employees entitled to per year?",
        "role": "employee",
        "ground_truth": (
            "Employees are entitled to 12 days of sick leave per year. Sick leave is non-cumulative, "
            "and a medical certificate is required for absences exceeding 2 days."
        )
    },

    # ── HR ROLE (hr_data.csv — pandas) ───────────────────────────────────────
    {
        "question": "How many employees are in the Finance department?",
        "role": "hr",
        "ground_truth": (
            "Based on the HR data, there are 16 employees in the Finance department."
        )
    },
    {
        "question": "What is the average attendance percentage across all employees?",
        "role": "hr",
        "ground_truth": (
            "The average attendance percentage across all 100 employees is approximately 90.74%."
        )
    },
    {
        "question": "Which employee has the highest performance rating?",
        "role": "hr",
        "ground_truth": (
            "Multiple employees have the highest performance rating of 5. "
            "Performance ratings in the HR data range from 1 to 5."
        )
    },

    # ── FINANCE ROLE (quarterly_financial_report.md + financial_summary.md) ──
    {
        "question": "What was the Q4 2024 gross margin?",
        "role": "finance",
        "ground_truth": (
            "The Q4 2024 gross margin was 64%, reflecting optimized pricing and operational efficiencies. "
            "Q4 revenue was $2.6 billion, up 35% year-over-year."
        )
    },
    {
        "question": "What is the overall gross margin for 2024 and how does it compare to 2023?",
        "role": "finance",
        "ground_truth": (
            "The annual gross margin for 2024 was 60%, up from 55% in 2023, which is above the "
            "industry benchmark of 55%. This reflects higher operational efficiency and cost control."
        )
    },
    {
        "question": "What was the cash flow from operations in 2024?",
        "role": "finance",
        "ground_truth": (
            "Cash flow from operations in 2024 was $50 million, a 20% increase over the prior year. "
            "However, delayed payment cycles from key vendors slightly impacted cash liquidity "
            "in the second half of 2024."
        )
    },

    # ── MARKETING ROLE ────────────────────────────────────────────────────────
    {
        "question": "What were the key marketing initiatives in Q1 2024?",
        "role": "marketing",
        "ground_truth": (
            "Q1 2024 key marketing initiatives included: European Market Entry with campaigns in the UK, "
            "Germany, and France; the launch of the InstantPay feature which achieved 52,000 sign-ups "
            "exceeding the 50,000 target; and increased social media engagement which achieved a 12% "
            "increase, surpassing the 10% target. The total Q1 marketing spend was $2 million."
        )
    },
    {
        "question": "What was the total marketing budget for 2024 and how was it allocated?",
        "role": "marketing",
        "ground_truth": (
            "The total marketing budget for 2024 was $15 million, allocated as follows: "
            "Digital Marketing $7M (47%), Event Marketing $3M (20%), "
            "Public Relations and Brand Awareness $2M (13%), "
            "Customer Retention $1M (7%), and Market Research and Analytics $2M (13%)."
        )
    },
    {
        "question": "What was the Q3 2024 marketing spend and what were the key campaigns?",
        "role": "marketing",
        "ground_truth": (
            "The Q3 2024 marketing spend was $2 million, allocated across digital advertising (35%), "
            "loyalty programs (20%), event sponsorships (20%), merchant partnerships (15%), and "
            "content marketing (10%). Key campaigns focused on customer retention through a "
            "points-based loyalty program and Latin American market expansion in Brazil, Mexico, "
            "and Colombia, with 15 merchant partnerships secured."
        )
    },

    # ── ENGINEERING ROLE (engineering_master_doc.md) ─────────────────────────
    {
        "question": "What cloud infrastructure does the company use?",
        "role": "engineering",
        "ground_truth": (
            "The company uses AWS as the primary cloud provider, utilizing EC2, ECS, Lambda, RDS, S3, "
            "CloudFront, and other managed services. Kubernetes is used for container orchestration, "
            "and Cloudflare provides CDN, DDoS protection, and Web Application Firewall (WAF)."
        )
    },
    {
        "question": "What databases does the engineering team use and for what purposes?",
        "role": "engineering",
        "ground_truth": (
            "The engineering team uses four databases: PostgreSQL for transactional data requiring "
            "ACID compliance; MongoDB for user profiles, preferences, and semi-structured data; "
            "Redis for caching, session management, and pub/sub messaging; and Amazon S3 for "
            "documents, statements, user uploads, and encrypted backups."
        )
    },

    # ── CXO LEVEL (cross-department) ─────────────────────────────────────────
    {
        "question": "Compare Q3 2024 revenue target with Q3 2024 marketing spend.",
        "role": "cxo_level",
        "ground_truth": (
            "In Q3 2024, the revenue target was $7.5 million (actual revenue generated was $7.2 million). "
            "The Q3 2024 marketing spend was $2 million. This gives a preliminary ROI of 3.6x "
            "against the target of 3.75x, meaning approximately $3.60 was generated per dollar "
            "of marketing spend."
        )
    },
]


def run_evaluation():
    print(f"Starting RAGAS evaluation on {len(TEST_CASES)} grounded test cases...")
    print("=" * 60)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    eval_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    for idx, tc in enumerate(TEST_CASES):
        print(f"[{idx+1}/{len(TEST_CASES)}] Role: {tc['role']:12s} | Q: {tc['question'][:60]}...")

        result = pipeline.query(tc["question"], tc["role"])

        # Determine context source
        if (
            result.get("sources")
            and len(result["sources"]) > 0
            and result["sources"][0].get("source") == "hr_data.csv"
        ):
            retrieved_contexts = ["Data queried directly from HR CSV using Pandas."]
        else:
            from rbac import get_role_filter
            search_filter = get_role_filter(tc["role"])
            search_kwargs = {"k": 5}
            if search_filter:
                search_kwargs["filter"] = search_filter
            retriever = pipeline.vectorstore.as_retriever(search_kwargs=search_kwargs)
            docs = retriever.invoke(tc["question"])
            retrieved_contexts = [doc.page_content for doc in docs] if docs else ["No relevant context found."]

        questions.append(tc["question"])
        answers.append(result["answer"])
        contexts.append(retrieved_contexts)
        ground_truths.append(tc["ground_truth"])

        print(f"         Answer preview: {result['answer'][:80]}...")
        print()

    print("=" * 60)
    print("Running RAGAS Metrics (faithfulness, answer_relevancy, context_recall)...")
    print("Note: faithfulness and context_recall require LLM-as-judge scoring.")
    print("      If scores return 0.0, this is a known Groq/Llama evaluator limitation.")
    print("=" * 60)

    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    }

    dataset = Dataset.from_dict(data)

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    print("\n=== Evaluation Results ===")
    print(scores)

    df = scores.to_pandas()

    # Add role and source file columns for easier analysis
    df.insert(0, "role", [tc["role"] for tc in TEST_CASES])
    df.insert(1, "question_short", [tc["question"][:50] for tc in TEST_CASES])

    df.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed results saved to evaluation_results.csv")

    # Print per-role summary
    print("\n=== Answer Relevancy by Role ===")
    if "answer_relevancy" in df.columns:
        role_summary = df.groupby("role")["answer_relevancy"].mean().round(4)
        print(role_summary.to_string())


if __name__ == "__main__":
    run_evaluation()
