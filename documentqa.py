from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.messages import BaseMessage
from typing import Sequence
import re
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from langchain_core.messages import BaseMessage

# === LLM & VectorStore ===
llm = ChatOllama(model="llama3")

#llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en", model_kwargs={"device": "cpu"})
vectorstore = Chroma(persist_directory="stores/insurance_metadata_v4", embedding_function=embeddings)

# === Prompt Templates ===
step_back_prompt = PromptTemplate.from_template("""
Identify the fundamental insurance concept needed to answer this question. Focus on general principles rather than specifics.

Question: {question}

Fundamental Concept:"""
)


reasoning_prompt = PromptTemplate.from_template("""
Answer the question using the provided insurance documents:

General Context:
{step_back_answer}

Insurance Policy Details ({plan} Plan):
{context}

Question: {question}

Rules:
1. Be concise and factual
2. Quote exact policy terms
3. If unsure, say "I couldn't find a definitive answer."

Answer:""")

confidence_prompt = PromptTemplate.from_template("""
Answer: {answer}

Documents:
{documents}

Score the confidence (0.0-1.0) using these STRICT criteria:
    1.0 = All key facts in answer are directly supported by documents with exact numbers
    0.8 = Minor wording differences but same meaning
    0.6 = Some details missing but main point correct
    0.4 = Partial match with important discrepancies
    0.2 = Contradicts documents
    0.0 = Completely unsupported
                                                 
Provide ONLY the numeric score with no explanation:""")

# === Utility Functions ===
def clean(text: str) -> str:
    text = ' '.join(text.split())
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    return text

def detect_plan(question: str):
    q = question.lower()
    for p in ["basic", "standard", "enhanced", "uhip", "ohip"]:
        if p in q:
            return p
    return None

# === Agent Nodes ===
def step_back_rag(state) -> dict:
    question = state["messages"][-1].content
    plan = detect_plan(question)
    print("plan:",plan)
    docs = vectorstore.similarity_search(question, k=5, filter={"plan_type": plan} if plan else None)
    context = "\n\n".join([clean(d.page_content) for d in docs])
    print("context:",context)
    #step_back = llm(step_back_prompt.format(question=question))
    step_back = llm.invoke(step_back_prompt.format(question=question)).content
    print("step_back:",step_back)
    answer = llm.invoke(reasoning_prompt.format(context=context, 
                                                step_back_answer=step_back, 
                                                question=question, 
                                                plan=plan or "All")
                                                ).content
    print("step_back_rag answer:",answer)
    return {**state, "context_docs": docs, "plan": plan, "answer": answer.strip()}

def evaluate_confidence(state) -> dict:
    context = "\n".join([clean(d.page_content[:300]) for d in state["context_docs"]])
    score = llm.invoke(confidence_prompt.format(answer=state["answer"], documents=context))
    try:
        confidence = float(score.strip())
    except Exception:
        confidence = 0.5
    return {**state, "confidence": confidence}

from langchain_core.messages import AIMessage
def correct_answer(state) -> dict:
    question = state["messages"][-1].content
    docs = vectorstore.similarity_search(
        question, k=10,
        filter={"plan_type": state["plan"]} if state["plan"] else None
    )
    print('correct_answer docs:', docs)
    context = "\n\n".join([clean(d.page_content) for d in docs])
    corrected = llm.invoke(f"""Revise this answer using more context:
                        Original Answer: {state['answer']}
                        New Context: {context}

                        Correction Rules:
                        - Use exact policy language
                        - Prefix with [Verified] if confident
                        - Mark uncertainties clearly
                        """)
    return {**state, "answer": corrected.strip(), "messages": state["messages"] + [AIMessage(content=corrected.strip())]}

def is_low_confidence(state) -> bool:
    return state["confidence"] < 0.5




# === Agent State ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context_docs: list
    plan: str | None
    answer: str
    confidence: float

def create_document_qa_graph():
    graph = StateGraph(AgentState)

    graph.add_node("step_back", step_back_rag)
    graph.add_node("evaluate_conf", evaluate_confidence)
    graph.add_node("correct", correct_answer)

    graph.add_edge("step_back", "evaluate_conf")
    graph.add_conditional_edges("evaluate_conf", is_low_confidence, {
        True: "correct",
        False: END
    })
    graph.add_edge("correct", END)

    graph.set_entry_point("step_back")
    return graph.compile()


routing_agent_prompt ="""
You are an intelligent query router. Your task is to decide how to handle the following user question — whether it should be answered by document retrieval, a pre-defined tool, or custom SQL generation.

You can respond with only one of the following options:
- "doc" → if the question relates to insurance documents, plans (e.g., Sunlife, UHIP, OHIP), or requires extracting context from uploaded PDFs (invoices, policies, etc.).
- "tool" → if the question can be answered by using any of the available predefined tools.
- "sql" → if the question requires a custom SQL query that is not directly answerable via the above tools or documents.

Here is what each mode supports:

DOCUMENT RETRIEVAL ("doc"):
- Handles questions about general insurance plans (basic, standard, enhanced, UHIP, OHIP, Sunlife, etc.).
- Retrieves and reasons over uploaded documents such as policies or invoices.

TOOLS ("tool"):
- Tools access structured insurance data directly. If the question clearly maps to any tool's input (like user_id, claim_id), use a tool.
- include:
        -  get_user_by_id: This tool retrieves a single user's details by their user_id
        -  get_users_by_provider: This tool finds all users associated with a specific insurance provider
        -  get_policies_by_user: This tool retrieves all insurance policies for a specific user
        -  get_active_policies: This tool list all currently active insurance policies
        -  get_claims_by_user: This tool retrieves all claims submitted by a specific user
        -  get_provider_details: This tool gets information about an insurance provider
        -  get_provider_plans: This tool list all available plans from a specific insurance provider
        -  get_payments_by_policy: This tool retrieves payment history for a specific policy
        -  get_coverage_limits: This tool gets coverage limits and usage for a specific user
        -  get_pre_authorizations: This tool retrieves pre-authorization requests for a user
        -  get_dental_details_by_user: This tool retrieve all dental details for a specific user with procedure details
        -  get_drug_details_by_user: This tool gets all prescription drug details for a user with medication details
        -  get_hospital_visits_by_user: This tool retrieve all hospital visits for a user with stay details
        -  get_vision_claims_by_user: This tool get all vision care claims for a user with product details
        -  get_user_coverage_limits: This tool retrieve all coverage limits and usage for a specific user
        -  get_claim_audit_logs: This tool get audit history for all claims belonging to a user
        -  get_user_claim_documents: This tool retrieve all documents submitted with a user's claims
        -  get_user_preferences: This tool get communication preferences and settings for a user
        -  get_user_communications: This tool retrieve all communications sent to/from a user

CUSTOM SQL ("sql"):
- Choose this when:
    - The question cannot be answered with available tools.
    - The question involves filtering, joins, or aggregation not captured by tool inputs.
    - You need to compose dynamic queries beyond the capability of predefined tools.

QUESTION:
{query}

Respond with only one word: "doc", "tool", or "sql".

"""


def is_tool_query_llm(query: str) -> bool:
    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke(routing_agent_prompt.format( query=query)).content.strip().lower()
    print("is_tool_query_llm:", response)
    return response

is_tool_query_llm("What is the Drug coverage reimburses in a year in a Sunlife personal health insurance basic plan")