import re
import sqlite3
from fastapi import FastAPI, HTTPException, Request
import pandas as pd
from pydantic import BaseModel
from documentqa import create_document_qa_graph, is_tool_query_llm
from utils import hash_password, init_db, generate_token
from auth_graph import build_auth_graph
from fastapi import Depends, HTTPException, Header
from utils import verify_token
from langchain.agents import Tool, initialize_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent


from langchain.agents import AgentExecutor
from langchain_community.utilities import SQLDatabase

from langchain_core.tools import tool
from langchain.agents import AgentExecutor

from langchain_core.tools import tool
from typing_extensions import TypedDict
from typing import Annotated, List
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from langchain.tools import tool
import sqlite3
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import Runnable
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
import uuid



app = FastAPI()
init_db()
engine = create_engine("sqlite:///claims.db")
print(inspect(engine).get_table_names()) 
db = SQLDatabase.from_uri("sqlite:///claims.db") 


# Input Models
class LoginInput(BaseModel):
    username: str
    password: str

class TokenInput(BaseModel):
    token: str

class ChatRequest(BaseModel):
    query: str



claims_tool_agent_prompt = """
You are a smart assistant who can access structured insurance data using pre-defined tools. Your goal is to help the user answer their question by calling the appropriate tool.

Available tools:
- Tools can fetch information about users, providers, policies, claims, payments, coverage, documents, audit logs, and more.
- Tools are described with their input arguments. Use them when the question matches their capabilities.
You have access to many tools. These tools enable you to retrieve and process invoice information from the database. Here are the tools:
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

Instructions:
- ALWAYS call a tool if one is available that can answer the question.
- NEVER generate SQL. Your job is to use tools.
- Do not answer questions from memory.
- If the user query cannot be handled by a tool, return: "I'm unable to answer this with available tools."

Respond by calling one of the tools if possible.
"""
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
import uuid
import asyncio
import nest_asyncio
from langgraph.prebuilt import create_react_agent # Import the pre-built ReAct agent creator
from langgraph.checkpoint.memory import MemorySaver # For short-term memory (thread-level state persistence)
from langgraph.store.memory import InMemoryStore # For long-term memory (storing user preferences)
from langchain.agents import AgentExecutor
from langchain_core.runnables import Runnable
from langgraph.graph.message import AnyMessage, add_messages # For managing messages in the graph state
from langgraph.managed.is_last_step import RemainingSteps # For tracking recursion limits
from typing_extensions import TypedDict 
from typing import Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
import uuid

#nest_asyncio.apply()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "chat_history"]

class State(TypedDict):
    """Represents the state of our LangGraph agent."""
    user_id: str
    
    messages: Annotated[list[AnyMessage], add_messages]
    
    loaded_memory: str
    
    remaining_steps: RemainingSteps 

async def run_agent_query( question: str): 
    # ✅ Start MCP client
    client = MultiServerMCPClient({
        "claims": {
            "command": "python",
            "args": ["/Users/nandhinirajasekaran/Desktop/LLM/LangGraph/AuthBased/mcp_test_server.py"],
            "transport": "stdio"
        }
    })

    async with client.session("claims") as session:
        # ✅ Load tools inside session scope
        tools = await load_mcp_tools(session)
        print("✅ Tools loaded:", [t.name for t in tools])
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        # ✅ Build React agent
        agent = create_react_agent(llm,tools=tools)

        # ✅ Wrap in LangGraph
        builder = StateGraph(AgentState)
        builder.add_node("agent", agent)
        builder.set_entry_point("agent")
        builder.set_finish_point("agent")
        graph = builder.compile()

        # ✅ Unique thread ID
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # ✅ Run query
        response = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )
        # Assuming `response` is the result from `agent.ainvoke(...)`
        ai_messages = [msg for msg in response["messages"] if isinstance(msg, AIMessage) and msg.content]

        # Get the last AI message (after tool call, with final answer)
        final_response = ai_messages[-1].content if ai_messages else "No AI response found."

        print("🧠 Agent Final Response:\n", final_response)
        

# 🏃‍♀️ Launch it
#await run_agent_query("What are the claims of User1")
#await run_agent_query("What are the details of User1")

import numpy as np
import networkx as nx
from scipy import spatial
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_embedding(text: str) -> np.ndarray:
    embedding = model.encode(text, convert_to_tensor=False)
    embedding = np.squeeze(embedding)
    if len(embedding.shape) == 0:
        embedding = np.array([embedding])
    if embedding.shape[0] < 384:
        embedding = np.pad(embedding, (0, 384 - embedding.shape[0]))
    elif embedding.shape[0] > 384:
        embedding = embedding[:384]
    assert embedding.shape == (384,), f"Unexpected embedding shape: {embedding.shape}"
    return embedding

def compute_similarity(embedding1, embedding2) -> float:
    return 1 - spatial.distance.cosine(embedding1, embedding2)

def load_schema_chunks(schema_path: str) -> list[dict]:
    tables = {}
    with open(schema_path, "r") as f:
        lines = f.readlines()
        #print("lines:",lines)

    current_table = None
    buffer = []
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("CREATE TABLE"):
            if buffer and current_table:
                tables[current_table] = "\n".join(buffer)
            buffer = [stripped]
            current_table = stripped.split()[2].strip("`\"")
        elif stripped == ");":
            buffer.append(stripped)
            if current_table:
                tables[current_table] = "\n".join(buffer)
            buffer = []
            current_table = None
        elif buffer:
            buffer.append(stripped)

    return [{"text": text, "metadata": {"table": table}} for table, text in tables.items()]

def create_graph(chunks: list[dict]) -> nx.Graph:
    G = nx.Graph()
    for i, chunk in enumerate(chunks):
        emb = compute_embedding(chunk["text"])
        G.add_node(i, text=chunk["text"], embedding=emb, metadata=chunk.get("metadata", {}))
    add_similarity_edges(G)
    return G

def add_similarity_edges(G: nx.Graph, threshold=0.7):
    for i in G.nodes():
        for j in G.nodes():
            if i < j:
                sim = compute_similarity(G.nodes[i]["embedding"], G.nodes[j]["embedding"])
                if sim > threshold:
                    G.add_edge(i, j, weight=sim)

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, BaseMessage
from operator import add as add_messages
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
import sqlglot
from difflib import get_close_matches
from langchain_core.messages import AIMessage


from sqlalchemy import create_engine, inspect
from typing import List, Optional, Union



class SQLAgentState(TypedDict):
    #messages: List
    messages: Annotated[list, "reduce"]
    retrieved_schema: List[dict]  # from retrieve_schema
    route: Optional[str]           # set by planner node
    sql_query: Optional[str]       # generated SQL
    execution_result: Optional[Union[str, List[dict]]]  # raw result
    agent_response: Optional[str] # agent response
    final_response: Optional[str]  # formatted answer


def retrieve_schema_context(state: SQLAgentState) -> SQLAgentState:
    question = state["messages"][-1].content
    query_embedding = model.encode(question)
    #print("query_embedding:",query_embedding)

    matches = []
    schema_chunks = load_schema_chunks("/Users/nandhinirajasekaran/Desktop/LLM/LangGraph/AuthBased/schema.sql")  # path relative to your project
    G = create_graph(schema_chunks)
    for _, data in G.nodes(data=True):
        #print("q:",query_embedding)
        similarity = float(compute_similarity(data["embedding"], query_embedding))
        if similarity > 0.4:
            matches.append({
                "text": data["text"],
                "similarity": similarity,
                "metadata": data.get("metadata", {})
            })

    # Sort and keep top 5
    top_matches = sorted(matches, key=lambda x: -x["similarity"])[:5]
    print("top matches:",top_matches)
    return {**state, "retrieved_schema": top_matches}


from typing import List, Optional, Union
from langchain_core.messages import BaseMessage

#class SQLAgentState(TypedDict):
#    messages: List[BaseMessage]
#    schema_context: Optional[str]  # from retrieve_schema
#    route: Optional[str]           # set by planner node
#    sql_query: Optional[str]       # generated SQL
#    validated_sql: Optional[str]   # validated SQL
#    execution_result: Optional[Union[str, List[dict]]]  # raw result
#    final_response: Optional[str]  # formatted answer

from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda



def planner_node(state: SQLAgentState) -> SQLAgentState:
    route = is_tool_query_llm(state["messages"])
    if route not in {"tool", "sql", "doc"}:
        route = "sql"  # fallback
        
    print("planner_node:", route)
    return {**state, "route": route}

planner: Runnable = RunnableLambda(planner_node)

sql_generator_prompt = PromptTemplate.from_template("""
You are an expert SQL agent. You are specialized for retrieving and processing information from db.
Your job is to translate natural language questions into SQL queries based on the given database schema.
Given the user question and the database schema context, generate an executable SQLite SQL query only.

    Schema:
    {schema}

    User Question:
    {question}
                                                    
    CORE RESPONSIBILITIES:
    - Retrieve and process claim information from the database
    - Provide detailed information about claims, including user details, drug details , dental details, hospital visits, insurance provider, provider plan, policies associated for the users, etc. when the user asks for it.
    - Always maintain a professional, friendly, and patient demeanor
    
    Rules:
    - Only generate SQL, Do NOT return explanations.
    - Do not create nested queries
    - never use SELECT *.   
    - Use only the tables and columns present in the schema.
    - Do NOT assume any table or column exists if it's not listed.
    - Always use table aliases only if needed                                    
    - Only use columns and tables in the schema
    - Target correct table and columns.
    - If needed, use JOINs between related tables
    - Ensure joins are valid
    - Use lowercase for SQL keywords.
    - Don't include `;` at the end.
                                          
Generate only the SQL Query:
""")

def clean_sql_response(text: str) -> str:
    """
    Remove markdown formatting and extract the raw SQL query.
    """
    # Remove ```sql and ``` blocks if present
    return text.strip().removeprefix("```sql").removesuffix("```").strip()

def generate_sql_query(state: SQLAgentState) -> SQLAgentState:
    prompt_input = sql_generator_prompt.format(
        schema="\n".join([d["text"] for d in state["retrieved_schema"]]),
        question=state["messages"][-1].content
    )

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    sql = llm.invoke(prompt_input).content.strip()
    return {**state, "sql_query": clean_sql_response(sql)}

def get_known_tables_and_columns(db_uri: str) -> dict:
    engine = create_engine(db_uri)
    inspector = inspect(engine)

    known = {}
    for table in inspector.get_table_names():
        columns = {col["name"].lower() for col in inspector.get_columns(table)}
        known[table.lower()] = columns
    return known

# Example usage
KNOWN_TABLES = get_known_tables_and_columns("sqlite:///claims.db")

#def best_table_match(bad_table, used_cols, known_tables):
    # Try to find best match where the table has the same columns
    #for table, columns in known_tables.items():
    #    if used_cols and used_cols.issubset(columns):
    #        return table
    #return None

def best_table_match(original_table: str, columns: set[str], known_tables: dict) -> str | None:

    candidates = []
    for table, table_cols in known_tables.items():
        overlap = len(columns & table_cols)
        score = overlap + (1 if table.startswith(original_table[:4]) else 0)
        candidates.append((table, score))

    # Pick the table with the best score and at least one overlapping column
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates and candidates[0][1] > 0 else None

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_semantic_best_match(column: str, candidates: set, threshold=0.7) -> str | None:
    column_vec = model.encode(column)
    best_score = 1.0
    best_match = None
    for c in candidates:
        score = cosine(column_vec, model.encode(c))
        if score < best_score and score < (1 - threshold):
            best_score = score
            best_match = c
    return best_match

semantic_aliases = {
    "id": ['audit_id','claim_id','user_id','provider_id'],
    "type": ["claim_type", "event_type", "document_type","plan_type","product_type"],
    "timestamp": ["event_time", "submitted_at","uploaded_at","sent_at"],
    "user": ["user_id", "performed_by"],
    "amount": ["amount_claimed", "amount_approved"],
}

from difflib import get_close_matches

def resolve_column_with_fallback(user_input_col: str, column_names: list[str], column_metadata: dict[str, str], semantic_aliases: dict[str, list[str]]) -> tuple[str | None, str]:
    # Fuzzy match
    from difflib import get_close_matches
    match = get_close_matches(user_input_col, column_names, n=1)
    if match:
        return match[0], "fuzzy"

    # Semantic alias match
    for canonical, aliases in semantic_aliases.items():
        if user_input_col.lower() == canonical.lower():
            for alias in aliases:
                if alias in column_names:
                    return alias, "semantic-alias"

    # Type match fallback (for inputs like 'timestamp', 'uuid')
    for col, col_type in column_metadata.items():
        base_type = col_type.split("(")[0].upper()
        if user_input_col.upper() == base_type:
            return col, "type-fallback"

    # Substring match (e.g., 'type' matches 'claim_type')
    for col in column_names:
        if user_input_col.lower() in col.lower():
            return col, "partial-substring"

    return "", "not-found"


def get_column_metadata_by_table(db_uri: str) -> dict[str, dict[str, str]]:
    """
    Returns a dict mapping table_name -> {column_name: column_type}
    """
    engine = create_engine(db_uri)
    inspector = inspect(engine)

    table_metadata = {}
    for table in inspector.get_table_names():
        column_info = inspector.get_columns(table)
        column_types = {
            col["name"].lower(): str(col["type"]).upper()
            for col in column_info
        }
        table_metadata[table.lower()] = column_types
    return table_metadata

metadata = get_column_metadata_by_table("sqlite:///claims.db")



def autocorrect_sql(sql: str, known_tables: dict) -> str:
    try:
        parsed = sqlglot.parse_one(sql)
        corrections = {}

        # Step 1: Map aliases to their actual table names
        alias_map = {
            alias.name.lower(): table.name.lower()
            for table in parsed.find_all(sqlglot.exp.Table)
            if (alias := table.args.get("alias")) and isinstance(alias, sqlglot.exp.TableAlias)
        }

        # Step 2: Fix unknown table names
        for table in parsed.find_all(sqlglot.exp.Table):

            original = table.name.lower()
            print("original:",original)
            alias = table.args.get("alias")

            if original not in known_tables:
                close_matches = get_close_matches(original, known_tables.keys(), n=1, cutoff=0.6)
                print("close_matches:",close_matches)
                if close_matches:
                    corrected = close_matches[0]
                    corrections[original] = corrected
                    table.set("this", corrected)
                    if alias:
                        alias_map[alias.name.lower()] = corrected

        # Step 3: Fix column names based on resolved table names
        for col in parsed.find_all(sqlglot.exp.Column):
            original_col = col.name.lower()
            print("original_col:",original_col)
            alias = col.table.lower() if col.table else None
            print("alias:",alias)
            table_name = alias_map.get(alias, alias)
            print("table_name:",table_name)

            possible_cols = (
                known_tables.get(table_name, set()) if table_name else set().union(*known_tables.values())
            )
            print("possible_cols:",possible_cols)
            if original_col not in possible_cols:
                best_col = get_close_matches(original_col, possible_cols, n=1, cutoff=0.6)
                if best_col and best_col[0] != original_col:
                    corrections[original_col] = best_col[0]
                    col.set("this", best_col[0])
                    print('Found the closest column with get_close_matches',best_col)
                if not best_col:
                        print("Not able to find column with get_close_matches for the original column:",original_col)
                        best_semantic = get_semantic_best_match(original_col, possible_cols)
                        if best_semantic:
                            corrections[original_col] = best_semantic
                            col.set("this", best_semantic)
                            print('Found the closest column with best semantic',best_semantic)
                        else:
                            print("Not able to find column with semantic search for the original column:",original_col)
                        if not best_semantic:
                            table_meta = metadata[table_name]
                            col_names = list(table_meta.keys())
                            resolved_col, method = resolve_column_with_fallback(
                                user_input_col=original_col,
                                column_names=col_names,
                                column_metadata=table_meta,
                                semantic_aliases=metadata
                            )
                            if resolved_col:
                                print('Found the closest column with semantic alias',resolved_col)
                                corrections[original_col] = resolved_col
                                col.set("this", resolved_col)
                            else:
                                print("Not able to find column with semantic aliases for the original column:",original_col)


        if corrections:
            print(f"✅ Corrections applied: {corrections}")

        return parsed.sql()

    except sqlglot.errors.ParseError as e:
        print(f"❌ Parse error: {e}")
        return sql  # fallback to original SQL if parsing fails

def _validate(parsed):
        alias_map = {
            alias.name.lower(): table.name.lower()
            for table in parsed.find_all(sqlglot.exp.Table)
            if (alias := table.args.get("alias")) and isinstance(alias, sqlglot.exp.TableAlias)
        }

        # Validate table names
        for table in parsed.find_all(sqlglot.exp.Table):
            name = table.name.lower()
            print("name:",name)
            if name not in KNOWN_TABLES:
                raise ValueError(f"❌ Unknown table: '{name}'")

        # Validate column names
        for col in parsed.find_all(sqlglot.exp.Column):
            col_name = col.name.lower()
            alias = col.table.lower() if col.table else None
            table_name = alias_map.get(alias, alias) if alias else None

            if table_name:
                if table_name not in KNOWN_TABLES:
                    raise ValueError(f"❌ Unknown table alias: '{alias}' → '{table_name}'")
                if col_name not in KNOWN_TABLES[table_name]:
                    raise ValueError(f"❌ Unknown column '{col_name}' in table '{table_name}'")
            elif not any(col_name in cols for cols in KNOWN_TABLES.values()):
                raise ValueError(f"❌ Unknown column: '{col_name}'")


db = SQLDatabase.from_uri("sqlite:///claims.db")  # Replace with your actual DB URI

def execute_sql_query(state: SQLAgentState) -> SQLAgentState:
    """
    Executes the generated SQL query using the configured SQL database.

    Args:
        state (SQLAgentState): The current state, must include 'sql_query'.

    Returns:
        SQLAgentState: Updated state including the result of the SQL execution.
    """
    sql = state["sql_query"]
    print("✅ SQL Query Exec:", sql)

    try:
        #print(" db.run(sql) :", db.run(sql))
        result = db.run(sql)
        print("📄 Result Preview:", str(result)[:200])
        return {**state, "execution_result": str(result)}  # ✅ Wrap inside state

    except Exception as e:
        
        result = f"❌ Error executing SQL: {e}"
        print(result)
        return {**state, "execution_result": f"❌ Error executing SQL: {e}"}
    

from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from typing import Dict
import uuid


def claims_agent_node() -> Runnable:
    # 1. Connect to MCP tool server
    client = MultiServerMCPClient({
        "claims": {
            "command": "python",
            "args": ["mcp_test_server.py"],
            "transport": "stdio"
        }
    })

    # 2. Get tools (awaited)
    async def setup_agent():
        tools = await client.get_tools()
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        print("✅ Agent is being created with tools")
        return create_react_agent(llm, tools=tools )

    # 3. Wrap into a LangGraph-compatible function node
    #async def run_tool_agent(state: Dict) -> Dict:
    async def run_tool_agent(state: SQLAgentState) -> SQLAgentState:
        agent = await setup_agent()
        config={
            "configurable": {
                #"thread_id": state.get("config", {}).get("thread_id", "claims-thread")
                "thread_id": str(uuid.uuid4)
            }
        }
        response = await agent.ainvoke(
            {"messages": state["messages"]},
            config=config
        )
        ai_messages = [msg for msg in response["messages"] if isinstance(msg, AIMessage) and msg.content]
        final_response = ai_messages[-1].content if ai_messages else "No AI response found."
        print("response in :",final_response)
        return {**state, "execution_result": final_response}

    print("✅ claims_agent_node initialized")
    return RunnableLambda(run_tool_agent)


def validate_sql_query(state) -> dict:
    query = state.get("sql_query")
    print("🔍 Query in validation:", query)

    try:
        parsed = sqlglot.parse_one(query)
        _validate(parsed)
        print("✅ SQL is valid")
        return {**state, "sql_valid": True}

    except Exception as e:
        print("⚠️ Validation failed:", e)
        corrected_sql = autocorrect_sql(query, KNOWN_TABLES)
        
        if corrected_sql != query:
            try:
                parsed = sqlglot.parse_one(corrected_sql)
                _validate(parsed)
                print("✅ Autocorrected SQL is valid")                    
                return {**state, "sql_query": corrected_sql, "sql_valid": True, "correction_note": "Autocorrected"}
            except Exception as e2:
                print("❌ Autocorrected SQL still invalid:", e2)
                auto_corrected_sql2 = autocorrect_sql(corrected_sql, KNOWN_TABLES)
                try:
                    _validate(sqlglot.parse_one(auto_corrected_sql2))
                    print("✅ Autocorrected SQL second attempt is valid")
                    return {**state, "sql_query": auto_corrected_sql2, "sql_valid": True, "correction_note": "Autocorrected"}
                except Exception as e3:
                    print("✅ Autocorrected SQL second attempt is invalid", e3)
                    return {**state, "sql_valid": False, "result": f"❌ Still invalid after autocorrect: {str(e3)}"}

        return {**state, "sql_valid": False, "result": f"❌ Invalid SQL: {str(e)}"}

def format_response(state: SQLAgentState) -> SQLAgentState:
    
    response = f"Here are the results:\n{state['execution_result']}"
    print("response in format_reponse:",response)
    print("AIMessage(content=response):", AIMessage(content=response))
    return {
        **state,
        "final_response": AIMessage(content=response)
    }

def show_graph(graph, xray=False):
    """
    Display a LangGraph mermaid diagram with fallback rendering.
    
    This function attempts to render a LangGraph as a visual diagram using Mermaid.
    It includes error handling to fall back to an alternative renderer if the default fails.
    
    Args:
        graph: The LangGraph object that has a get_graph() method for visualization
        xray (bool): Whether to show internal graph details in xray mode
        
    Returns:
        Image: An IPython Image object containing the rendered graph diagram
    """
    from IPython.display import Image
    
    try:
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        print(f"Default renderer failed ({e}), falling back to pyppeteer...")
        
        import nest_asyncio
        nest_asyncio.apply()
        
        from langchain_core.runnables.graph import MermaidDrawMethod
        
        return Image(graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER))
    

from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from typing import Dict
import uuid

def document_qa(state: SQLAgentState) -> SQLAgentState:
    try:
        agent = create_document_qa_graph()
        question = state["messages"][-1].content
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        print("\n💬 Answer:", result["answer"])

        return { **state,"final_response": AIMessage(content=result["answer"])}
    except Exception as e:
        result = f"❌ Error while fetching from docs: {e}"
        print(result)
        return {**state, "final_response": f"❌ Error while fetching from docs: {e}"}

#class SQLAgentState(TypedDict):
#    messages: List[BaseMessage]
    
builder = StateGraph(SQLAgentState)
builder.add_node("planner", planner)
builder.add_node("claims_agent", claims_agent_node())
builder.add_node("document_qa", document_qa)

builder.add_node("retrieve_schema", retrieve_schema_context)
builder.add_node("generate_sql", generate_sql_query)
builder.add_node("validate_sql", validate_sql_query)
builder.add_node("execute", execute_sql_query)

builder.add_node("format_response", format_response)

builder.add_conditional_edges("planner", lambda state: state["route"], {
    "tool": "claims_agent",
    "sql": "retrieve_schema",
    "doc":"document_qa"
})

# Entry → Planner
#builder.set_entry_point("retrieve_schema")
builder.set_entry_point("planner")
#builder.add_edge("retrieve_schema", "planner")

# SQL path
builder.add_edge("retrieve_schema","generate_sql")
builder.add_edge("generate_sql", "validate_sql")
builder.add_edge("validate_sql", "execute")
builder.add_edge("execute", "format_response")
builder.add_edge("claims_agent", "format_response")

#builder.add_edge("document_qa", "format_response")
# Final output
builder.set_finish_point("format_response")
builder.set_finish_point("document_qa")

# Compile
in_memory_store = InMemoryStore()
checkpointer = MemorySaver()
claims_assisting_agent = builder.compile(name="claims_assisting_agent", checkpointer=checkpointer, store = in_memory_store)


show_graph(claims_assisting_agent)


@app.post("/login")
def login(data: LoginInput):
    graph = build_auth_graph(token_based=False)
    try:
        result = graph.invoke({"username": data.username, "password": data.password})
        print(result)
        token = generate_token(result["username"], result["role"])
        print("token", token , "username", result["username"],"role", result["role"])
        return {"token": token, "username": result["username"], "role": result["role"]}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/validate-token")
def validate_token(data: TokenInput):
    graph = build_auth_graph(token_based=True)
    try:
        result = graph.invoke({"token": data.token})
        return {"status": "valid", "username": result["username"], "role": result["role"]}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

def get_current_user(request: Request) -> dict:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = auth_header.replace("Bearer ", "").strip()
    try:
        return verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@app.get("/me")
def get_me(user=Depends(get_current_user)):
    return {
        "username": user["username"],
        "role": user["role"]
    }

@app.get("/admin-area")
def admin_dashboard(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied. Admins only.")
    return {"message": f"Welcome Admin #{user['username']}"}

@app.get("/users")
def list_users(user=Depends(get_current_user)):
    print("here in users")
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    conn = sqlite3.connect("claims.db")
    c = conn.cursor()
    c.execute("SELECT  username, role, last_login FROM auth_users")
    rows = c.fetchall()
    conn.close()
    return [{"UserName": r[0], "Role": r[1], "Last Login": r[2]} for r in rows]

class RegisterInput(BaseModel):
    username: str
    password: str
    role: str  # new field

@app.post("/register")
def register_user(data: RegisterInput, user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admin can register users")

    conn = sqlite3.connect("claims.db")
    c = conn.cursor()

    if c.execute("SELECT * FROM users WHERE username=?", (data.username,)).fetchone():
        raise HTTPException(status_code=409, detail="User already exists")

    c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
              (data.username, hash_password(data.password), data.role))
    conn.commit()
    conn.close()
    return {"username": data.username}

def extract_user_id_from_text(text: str) -> str | None:
    # Look for "for user123", "of user_abc", etc.
    match = re.search(r"\b(?:for|of|about)\s+user[_\-]?(?P<uid>[\w\-]+)", text)
    if match:
        return f"user{match.group('uid')}"  # restore 'user' prefix if missing
    # Fallback: just try to find "userXYZ"
    match = re.search(r"user[_\-]?([\w\-]+)", text)
    return match.group(0) if match else None

async def claims_assisting_agent_invoke(question: str):
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": str(thread_id)}}
    answer =  await claims_assisting_agent.ainvoke(
            {
                "messages": [HumanMessage(content=question)],
            },config=config
        )  
    return answer
    
@app.post("/chat")
async def chat_with_agent(request: ChatRequest, user=Depends(get_current_user)):
    print(user)
    role = user["role"]
    print('role:',role)
    username = user["user_id"]
    print('username:',username)
    query_text = request.query
    print('query_text:',query_text)
    query_user_id = extract_user_id_from_text(query_text)
    print('query_user_id:',query_user_id)

    if not query_text:
        raise HTTPException(status_code=400, detail="Query is required")

    if role == "user":
        # If user tries to query another user's claims
        if query_user_id and query_user_id != username:
            raise HTTPException(status_code=403, detail="Users can only access their own claims.")
            
        # If no user_id in query, auto-inject
        if not query_user_id:
            query_text += f" where user_id = '{username}'"

    question = f"{query_text}. Only return info for user_id {username}."
    
    try:
        thread_id = uuid.uuid4()
        config = {"configurable": {"thread_id": str(thread_id)}}
        #answer = claim_information_subagent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        #answer = sql_graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        answer = await claims_assisting_agent_invoke(question)
        final_message = answer.get("final_response")
        #final_message = next((m.content for m in reversed(messages) if m.type == "ai"), None)
        if not final_message or not isinstance(final_message, AIMessage):
            raise ValueError("No AI message returned.")

        return {"answer": final_message.content}


    except Exception as e:
        print("❌ ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

