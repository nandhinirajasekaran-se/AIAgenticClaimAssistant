from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from utils import get_user, hash_password, verify_token

print("✅ auth_graph.py loaded")
class AgentState(TypedDict):
    username: Optional[str]
    password: Optional[str]
    token: Optional[str]
    user_id: Optional[int]
    role: Optional[str]

# Username/password auth step
def auth_with_credentials(state: AgentState) -> AgentState:
    user = get_user(state["username"])
    if not user:
        raise ValueError("User not found")
    if user[1] != hash_password(state["password"]):
        raise ValueError("Invalid password")
    return {**state, "user_id": user[0], "role": user[2]}

# Token-based auth step
def auth_with_token(state: AgentState) -> AgentState:
    print("🔐 Verifying token:", state["token"])
    verified = verify_token(state["token"])
    return {**state, "user_id": verified["user_id"], "role": verified["role"]}

#def rag_agent(state: dict) -> dict:
#    question = state["query"]
#    answer = rag_chain.run(question)
#    return {**state, "answer": answer}

# Graph
def build_auth_graph(token_based: bool = False):
    graph = StateGraph(AgentState)
    if token_based:
        graph.add_node("auth", auth_with_token)
    else:
        graph.add_node("auth", auth_with_credentials)

    graph.set_entry_point("auth")
    graph.add_edge("auth", END)
    return graph.compile()


from langchain.tools import tool
import sqlite3

DB_PATH = "claims.db"

def fetch(query: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return result

@tool
def get_claims_by_user(user_id: str) -> list[dict]:
    """
    Returns all claims for the given user_id.
    Useful to review user-specific claim history.
    """
    return fetch("SELECT * FROM claims WHERE user_id = ?", (user_id,))

@tool
def get_active_policies_by_user(user_id: str) -> list[dict]:
    """
    Retrieve all active insurance policies for a given user.
    """
    return fetch("""
        SELECT * FROM policies 
        WHERE user_id = ? AND active = 1
    """, (user_id,))



def fetch(query: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return result

@tool
def get_claims_by_user(user_id: str) -> list[dict]:
    """
    Returns all claims for the given user_id.
    Useful to review user-specific claim history.
    """
    return fetch("SELECT * FROM claims WHERE user_id = ?", (user_id,))

@tool
def get_active_policies_by_user(user_id: str) -> list[dict]:
    """
    Retrieve all active insurance policies for a given user.
    """
    return fetch("""
        SELECT * FROM policies 
        WHERE user_id = ? AND active = 1
    """, (user_id,))

@tool
def get_provider_plans(provider_id: str) -> list[dict]:
    """
    Return insurance plans offered by a specific provider.
    """
    return fetch("SELECT * FROM provider_plans WHERE provider_id = ?", (provider_id,))
