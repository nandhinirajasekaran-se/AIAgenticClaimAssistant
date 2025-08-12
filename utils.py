import sqlite3
import uuid
import jwt
from datetime import datetime, timedelta
from hashlib import sha256

SECRET = "super-secret-key"

def hash_password(password):
    return sha256(password.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect("claims.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users_data (id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT, role TEXT)")
    conn.commit()
    c.execute("SELECT * FROM users_data WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users_data (username, password_hash, role) VALUES (?, ?, ?)", 
                  ("admin", hash_password("admin123"), "admin"))
        conn.commit()
    conn.close()

    conn.close()


def get_user(username):
    conn = sqlite3.connect("claims.db")
    c = conn.cursor()
    print('get users')
    c.execute("SELECT user_id, password_hash, role FROM auth_users WHERE user_id=?", (username,))
    row = c.fetchone()
    conn.close()
    return row

def generate_token(user_id, role):
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        return {"user_id": payload["sub"], "role": payload["role"]}
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")

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