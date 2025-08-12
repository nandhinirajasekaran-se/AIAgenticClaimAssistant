from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, BaseMessage
from operator import add as add_messages
from embedder import create_graph, compute_similarity, load_schema_chunks, model
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase


class SQLAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sql_query: str
    retrieved_schema: list[dict]
    result: str


def retrieve_schema_context(state: SQLAgentState) -> SQLAgentState:
    question = state["messages"][-1].content
    query_embedding = model.encode(question)

    matches = []
    schema_chunks = load_schema_chunks("schema.sql")  # path relative to your project
    G = create_graph(schema_chunks)
    for _, data in G.nodes(data=True):
        similarity = compute_similarity(data["embedding"], query_embedding)
        if similarity > 0.7:
            matches.append({
                "text": data["text"],
                "similarity": similarity,
                "metadata": data.get("metadata", {})
            })

    # Sort and keep top 5
    top_matches = sorted(matches, key=lambda x: -x["similarity"])[:5]
    return {**state, "retrieved_schema": top_matches}




sql_generator_prompt = PromptTemplate.from_template("""
You are an expert SQL agent. Given the user question and the database schema context, generate an executable SQLite SQL query.

Schema:
{schema}

User Question:
{question}

Rules:
- Only generate SQL, no explanation.
- Only use columns and tables in the schema
- Target correct table and columns.
- Ensure joins are valid
- Do NOT hallucinate table names
- Use lowercase for SQL keywords.
- Don't include `;` at the end.

SQL Query:
""")

def generate_sql_query(state: SQLAgentState) -> SQLAgentState:
    schema_text = "\n\n".join([chunk["text"] for chunk in state["retrieved_schema"]])
    question = state["messages"][-1].content
    #prompt = sql_generator_prompt.format(schema=schema_text, question=question)

    prompt_input = sql_generator_prompt.format(
        schema="\n".join([d["text"] for d in state["retrieved_schema"]]),
        question=state["messages"][-1].content
    )

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    sql = llm.invoke(prompt_input).content.strip()
    return {**state, "sql_query": clean_sql_response(sql)}

# You can pass this in as a parameter instead if needed
db = SQLDatabase.from_uri("sqlite:///claims.db")  # Replace with your actual DB URI

def clean_sql_response(text: str) -> str:
    """
    Remove markdown formatting and extract the raw SQL query.
    """
    # Remove ```sql and ``` blocks if present
    return text.strip().removeprefix("```sql").removesuffix("```").strip()


def execute_sql_query(state: SQLAgentState) -> SQLAgentState:
    """
    Executes the generated SQL query using the configured SQL database.

    Args:
        state (SQLAgentState): The current state, must include 'sql_query'.

    Returns:
        SQLAgentState: Updated state including the result of the SQL execution.
    """
    sql = state["sql_query"]
    print("✅ SQL Query:", state["sql_query"])

    try:
        sql = clean_sql_response(sql)
        print('SQL:',sql)
        print(" db.run(sql) :", db.run(sql))
        result = db.run(sql)
        #print("📄 Result Preview:", state["result"][:200])
        return {**state, "result": str(result)}  # ✅ Wrap inside state

    except Exception as e:
        result = f"❌ Error executing SQL: {e}"
        return {**state, "result": f"❌ Error executing SQL: {e}"}


from langchain_core.messages import AIMessage

def format_response(state: SQLAgentState) -> SQLAgentState:
    response = f"Here are the results:\n{state['result']}"
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)]
    }


import sqlglot
from difflib import get_close_matches

from sqlalchemy import create_engine, inspect

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

print("KNOWN_TABLES:", KNOWN_TABLES)

def best_table_match1(table_name: str, used_columns: set[str], known_tables: dict) -> str | None:
    candidates = []
    for tname, columns in known_tables.items():
        overlap = len(used_columns & columns)
        if overlap:
            score = overlap + (1 if tname.startswith(table_name[:3]) else 0)
            candidates.append((tname, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates else None

def best_table_match(bad_table, used_cols, known_tables):
    # Try to find best match where the table has the same columns
    for table, columns in known_tables.items():
        if used_cols and used_cols.issubset(columns):
            return table
    return None



def autocorrect_sql(sql: str, known_tables: dict) -> str:
    try:
        parsed = sqlglot.parse_one(sql)
        corrections = {}

        # Correct table names
        for table in parsed.find_all(sqlglot.exp.Table):
            original = table.name
            used_cols = set() 

            if original not in known_tables:
                used_cols = {
                    col.name.lower()
                    for col in parsed.find_all(sqlglot.exp.Column)
                    if col.table and col.table.lower() == original
                }
            
                best_match = best_table_match(original, used_cols, known_tables)
                if best_match:
                    corrections[original] = best_match
                    table.set("this", best_match)

            elif original.lower() not in corrections:
                corrected = get_close_matches(original.lower(), known_tables.keys(), n=1, cutoff=0.6)
                if corrected and corrected[0] != original.lower():
                    corrections[original] = corrected[0]
                    table.set("this", corrected[0])

        # Correct column names
        for col in parsed.find_all(sqlglot.exp.Column):
            original = col.name
            table = col.table.lower() if col.table else None

            possible_cols = (
                known_tables.get(table, set()) if table else set().union(*known_tables.values())
            )
            
            corrected = get_close_matches(original.lower(), possible_cols, n=1, cutoff=0.6)
            if corrected and corrected[0] != original.lower():
                if not table or corrected[0] in known_tables.get(table, set()):
                    corrections[original] = corrected[0]
                    col.set("this", corrected[0])

        if corrections:
            print(f"✅ Corrections applied: {corrections}")

        return parsed.sql()

    except sqlglot.errors.ParseError as e:
        print(f"❌ Parse error during autocorrect: {e}")
        return sql  # return original if unfixable

def validate_sql_query(state) -> dict:
    query = state.get("sql_query")
    print("query in validation:", query)
    try:
        parsed = sqlglot.parse_one(query)
        # validate as before...
        print("Table:", parsed.find_all(sqlglot.exp.Table))
        for table in parsed.find_all(sqlglot.exp.Table):
            if table.name.lower() not in KNOWN_TABLES:
                raise ValueError(f"Unknown table: {table.name}")
        print("Column:", parsed.find_all(sqlglot.exp.Column))
        for col in parsed.find_all(sqlglot.exp.Column):
            col_name = col.name.lower()
            table = col.table.lower() if col.table else None
            if table:
                if col_name not in KNOWN_TABLES.get(table, set()):
                    raise ValueError(f"Invalid column '{col_name}' in table '{table}'")
            elif not any(col_name in cols for cols in KNOWN_TABLES.values()):
                raise ValueError(f"Unknown column: {col_name}")

        print("sql_valid No issue")
        return {**state, "sql_valid": True}

    except Exception as e:
        corrected_sql = autocorrect_sql(query, KNOWN_TABLES)
        if corrected_sql != query:
            try:
                parsed = sqlglot.parse_one(corrected_sql)
                for table in parsed.find_all(sqlglot.exp.Table):
                    if table.name.lower() not in KNOWN_TABLES:
                        raise ValueError(f"Unknown table: {table.name}")
                # same column checks...
                print("corrected_sql:",corrected_sql)
                return {**state, "sql_query": corrected_sql, "sql_valid": True, "correction_note": "Autocorrected"}
            except Exception as e2:
                return {**state, "sql_valid": False, "result": f"❌ Still invalid: {str(e2)}"}
        return {**state, "sql_valid": False, "result": f"❌ Invalid SQL: {str(e)}"}

    #except Exception as e:
        # Try to autocorrect
    #    corrected_sql = autocorrect_sql(query, KNOWN_TABLES)
    #    if corrected_sql != query:
    #        return {**state, "sql_query": corrected_sql, "sql_valid": True, "correction_note": "Corrected syntax"}
    #    return {**state, "sql_valid": False, "result": f"❌ Invalid SQL: {str(e)}"}
