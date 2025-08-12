import streamlit as st
import requests

API_URL = "http://localhost:8081"

st.set_page_config(page_title="🔐 MCP Auth System", layout="centered")
st.title("💬 Claim Assistant")

# Session & memory setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "token" not in st.session_state:
    st.session_state["token"] = None
if "user_info" not in st.session_state:
    st.session_state["user_info"] = {}

# Navigation
st.sidebar.header("Navigation")
role = st.session_state.get("user_info", {}).get("role")
is_logged_in = st.session_state["token"] is not None

menu_options = ["🔑 Login", "🧾 Validate Token"]
if is_logged_in and role == "admin":
    menu_options += ["👥 Register User (admin)", "📋 View User List (admin)"]
menu_options.append("🚪 Logout")
menu = st.sidebar.radio("Go to", menu_options)

st.sidebar.markdown("---")
if is_logged_in:
    st.sidebar.success(f"Logged in as: {role}")
else:
    st.sidebar.warning("Not logged in")

# 🔐 LOGIN + CHAT
if menu == "🔑 Login":

    if not is_logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            res = requests.post(f"{API_URL}/login", json={
                "username": username,
                "password": password
            })
            if res.status_code == 200:
                data = res.json()
                st.session_state["token"] = data["token"]
                st.session_state["user_info"] = {
                    "username": data["username"],
                    "role": data["role"]
                }
                st.success(f"Welcome, #{data['username']} ({data['role']})")
                st.rerun()
            else:
                st.error(f"Login failed: {res.json().get('detail')}")
    else:
        #st.markdown("### 💬 Ask about your claim")

        user_query = st.chat_input("Ask about your claim...")
        if user_query:
            #st.chat_message("user").write(user_query)
            st.session_state.chat_history.append(("user", user_query))

            headers = {"Authorization": f"Bearer {st.session_state['token']}"}
            API_URL_AGENT = "http://localhost:8082"
            res = requests.post(f"{API_URL}/chat", json={"query": user_query}, headers=headers)
            if res.status_code == 200:
                answer = res.json()["answer"]
            else:
                answer = f"❌ {res.json().get('detail', 'Error')}"

            #st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append(("assistant", answer))

        # Display chat history
        for role, message in st.session_state.chat_history:
            st.chat_message(role).write(message)

        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ✅ VALIDATE TOKEN PAGE
elif menu == "🧾 Validate Token":
    st.title("🧾 Validate Token")
    token = st.text_area("Paste JWT Token", value=st.session_state.get("token", ""))
    if st.button("Validate Token"):
        res = requests.post(f"{API_URL}/validate-token", json={"token": token.strip()})
        if res.status_code == 200:
            data = res.json()
            st.success(f"✅ Token valid. User #{data['username']} (role: {data['role']})")
        else:
            st.error(f"❌ {res.json().get('detail')}")

# ➕ REGISTER USER (ADMIN ONLY)
elif menu == "👥 Register User (admin)":
    st.title("👥 Admin - Register New User")
    if not is_logged_in or role != "admin":
        st.warning("Admins only.")
    else:
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        new_role = st.selectbox("Assign Role", ["user", "admin"])

        if st.button("Register User"):
            headers = {"Authorization": f"Bearer {st.session_state['token']}"}
            res = requests.post(f"{API_URL}/register", json={
                "username": new_username,
                "password": new_password,
                "role": new_role
            }, headers=headers)
            if res.status_code == 200:
                st.success(f"✅ Registered: {res.json()['username']}")
            else:
                st.error(f"❌ {res.json().get('detail')}")

# 📋 VIEW USER LIST (ADMIN ONLY)
elif menu == "📋 View User List (admin)":
    st.title("📋 Admin - User List")
    if not is_logged_in or role != "admin":
        st.warning("Admins only.")
    else:
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}
        res = requests.get(f"{API_URL}/users", headers=headers)
        if res.status_code == 200:
            st.table(res.json())
        else:
            st.error("Unable to fetch users.")

# 🚪 LOGOUT
elif menu == "🚪 Logout":
    st.title("🚪 Logout")
    if is_logged_in:
        st.session_state["token"] = None
        st.session_state["user_info"] = {}
        st.session_state["chat_history"] = []
        st.success("Logged out.")
        st.rerun()
    else:
        st.info("You're already logged out.")
