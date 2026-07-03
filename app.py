import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os

from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="AI Internet Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"  
)

st.markdown("""
<style>

/* Hide Streamlit branding */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header{
    background:transparent;
}

/* Main background */
/* Main background */
.stApp{
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #111827 50%,
        #0f172a 100%
    );
}

/* Chat container */
.block-container{
    max-width:900px;
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Chat bubbles */

[data-testid="stChatMessage"]{
    border-radius:18px;
    padding:15px;
    margin-bottom:12px;
}

/* User bubble */

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){
    background:#1e293b;
}

/* Assistant bubble */

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){
    background:#172554;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#111827;
}

/* Buttons */

.stButton>button{
    width:100%;
    border-radius:12px;
    height:45px;
    background:#2563eb;
    color:white;
    border:none;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1d4ed8;
}

/* Chat input */

.stChatInputContainer{
    padding-bottom:20px;
}

/* Title */

h1{
    text-align:center;
}

/* Caption */

p{
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Internet Chatbot")

st.markdown(
"""
<div style='text-align:center;color:#94a3b8;font-size:18px;margin-bottom:30px;'>

Powered by <b>Mistral AI</b> + <b>Tavily Search</b>

</div>
""",
unsafe_allow_html=True
)

# -------------------- TAVILY --------------------

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def search_from_internet(query: str):
    """
    Search latest information from the internet.
    """
    result = tavily_client.search(query=query)
    return str(result)

# -------------------- MODEL --------------------

model = ChatMistralAI(
    model="mistral-small-latest"
)

# -------------------- AGENT --------------------

agent = create_agent(
    model=model,
    tools=[search_from_internet]
)

# -------------------- SESSION MEMORY --------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- SIDEBAR --------------------

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        width=90
    )

    st.title("AI Assistant")

    st.write("---")

    st.success("🟢 Internet Search Enabled")

    st.write("")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages=[]
        st.session_state.history=[]
        st.rerun()

    st.write("---")

    st.caption("Built using")
    st.caption("• Mistral AI")
    st.caption("• Tavily Search")
    st.caption("• LangChain")
    st.caption("• Streamlit")

# -------------------- DISPLAY CHAT --------------------

for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# -------------------- USER INPUT --------------------

prompt = st.chat_input("Ask me anything...")

if prompt:

    # Display user message
    with st.chat_message("user", avatar="👨"):
        st.markdown(prompt)

    st.session_state.history.append({
        "role": "user",
        "content": prompt
    })

    # Save LangChain message
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # AI Response
    with st.chat_message("assistant", avatar="🤖"):

        with st.spinner("🔍 Searching the Internet..."):

            response = agent.invoke({
                "messages": st.session_state.messages
            })

            new_messages = response["messages"][len(st.session_state.messages):]

            st.session_state.messages.extend(new_messages)

            last_message = st.session_state.messages[-1]

            if isinstance(last_message.content, list):

                answer = ""

                for block in last_message.content:

                    if block.get("type") == "text":
                        answer += block["text"]

            else:
                answer = last_message.content

            st.markdown(
                f"""
                <div style="
                    background:#172554;
                    padding:18px;
                    border-radius:15px;
                    border-left:5px solid #3b82f6;
                    color:white;
                    line-height:1.7;
                    font-size:16px;
                ">
                {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.session_state.history.append({
                "role": "assistant",
                "content": answer
            })