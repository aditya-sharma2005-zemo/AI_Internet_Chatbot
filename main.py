# from dotenv import load_dotenv
# load_dotenv()

# from langchain_mistralai import ChatMistralAI
# from tavily import TavilyClient
# from langchain.agents import create_agent
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# from langchain.tools import tool
# import os

# tavily_client=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# @tool
# def search_from_internet(query:str):
#     """
#     Use this tool to get latest information from the internet.
#     """
#     result=tavily_client.search(query=query)
#     return str(result)

# Model=ChatMistralAI(
#     model='mistral-small-latest',
# )

# agent=create_agent(
#     model=Model,
#     tools=[search_from_internet]
# )

# response=agent.invoke({
#     "messages":[HumanMessage("What is the advantage of using X-shaped rudder and pumpjet + Nuclear_electric Propulsion system combined....how much sound reduction is achieved in this configuration than conventional system give me factual values?")]
# })
# print(response["messages"][-1].text)
    





# result=tavily_client.search(query="What is the advantage of using X-shaped rudder and pumpjet + Nuclear_electric Propulsion   system combined....how much sound reduction is achieved in this configuration than conventional system give me factual values?")

# print(result)





from dotenv import load_dotenv
load_dotenv()

import os

from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool

# ------------------ Tavily ------------------

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_from_internet(query: str):
    """
    Use this tool to get the latest information from the internet.
    """
    result = tavily_client.search(query=query)
    return str(result)

# ------------------ Model ------------------

model = ChatMistralAI(
    model="mistral-small-latest",
)

# ------------------ Agent ------------------

agent = create_agent(
    model=model,
    tools=[search_from_internet]
)

# ------------------ Chat Memory ------------------

messages = []

print("AI Chatbot")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("\nYou : ")

    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break

    # Save user message
    messages.append(HumanMessage(content=user_input))

    # Invoke agent with full conversation history
    response = agent.invoke({
        "messages": messages
    })
    
    # Save all new messages returned by the agent
    new_messages = response["messages"][len(messages):]

    messages.extend(new_messages)

    # Print only the final AI response
    last_message = messages[-1]

    if isinstance(last_message.content, list):
        print("\nAI :", "".join(
            block["text"] for block in last_message.content
        if block.get("type") == "text"
        ))
    else:
        print("\nAI :", last_message.content)
    
