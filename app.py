import os
from dotenv import load_dotenv

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_groq import ChatGroq

from tools import get_tools

# Load environment variables
load_dotenv()

# Get API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192",
    temperature=0
)

# Load tools
tools = get_tools()

# Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run chatbot loop
print("AI Agent Started (type 'exit' to quit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = agent.run(user_input)

    print("Agent:", response)
