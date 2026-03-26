import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

from app.agent.chroma import vector_store
from app.agent.state import AgentState

from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from app.agent.tools import retrieve_context
from phoenix.otel import register

from app.utils.helpers import load_prompt

# configure the Phoenix tracer
tracer_provider = register(
  project_name="rag_agent", # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed OI dependencies
)
load_dotenv()
DEFAULT_MODEL = os.getenv("MISC.DEFAULT_MODEL")
ollama_model = ChatOllama(model=DEFAULT_MODEL, temperature=0.7)


agent = create_agent(
    ollama_model,
    tools=[retrieve_context],
    system_prompt=load_prompt('rag', 'instructions.txt'),
)
