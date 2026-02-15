import os
from typing import List, Literal, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import API keys from config
from config import GROQ_API_KEY, TAVILY_API_KEY
from vectorstore import get_retriever

# --- Tools ---
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tavily = TavilySearch(max_results=3, topic="general")


class RouteDecision(BaseModel):
    route: Literal["rag", "web", "answer", "end"]
    reply: str | None = Field(None, description="filled only when route =='end'")
    

class AgentState(TypedDict, total=False):
    message: List[BaseMessage]
    route: Literal["rag", "web", "answer", "end"]
    rag: str
    web: str
    web_search_enabled: bool


