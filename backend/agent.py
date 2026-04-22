import os
from typing import List, Literal, TypedDict, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import API keys from config
from backend.config import GROQ_API_KEY, TAVILY_API_KEY
from backend.vectorstore import get_retriever

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

class RouteDecision(BaseModel):
    route: Literal["rag", "web", "answer"]
    reply: str | None = Field(None, description="filled only when answering immediately")

class AgentState(TypedDict, total=False):
    message: List[BaseMessage]
    route: str
    rag_docs: str
    web_docs: str
    web_search_enabled: bool

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
structured_llm_router = llm.with_structured_output(RouteDecision)

def router_node(state: AgentState):
    """Determine the next step based on user input."""
    messages = state.get("message", [])
    if not messages:
        return {"route": "answer"}
    last_message = messages[-1].content
    
    system_prompt = """You are an intelligent assistant routing user queries.
    Decide the appropriate tool to answer the user query:
    1. 'rag': For questions that likely depend on uploaded local knowledge/documents.
    2. 'web': For questions requiring recent information or real-time internet search.
    3. 'answer': For general conversation, greetings, or questions you can answer without external data.
    
    If web search is disabled, DO NOT choose 'web'.
    """
    
    # Check if web search is enabled
    web_enabled = state.get("web_search_enabled", True)
    if not web_enabled:
        system_prompt += "\nNOTE: Web search is DISABLED. Do not choose 'web'."
        
    messages_to_pass = [SystemMessage(content=system_prompt)] + messages
    decision = structured_llm_router.invoke(messages_to_pass)
    
    # Ensure fallback
    route = decision.route
    if route == "web" and not web_enabled:
        route = "rag" # Fallback to RAG or answer
        
    return {"route": route}

def rag_node(state: AgentState):
    """Retrieve documents from Pinecone."""
    messages = state.get("message", [])
    if not messages:
        return {"rag_docs": ""}
    last_message = messages[-1].content
    
    retriever = get_retriever()
    docs = retriever.invoke(last_message)
    doc_content = "\n\n".join([d.page_content for d in docs])
    
    return {"rag_docs": doc_content}

def web_search_node(state: AgentState):
    """Search the web using Tavily."""
    messages = state.get("message", [])
    if not messages:
        return {"web_docs": ""}
    last_message = messages[-1].content
    
    tool = TavilySearchResults(max_results=3)
    results = tool.invoke({"query": last_message})
    
    doc_content = "\n\n".join([f"Source: {res.get('url', 'N/A')}\n{res.get('content', '')}" for res in results])
    return {"web_docs": doc_content}

def generate_node(state: AgentState):
    """Generate the final answer based on all gathered info."""
    messages = state.get("message", [])
    rag_docs = state.get("rag_docs", "")
    web_docs = state.get("web_docs", "")
    
    context = ""
    if rag_docs:
        context += f"LOCAL KNOWLEDGE BASE:\n{rag_docs}\n\n"
    if web_docs:
        context += f"WEB SEARCH RESULTS:\n{web_docs}\n\n"
        
    system_prompt = "You are a helpful assistant. Use the provided context to answer the user's question. If you don't know the answer, just say you don't know. Do not make up information."
    if context:
        system_prompt += f"\n\nContext:\n{context}"
        
    messages_to_pass = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(messages_to_pass)
    
    return {"message": [response]}

# Define the graph
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("rag", rag_node)
workflow.add_node("web", web_search_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "router")

def decide_next(state: AgentState):
    return state.get("route", "answer")

workflow.add_conditional_edges(
    "router",
    decide_next,
    {
        "rag": "rag",
        "web": "web",
        "answer": "generate"
    }
)

workflow.add_edge("rag", "generate")
workflow.add_edge("web", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def run_agent(query: str, thread_id: str = "default", web_search_enabled: bool = True):
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "message": [HumanMessage(content=query)],
        "web_search_enabled": web_search_enabled
    }
    result = app.invoke(state, config=config)
    messages = result.get("message", [])
    if messages:
        return messages[-1].content
    return "No response generated."
