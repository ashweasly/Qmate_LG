# langgraph_analyzer.py
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv
import os
from ragtemplate.llmModel import llmChatModel
import ast
import re

load_dotenv()

# ============ STATE DEFINITION ============
class AnalysisState(TypedDict):
    """Centralized state management for the analysis workflow"""
    file_list: list[str]
    user_question: str
    current_file_index: int
    current_file_path: str
    file_content: str
    file_type: str
    analysis_result: str
    dependencies: dict
    all_results: dict
    dep_graph: nx.DiGraph
    error: str

# ============ INITIALIZE LLM ============
model_url = os.getenv("MODEL_URL")
model_config = [{
    "modelPlatform": "ITAAP",
    "modeltype": "AzureAI",
    "model": "gpt-4o",
    "modelId": "gpt-4o-itaap",
    "modelUrl": model_url,
    "modelVersion": "2024-08-01-preview",
    "client_id": os.getenv("CLIENT_ID"),
    "client_secret": os.getenv("CLIENT_SECRET"),
}]
llm = llmChatModel("gpt-4o-itaap", model_config=model_config).get_model()

# ============ UTILITY FUNCTIONS ============
def read_text_from_file(filename: str) -> str:
    """Read file content with error handling"""
    try:
        file_path = os.path.join(os.path.dirname(__file__), filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File '{filename}' not found."
    except Exception as e:
        return f"Error: {str(e)}"

def extract_dependencies(code: str, code_type: str) -> dict:
    """Extract table names and dependencies from code"""
    dependencies = {'tables': set(), 'files': set(), 'imports': set()}
    
    if code_type.lower() == 'sql':
        from_tables = re.findall(r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', code, re.IGNORECASE)
        join_tables = re.findall(r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', code, re.IGNORECASE)
        dependencies['tables'].update(from_tables + join_tables)
    
    elif code_type.lower() == 'python':
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        dependencies['imports'].add(name.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    dependencies['imports'].add(node.module)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ['open', 'read_csv', 'load']:
                        if len(node.args) > 0 and isinstance(node.args[0], (ast.Str, ast.Constant)):
                            val = node.args[0].s if isinstance(node.args[0], ast.Str) else node.args[0].value
                            if isinstance(val, str):
                                dependencies['files'].add(val)
        except:
            pass
    
    return {k: list(v) for k, v in dependencies.items()}

# ============ LANGGRAPH NODES ============
def initialize_state(state: AnalysisState) -> AnalysisState:
    """Initialize the analysis workflow"""
    state["all_results"] = {}
    state["dep_graph"] = nx.DiGraph()
    state["current_file_index"] = 0
    state["error"] = ""
    return state

def load_file(state: AnalysisState) -> AnalysisState:
    """Load next file"""
    if state["current_file_index"] >= len(state["file_list"]):
        return state
    
    file_path = state["file_list"][state["current_file_index"]]
    state["current_file_path"] = file_path
    state["file_content"] = read_text_from_file(file_path)
    
    if state["file_content"].startswith("Error:"):
        state["error"] = state["file_content"]
        return state
    
    state["file_type"] = 'sql' if file_path.endswith('.sql') else 'python'
    return state

def analyze_file(state: AnalysisState) -> AnalysisState:
    """Analyze the loaded file with LLM"""
    if state["error"]:
        return state
    
    prompt = f"""You are a code analysis expert. Analyze this {state['file_type'].upper()} code:

Code:
{state['file_content']}

Provide:
1. Filter conditions in plain English
2. Key metrics/KPIs
3. Table/file dependencies
4. Use bold headings and clear formatting

"""
    if state["user_question"]:
        prompt += f"\nUser question: {state['user_question']}"
    
    try:
        response = llm.invoke(prompt)
        state["analysis_result"] = response.content if hasattr(response, 'content') else str(response)
        state["dependencies"] = extract_dependencies(state["file_content"], state["file_type"])
    except Exception as e:
        state["error"] = f"LLM Error: {str(e)}"
    
    return state

def store_result(state: AnalysisState) -> AnalysisState:
    """Store analysis result and update dependency graph"""
    if state["error"]:
        return state
    
    file_path = state["current_file_path"]
    
    # Store result
    state["all_results"][file_path] = {
        'analysis': state["analysis_result"],
        'dependencies': state["dependencies"],
        'file_name': Path(file_path).name
    }
    
    # Update dependency graph
    state["dep_graph"].add_node(file_path)
    for table in state["dependencies"].get('tables', []):
        state["dep_graph"].add_edge(file_path, f"table:{table}")
    for imp in state["dependencies"].get('imports', []):
        state["dep_graph"].add_edge(file_path, f"import:{imp}")
    for dep_file in state["dependencies"].get('files', []):
        state["dep_graph"].add_edge(file_path, f"file:{dep_file}")
    
    return state

def next_file(state: AnalysisState) -> AnalysisState:
    """Move to next file"""
    state["current_file_index"] += 1
    state["error"] = ""
    return state

# ============ CONDITIONAL EDGES ============
def should_continue(state: AnalysisState) -> str:
    """Determine if we should process more files"""
    if state["current_file_index"] < len(state["file_list"]):
        return "load_file"
    return END

def should_store(state: AnalysisState) -> str:
    """Determine if we should store the result"""
    if state["error"]:
        return "next_file"
    return "store_result"

# ============ BUILD GRAPH ============
def create_analysis_graph():
    """Create and compile the LangGraph workflow"""
    workflow = StateGraph(AnalysisState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("load_file", load_file)
    workflow.add_node("analyze_file", analyze_file)
    workflow.add_node("store_result", store_result)
    workflow.add_node("next_file", next_file)
    
    # Add edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "load_file")
    workflow.add_edge("load_file", "analyze_file")
    workflow.add_conditional_edges("analyze_file", should_store)
    workflow.add_edge("store_result", "next_file")
    workflow.add_conditional_edges("next_file", should_continue)
    
    return workflow.compile()

# ============ PUBLIC API ============
def analyze_multiple_files(file_list: list[str], user_question: str = None) -> tuple:
    """Main entry point for analysis"""
    graph = create_analysis_graph()
    
    initial_state: AnalysisState = {
        "file_list": file_list,
        "user_question": user_question or "",
        "current_file_index": 0,
        "current_file_path": "",
        "file_content": "",
        "file_type": "",
        "analysis_result": "",
        "dependencies": {},
        "all_results": {},
        "dep_graph": nx.DiGraph(),
        "error": ""
    }
    
    final_state = graph.invoke(initial_state)
    
    return final_state["all_results"], final_state["dep_graph"]