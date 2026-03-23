from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace
from dotenv import load_dotenv
import requests
import json
import base64
import re
import os

load_dotenv()


class ITAAPLLMClient:
    def __init__(self):
        self.model_url = os.getenv("MODEL_URL")
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.model_scope = os.getenv("MODEL_SCOPE", "")
        self.token = self._get_token()
        
    def _get_token(self):
        tenant_id = os.getenv("AZURE_TENANT_ID")
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        
        scope = self.model_scope.replace('&scope=', '').replace('%3A', ':').replace('%2F', '/').replace('%2E', '.')
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': scope
        }
        
        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            return response.json()['access_token']
        except Exception as e:
            print(f"Token acquisition failed: {e}")
            raise
    
    def invoke(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000):
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
        }
        
        payload = {
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        
        endpoint = f"{self.model_url.rstrip('/')}/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
        
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.text}")
        
        result = response.json()
        return type('Response', (), {'content': result['choices'][0]['message']['content']})()


def get_databricks_client():
    host = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    token = os.getenv("DATABRICKS_ACCESS_TOKEN")
    
    if not host or not token:
        raise ValueError("Missing Databricks credentials in .env")
    
    return WorkspaceClient(host=host.rstrip('/'), token=token)


class KPIAgentState(TypedDict):
    user_question: str
    kpi_name: str
    search_keywords: List[str]
    found_notebooks: List[Dict]
    current_notebook_idx: int
    analyzed_notebooks: List[Dict]
    unresolved_tables: List[str]
    visited_notebooks: set
    depth: int
    max_depth: int
    combined_analysis: str
    flowchart: str
    sql_snippets: List[str]
    filter_conditions: List[str]


llm = ITAAPLLMClient()


def extract_kpi_node(state: KPIAgentState) -> KPIAgentState:
    prompt = f"""Extract the KPI/metric name from this question:
    
Question: {state['user_question']}

Return ONLY valid JSON (no markdown, no explanation):
{{
    "kpi_name": "the metric/column name mentioned",
    "search_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}}

For search_keywords:
- Include exact column/metric names
- Include table names if mentioned  
- Include notebook names if mentioned
- Include SQL keywords (SELECT, FROM, WHERE)
- Focus on technical terms, not generic words

Examples:
- "How is NetValueEUR calculated?" → ["NetValueEUR", "NetValue", "EUR", "currency", "calculation"]
- "What does NB_GL_Implementation do?" → ["NB_GL_Implementation", "implementation", "gold", "load"]
"""
    
    response = llm.invoke(prompt, temperature=0.3)
    content = response.content.strip()
    
    if content.startswith('```'):
        content = content.split('```')[1]
        if content.startswith('json'):
            content = content[4:]
    
    parsed = json.loads(content.strip())
    
    state['kpi_name'] = parsed['kpi_name']
    state['search_keywords'] = parsed['search_keywords']
    state['search_keywords'].extend(['SELECT', 'FROM', 'CREATE', 'gold', 'silver'])
    state['search_keywords'] = list(set(state['search_keywords']))
    
    print(f"KPI: {state['kpi_name']}")
    print(f"Keywords: {state['search_keywords']}")
    
    return state


def search_notebooks_node(state: KPIAgentState) -> KPIAgentState:
    w = get_databricks_client()
    
    search_paths = [
        '/O2C_Insights_Analytics/Transformations_to_Gold/E2E_Order_Deliver/MOM/Sales_Order_Item_Measures',
        '/O2C_Insights_Analytics/Transformations_to_Gold/E2E_Order_Deliver/MOM/Sales_Line_Measures_Alerts',
        '/O2C_Insights_Analytics/Transformations_to_Gold/E2E_Order_Deliver/MOM/Alerts',
    ]
    
    found_notebooks = []
    keywords = state['search_keywords']
    
    for base_path in search_paths:
        try:
            items = list(w.workspace.list(base_path, recursive=True))
            
            for item in items:
                if item.object_type == workspace.ObjectType.NOTEBOOK:
                    if item.path in state.get('visited_notebooks', set()):
                        continue
                    
                    try:
                        export = w.workspace.export(item.path, format=workspace.ExportFormat.SOURCE)
                        raw_content = export.content
                        
                        content = raw_content.decode('utf-8') if isinstance(raw_content, bytes) else raw_content
                        
                        if content and not content.startswith('#') and not content.startswith('--'):
                            try:
                                content = base64.b64decode(content).decode('utf-8')
                            except:
                                pass
                        
                        score = 0
                        content_lower = content.lower()
                        matches = {}
                        
                        for keyword in keywords:
                            count = content_lower.count(keyword.lower())
                            if count > 0:
                                matches[keyword] = count
                                score += count
                        
                        if score > 0:
                            found_notebooks.append({
                                'path': item.path,
                                'content': content,
                                'relevance_score': score
                            })
                            
                    except Exception as e:
                        print(f"Error reading {item.path.split('/')[-1]}: {e}")
            
            if len(found_notebooks) >= 5:
                break
                        
        except Exception as e:
            print(f"Error searching {base_path}: {e}")
    
    found_notebooks.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    state['found_notebooks'] = found_notebooks[:5]
    state['current_notebook_idx'] = 0
    
    if 'analyzed_notebooks' not in state:
        state['analyzed_notebooks'] = []
    
    print(f"Found {len(state['found_notebooks'])} relevant notebooks")
    
    return state


def analyze_notebook_node(state: KPIAgentState) -> KPIAgentState:
    idx = state['current_notebook_idx']
    
    if idx >= len(state['found_notebooks']):
        return state
    
    notebook = state['found_notebooks'][idx]
    code_sample = notebook['content'][:8000] + ("\n... (truncated)" if len(notebook['content']) > 8000 else "")
    
    prompt = f"""Analyze this code for KPI: {state['kpi_name']}

User Question: {state['user_question']}
Notebook: {notebook['path']}

```
{code_sample}
```

Provide JSON response (no markdown):
{{
    "is_relevant": true,
    "summary": "Brief explanation",
    "flowchart_mermaid": "graph TD; A[Start]-->B[Filter];",
    "filter_conditions": ["WHERE condition1"],
    "sql_snippets": ["SELECT snippet"],
    "tables_used": ["table_name"]
}}

Set is_relevant to TRUE if this notebook mentions {state['kpi_name']} or related calculations.
Extract ALL filter conditions and tables referenced.
"""
    
    response = llm.invoke(prompt, temperature=0.3, max_tokens=2000)
    content = response.content.strip()
    
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    elif '```' in content:
        content = content.split('```')[1].split('```')[0]
    
    try:
        analysis = json.loads(content.strip())
    except json.JSONDecodeError:
        analysis = {
            "is_relevant": True,
            "summary": "Analysis failed but notebook contains relevant keywords",
            "flowchart_mermaid": "",
            "filter_conditions": [],
            "sql_snippets": [],
            "tables_used": []
        }
    
    state['analyzed_notebooks'].append({
        'path': notebook['path'],
        'analysis': analysis,
        'content': notebook['content']
    })
    
    if 'visited_notebooks' not in state:
        state['visited_notebooks'] = set()
    state['visited_notebooks'].add(notebook['path'])
    
    if 'unresolved_tables' not in state:
        state['unresolved_tables'] = []
    
    for table in analysis.get('tables_used', []):
        if table not in state['unresolved_tables']:
            state['unresolved_tables'].append(table)
    
    state['current_notebook_idx'] += 1
    
    return state


def should_continue(state: KPIAgentState) -> str:
    if state['current_notebook_idx'] < len(state['found_notebooks']) and state['current_notebook_idx'] < 5:
        return "analyze_more"
    
    has_deps = len(state.get('unresolved_tables', [])) > 0
    under_limit = state.get('depth', 0) < state.get('max_depth', 2)
    
    if has_deps and under_limit:
        return "resolve_deps"
    
    return "synthesize"


def resolve_dependencies_node(state: KPIAgentState) -> KPIAgentState:
    tables_to_find = state['unresolved_tables'][:3]
    state['unresolved_tables'] = state['unresolved_tables'][3:]
    state['search_keywords'].extend(tables_to_find)
    state['depth'] = state.get('depth', 0) + 1
    state['current_notebook_idx'] = 0
    state['found_notebooks'] = []
    
    return state


def synthesize_node(state: KPIAgentState) -> KPIAgentState:
    all_analyses = state['analyzed_notebooks']
    relevant = all_analyses
    
    if not relevant:
        state['combined_analysis'] = f"No notebooks found for KPI: {state['kpi_name']}"
        return state
    
    analyses_text = ""
    for nb in relevant:
        analyses_text += f"\nNotebook: {nb['path']}\n"
        analyses_text += f"Analysis: {nb['analysis']}\n"
        analyses_text += "---\n"
    
    synthesis_prompt = f"""User asked: {state['user_question']}

Analyzed {len(relevant)} notebooks for KPI: {state['kpi_name']}

Analysis results:
{analyses_text}

Provide answer in this format:

## 1. Answer
Explain {state['kpi_name']} in 2-3 lines (max 11 words per line)

## 2. Flow Chart
```mermaid
graph TD
    Start[Data Source] --> Filter1[First Filter]
    Filter1 --> Filter2[Second Filter]
    Filter2 --> Aggregate[Aggregation]
    Aggregate --> Result[Final KPI]
```

## 3. SQL Code
```sql
-- Relevant SQL/PySpark code
SELECT ... FROM ... WHERE ...
```

## 4. Summary
Explain how {state['kpi_name']} is calculated.
"""
    
    response = llm.invoke(synthesis_prompt, temperature=0.5, max_tokens=3000)
    state['combined_analysis'] = response.content
    
    mermaid_match = re.search(r'```mermaid\n(.*?)```', response.content, re.DOTALL)
    state['flowchart'] = mermaid_match.group(1) if mermaid_match else ''
    
    sql_matches = re.findall(r'```sql\n(.*?)```', response.content, re.DOTALL)
    state['sql_snippets'] = sql_matches
    
    all_filters = []
    for nb in relevant:
        all_filters.extend(nb['analysis'].get('filter_conditions', []))
    state['filter_conditions'] = all_filters
    
    return state


def create_kpi_agent():
    workflow = StateGraph(KPIAgentState)
    
    workflow.add_node("extract_kpi", extract_kpi_node)
    workflow.add_node("search_notebooks", search_notebooks_node)
    workflow.add_node("analyze_notebook", analyze_notebook_node)
    workflow.add_node("resolve_dependencies", resolve_dependencies_node)
    workflow.add_node("synthesize", synthesize_node)
    
    workflow.set_entry_point("extract_kpi")
    workflow.add_edge("extract_kpi", "search_notebooks")
    workflow.add_edge("search_notebooks", "analyze_notebook")
    
    workflow.add_conditional_edges(
        "analyze_notebook",
        should_continue,
        {
            "analyze_more": "analyze_notebook",
            "resolve_deps": "resolve_dependencies",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_edge("resolve_dependencies", "search_notebooks")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


def run_kpi_agent(user_question: str):
    agent = create_kpi_agent()
    
    initial_state = {
        'user_question': user_question,
        'kpi_name': '',
        'search_keywords': [],
        'found_notebooks': [],
        'current_notebook_idx': 0,
        'analyzed_notebooks': [],
        'unresolved_tables': [],
        'visited_notebooks': set(),
        'depth': 0,
        'max_depth': 2,
    }
    
    print("\n" + "="*80)
    print("Analyzing...")
    print("="*80 + "\n")
    
    result = agent.invoke(initial_state)
    
    return result


def main():
    print("\n" + "="*80)
    print("KPI Agent - Interactive Mode")
    print("="*80)
    print("Ask questions about KPIs and metrics")
    print("Type 'exit' or 'quit' to end the session")
    print("="*80 + "\n")
    
    while True:
        try:
            question = input("\n📊 Your question: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            result = run_kpi_agent(question)
            
            print("\n" + "="*80)
            print("ANSWER")
            print("="*80)
            print(result['combined_analysis'])
            print(f"\nNotebooks analyzed: {len(result['analyzed_notebooks'])}")
            print(f"Filter conditions: {len(result.get('filter_conditions', []))}")
            print("="*80)
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()