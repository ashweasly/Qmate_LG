import streamlit as st
from notebook_agent import run_kpi_agent

st.set_page_config(page_title="Query Mate", layout="wide")

st.image("Philips_logo.png", width=250)
 
# Custom title with colored "VerifAI" and smaller subtitle
st.markdown("""
    <h1 style="margin-bottom:0;">
        <span style="color:#0B5ED7;">Query Mate-Google for your scripts </span>
    </h1>
    <h4 style="margin-top:0; font-weight:normal;">
        Python / SQL / PySpark/QVD
    </h4>
    """, unsafe_allow_html=True)
 


st.title("📊 Query Mate")
st.write("Ask questions about KPIs and metrics in your Databricks notebooks.")

question = st.text_input("Enter your KPI question:", "")

# ...existing code...
if st.button("Analyze") and question.strip():
    with st.spinner("Analyzing..."):
        try:
            result = run_kpi_agent(question)
            st.success("Analysis complete!")

            st.header("1. Answer")
            st.markdown(result.get('combined_analysis', 'No answer found.'), unsafe_allow_html=True)

            if result.get('flowchart'):
                st.header("2. Flow Chart")
                st.markdown("```mermaid\n" + result['flowchart'] + "\n```")

            if result.get('sql_snippets'):
                st.header("3. SQL Code")
                for idx, sql in enumerate(result['sql_snippets']):
                    st.code(sql, language='sql')

            st.header("4. Summary")
            st.write(f"Notebooks analyzed: {len(result.get('analyzed_notebooks', []))}")
            st.write(f"Filter conditions: {len(result.get('filter_conditions', []))}")

        except Exception as e:
            st.error(f"Error: {e}")
# ...existing code...