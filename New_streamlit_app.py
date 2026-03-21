# streamlit_app.py (simplified)
import streamlit as st
import tempfile
import os
from pathlib import Path
from langgraph_functions import analyze_multiple_files, read_text_from_file, llm

st.set_page_config(layout="wide")
st.image("Philips_logo.png", width=250)

st.markdown("""
    <h1><span style="color:#0B5ED7;">Query Mate</span></h1>
    <h4>Python / SQL / PySpark Code Analysis</h4>
    """, unsafe_allow_html=True)

# Input mode
mode = st.radio("Input mode", ["Upload files", "Enter filenames"], index=0)
file_list = []

if mode == "Upload files":
    uploaded = st.file_uploader("Upload files", accept_multiple_files=True, type=['py', 'sql', 'txt'])
    if uploaded:
        tmp_dir = tempfile.mkdtemp()
        for up in uploaded:
            tmp_path = os.path.join(tmp_dir, up.name)
            with open(tmp_path, "wb") as f:
                f.write(up.getbuffer())
            file_list.append(tmp_path)
else:
    filenames = st.text_area("Enter filenames (one per line)", height=120)
    file_list = [line.strip() for line in filenames.splitlines() if line.strip()]

user_question = st.text_input("Optional question:", "")

if st.button("Analyze"):
    if not file_list:
        st.warning("No files provided.")
    else:
        with st.spinner("Analyzing..."):
            all_results, dep_graph = analyze_multiple_files(file_list, user_question or None)
        
        st.success("Analysis complete!")
        
        for file_path, result in all_results.items():
            st.markdown("---")
            st.markdown(f"**File:** {Path(file_path).name}")
            st.code(result['analysis'])
            
            # Follow-up Q&A
            st.markdown("### Follow-up Question")
            followup = st.text_input(f"Ask about {Path(file_path).name}:", key=file_path)
            if st.button("Ask", key=f"btn_{file_path}"):
                if followup:
                    reply = llm.invoke(f"Prior analysis:\n{result['analysis']}\n\nQuestion: {followup}")
                    st.code(reply.content if hasattr(reply, 'content') else str(reply))