import streamlit as st
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

# Import query service functions
try:
    from query_service import keyword_search, semantic_search, hybrid_search, advanced_search, get_db_connection
    QUERY_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import query_service: {e}")
    QUERY_SERVICE_AVAILABLE = False

# Define get_db_connection here if import fails
if not QUERY_SERVICE_AVAILABLE:
    def get_db_connection():
        """Create database connection"""
        return psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME', 'research_kb'),
            cursor_factory=RealDictCursor
        )

# Page configuration
st.set_page_config(
    page_title="Research Intelligence Search Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'search_active' not in st.session_state:
    st.session_state.search_active = False
if 'activated_agents' not in st.session_state:
    st.session_state.activated_agents = []
if 'trigger_search' not in st.session_state:
    st.session_state.trigger_search = False
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'search_mode' not in st.session_state:
    st.session_state.search_mode = "Semantic"

def search_with_embeddings(query, search_types):
    """Perform semantic or keyword search"""
    results = {"all_files": [], "total_found": 0}
    
    if not query or not QUERY_SERVICE_AVAILABLE:
        if not QUERY_SERVICE_AVAILABLE:
            st.error("❌ Query service not available. Please check query_service.py")
        return results
    
    try:
        search_results = []
        
        if "Semantic" in search_types:
            search_results = semantic_search(query, top_k=10, min_score=0.50)
        elif "Keyword" in search_types:
            search_results = keyword_search(query, top_k=15, min_score=0.0)
        else:
            search_results = semantic_search(query, top_k=10, min_score=0.50)
        
        for result in search_results:
            file_info = {
                "name": result['file_name'],
                "path": result['file_path'],
                "type": result['file_type'],
                "relevance_score": result['relevance_score'],
            }
            results["all_files"].append(file_info)
        
        results["total_found"] = len(results["all_files"])
        
    except Exception as e:
        st.error(f"❌ Error in embedding search: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    
    return results

def main():
    st.markdown("""
    <div style="background:#6fa8dc; padding:10px; border-radius:8px; text-align:center;">
        <h1 style="margin:0; color:#1a1a1a;">🧠 RESEARCH INTELLIGENT SEARCH PLATFORM</h1>
        <p style="margin:0; color:#1a1a1a; font-size:0.9rem;">Advanced Multi-Agent Knowledge Discovery System</p>
    </div>
    """, unsafe_allow_html=True)
    
    search_query = st.text_input(
        "Search Query",
        placeholder="Enter your search...",
        key="search_input",
        label_visibility="collapsed"
    )
    
    if st.button("🔍 Search", type="primary", use_container_width=True):
        st.session_state.search_query = st.session_state.search_input
        st.session_state.trigger_search = True
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔤 Keyword", use_container_width=True, type="primary" if st.session_state.search_mode=="Keyword" else "secondary"):
            st.session_state.search_mode="Keyword"
    with col2:
        if st.button("🧠 Semantic", use_container_width=True, type="primary" if st.session_state.search_mode=="Semantic" else "secondary"):
            st.session_state.search_mode="Semantic"
    with col3:
        if st.button("💬 LLM Q&A", use_container_width=True, type="primary" if st.session_state.search_mode=="LLM" else "secondary"):
            st.session_state.search_mode="LLM"
    
    st.markdown(f'<p style="font-size:0.65rem; color:#666;">Selected: {st.session_state.search_mode} {"Search" if st.session_state.search_mode!="LLM" else "Mode"}</p>', unsafe_allow_html=True)
    
    if st.session_state.trigger_search and st.session_state.search_query:
        st.session_state.trigger_search=False
        query_text = st.session_state.search_query
        
        st.markdown(f'<p style="font-size:0.65rem; color:#666;">⏳ Performing {st.session_state.search_mode} Search...</p>', unsafe_allow_html=True)
        
        if st.session_state.search_mode=="LLM":
            try:
                from qa_service import answer_question
                result = answer_question(query_text)
                answer = result.get('answer', 'No answer generated')
                
                st.markdown(f'<p style="font-size:0.65rem; color:#00cc66;">✅ GPT-4o Response Generated</p>', unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(f'<h3>💬 LLM Q&A Response</h3>', unsafe_allow_html=True)
                st.markdown(f'<div style="background:#f0f8ff; padding:1rem; border-radius:10px; border-left:4px solid #6fa8dc;"><p>{answer}</p></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error with GPT-4o: {str(e)}")
        else:
            search_results = search_with_embeddings(query_text, [st.session_state.search_mode])
            st.markdown(f'<p style="font-size:0.65rem; color:#00cc66;">✅ Search Complete</p>', unsafe_allow_html=True)
            
            if search_results["total_found"]>0:
                st.markdown("---")
                st.markdown('<h3>🔍 Search Results</h3>', unsafe_allow_html=True)
                for i, file_info in enumerate(search_results["all_files"], 1):
                    st.markdown(f'<p><strong>{i}. {file_info["name"]}</strong></p>', unsafe_allow_html=True)
                    st.markdown(f'<p>Type: {file_info["type"]} | Score: {file_info["relevance_score"]:.3f}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p>📍 Path: {file_info["path"]}</p>', unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.warning("No matching files found")
    
    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#888; padding:1rem;">🧠 Research Intelligence Search Platform | Powered by Multi-Agent Architecture & Amazon Nova-2</div>', unsafe_allow_html=True)

if __name__=="__main__":
    main()
