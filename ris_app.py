import streamlit as st
import os
import glob
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
from pathlib import Path

# Import query service functions
try:
    from query_service import keyword_search, semantic_search, hybrid_search, advanced_search, get_db_connection
    QUERY_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import query_service: {e}")
    QUERY_SERVICE_AVAILABLE = False
    
    # Define get_db_connection here if import fails
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from dotenv import load_dotenv
    
    load_dotenv()
    
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

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: #6fa8dc;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 0.5rem;
        margin-top: -1rem;
        box-shadow: 0 4px 15px rgba(111, 168, 220, 0.3);
    }
    
    .main-title {
        color: #1a1a1a;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.2rem;
        text-shadow: none;
    }
    
    .subtitle {
        color: #1a1a1a;
        font-size: 0.8rem;
        margin-bottom: 0;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# System-wide search paths
SEARCH_PATHS = [
    r"D:\Users\2394071\Desktop\research intelligent search",
    r"D:\Users\2394071\Desktop",
    r"D:\Users\2394071\Documents",
    r"C:\Users\2394071\Downloads",
]

# Agent definitions with dynamic status
def get_agent_definitions():
    """Get agent definitions with current status"""
    base_agents = {
        "Coordinator": {"icon": "🎯", "status": "Active", "description": "System management"},
        "File Watcher": {"icon": "📁", "status": "Monitoring", "description": "File system monitoring"},
        "Text Processor": {"icon": "📝", "status": "Ready", "description": "Document processing"},
        "Image Analyzer": {"icon": "🖼️", "status": "Ready", "description": "Visual content analysis"},
        "Data Processor": {"icon": "📊", "status": "Ready", "description": "Structured data handling"},
        "Embedding Generator": {"icon": "🧠", "status": "Ready", "description": "Vector embeddings"},
        "Relationship Mapper": {"icon": "🔗", "status": "Ready", "description": "Knowledge connections"},
        "Query Handler": {"icon": "🔍", "status": "Ready", "description": "Search & retrieval"}
    }
    return base_agents

def search_with_embeddings(query, search_types):
    """Enhanced search using query_service with embeddings"""
    results = {
        "documents": [],
        "images": [],
        "charts": [],
        "data": [],
        "excel": [],
        "all_files": [],
        "total_found": 0
    }
    
    if not query or not QUERY_SERVICE_AVAILABLE:
        if not QUERY_SERVICE_AVAILABLE:
            st.error("❌ Query service not available. Please check query_service.py")
        return results
    
    try:
        # Determine which search function to use based on selected types
        search_results = []
        
        if "Semantic" in search_types:
            # Use semantic search - Very high threshold to exclude screenshots
            search_results = semantic_search(query, top_k=10, min_score=0.75)
            
        elif "Keyword" in search_types:
            # Use keyword search only
            search_results = keyword_search(query, top_k=15, min_score=0.0)
            
        else:
            # Default to semantic search
            search_results = semantic_search(query, top_k=10, min_score=0.75)
        
        # Convert search results to dashboard format
        for result in search_results:
            file_info = {
                "name": result['file_name'],
                "path": result['file_path'],
                "type": result['file_type'],
                "relevance_score": result['relevance_score'],
            }
            results["all_files"].append(file_info)
        
        # Calculate total
        results["total_found"] = len(results["all_files"])
        
        # Display search statistics
        if results["total_found"] > 0:
            st.success(f"✅ Found {results['total_found']} relevant documents")
            
            # Show score distribution
            scores = [r['relevance_score'] for r in results["all_files"]]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            st.info(f"📊 Relevance Scores - Avg: {avg_score:.3f} | Max: {max_score:.3f} | Min: {min_score:.3f}")
        
    except Exception as e:
        st.error(f"❌ Error in embedding search: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    
    return results

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

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🧠 RESEARCH INTELLIGENT SEARCH PLATFORM</h1>
        <p class="subtitle">Advanced Multi-Agent Knowledge Discovery System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search box - FIXED VERSION
    st.markdown('<p style="font-size: 0.75rem; margin: 0.5rem 0 0.2rem 0; color: #000; font-weight: bold;">🧠 Multi-Modal Search Query</p>', unsafe_allow_html=True)
    
    # Use text_input with key to auto-update session state
    search_query = st.text_input(
        "Search Query",
        placeholder="Search your files...",
        key="search_input",  # This auto-updates st.session_state.search_input
        label_visibility="collapsed"
    )
    
    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True):
        # Update session state when button is clicked
        st.session_state.search_query = st.session_state.search_input
        st.session_state.trigger_search = True
    
    # Search Mode Selection
    st.markdown('<p style="font-size: 0.7rem; margin: 0.3rem 0 0.2rem 0; color: #000; font-weight: bold;">🎯 Search Mode</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔤 Keyword", use_container_width=True, type="primary" if st.session_state.search_mode == "Keyword" else "secondary"):
            st.session_state.search_mode = "Keyword"
    with col2:
        if st.button("🧠 Semantic", use_container_width=True, type="primary" if st.session_state.search_mode == "Semantic" else "secondary"):
            st.session_state.search_mode = "Semantic"
    with col3:
        if st.button("💬 LLM Q&A", use_container_width=True, type="primary" if st.session_state.search_mode == "LLM" else "secondary"):
            st.session_state.search_mode = "LLM"
    
    st.markdown(f'<p style="font-size: 0.65rem; margin: 0.2rem 0; color: #666; font-style: italic;">Selected: {st.session_state.search_mode} {"Search" if st.session_state.search_mode != "LLM" else "Mode"}</p>', unsafe_allow_html=True)
    
    # Process search if triggered
    if st.session_state.trigger_search and st.session_state.search_query:
        search_query = st.session_state.search_query
        st.session_state.trigger_search = False  # Reset trigger
        
        if search_query:
            st.markdown('<h4 style="font-size: 0.85rem; margin: 1rem 0 0.5rem 0;">📊 Search Progress</h4>', unsafe_allow_html=True)
            
            # Check if LLM Q&A mode
            if st.session_state.search_mode == "LLM":
                st.markdown(f'<p style="font-size: 0.65rem; color: #666;">⏳ Processing with GPT-4o...</p>', unsafe_allow_html=True)
                
                try:
                    from qa_service import answer_question
                    
                    # Get answer from GPT-4o
                    result = answer_question(search_query)
                    answer = result.get('answer', 'No answer generated')
                    
                    st.markdown(f'<p style="font-size: 0.65rem; color: #00cc66;">✅ GPT-4o Response Generated</p>', unsafe_allow_html=True)
                    
                    # Display answer
                    st.markdown("---")
                    st.markdown('<h3 style="font-size: 0.95rem;">💬 LLM Q&A Response</h3>', unsafe_allow_html=True)
                    st.markdown(f'<div style="background: #f0f8ff; padding: 1rem; border-radius: 10px; border-left: 4px solid #6fa8dc;"><p style="font-size: 0.85rem; color: #000; line-height: 1.6;">{answer}</p></div>', unsafe_allow_html=True)
                    
                    st.stop()
                    
                except Exception as e:
                    st.error(f"❌ Error with GPT-4o: {str(e)}")
                    st.info("💡 Make sure qa_service.py is configured correctly")
                    st.stop()
            else:
                # Perform search
                st.markdown(f'<p style="font-size: 0.65rem; color: #666;">⏳ Performing {st.session_state.search_mode} Search...</p>', unsafe_allow_html=True)
                
                search_results = search_with_embeddings(search_query, [st.session_state.search_mode])
                
                st.markdown(f'<p style="font-size: 0.65rem; color: #00cc66;">✅ Search Complete</p>', unsafe_allow_html=True)
                
                # Display results
                if search_results["total_found"] > 0:
                    st.markdown("---")
                    st.markdown('<h3 style="font-size: 0.95rem;">🔍 Search Results</h3>', unsafe_allow_html=True)
                    
                    for i, file_info in enumerate(search_results["all_files"], 1):
                        st.markdown(f'<p style="font-size: 0.75rem;"><strong>{i}. 📄 {file_info["name"]}</strong></p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size: 0.65rem;">Type: {file_info["type"]} | Score: {file_info["relevance_score"]:.3f}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size: 0.7rem;"><strong>📍 Path:</strong> {file_info["path"]}</p>', unsafe_allow_html=True)
                        st.markdown("---")
                else:
                    st.warning("No matching files found")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        🧠 Research Intelligence Search Platform | Powered by Multi-Agent Architecture & Amazon Nova-2
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
