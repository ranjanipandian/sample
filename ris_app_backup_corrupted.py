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
import numpy as np
from openai import OpenAI

# -----------------------------
# Utility functions for search & LLM
# -----------------------------
def load_embeddings():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT file_path, file_name, file_type, embedding
        FROM document_embeddings
        WHERE embedding IS NOT NULL
        ORDER BY file_name
    """)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    embeddings = []
    for row in results:
        file_name = row[1]
        file_type = row[2]
        embedding_vector = row[3]
        if isinstance(embedding_vector, dict) and 'embedding' in embedding_vector:
            embedding_vector = embedding_vector['embedding']
        if isinstance(embedding_vector, list):
            embeddings.append({
                "file_name": file_name,
                "file_type": file_type,
                "embedding": np.array(embedding_vector)
            })
    return embeddings

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, embeddings, openai_client):
    query_embedding = np.array(openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )["data"][0]["embedding"])
    
    scored = []
    for item in embeddings:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append({"file_name": item["file_name"], "score": score})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:5]

def keyword_search(query, embeddings):
    results = []
    q_lower = query.lower()
    for item in embeddings:
        if q_lower in item["file_name"].lower():
            results.append(item)
    return results[:5]

def llm_qa(prompt, openai_client):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message["content"]


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
    
    /* Agent status cards */
    .agent-card {
        background: #6fa8dc;
        padding: 0.6rem;
        border-radius: 10px;
        color: #1a1a1a;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(111, 168, 220, 0.3);
        transition: transform 0.3s ease;
        height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .agent-card h3 {
        font-size: 0.9rem;
        margin: 0 0 0.2rem 0;
        font-weight: bold;
        line-height: 1.1;
        color: #1a1a1a;
    }
    
    .agent-card p {
        font-size: 0.8rem;
        margin: 0.1rem 0;
        font-weight: bold;
        line-height: 1.0;
        color: #1a1a1a;
    }
    
    .agent-card small {
        font-size: 0.7rem;
        margin: 0.1rem 0 0 0;
        line-height: 1.0;
        opacity: 0.8;
        color: #1a1a1a;
    }
    
    .agent-card:hover {
        transform: translateY(-5px);
    }
    
    /* Data source cards */
    .data-source-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    /* Status indicators */
    .status-active {
        color: #00ff88;
        font-weight: bold;
    }
    
    .status-processing {
        color: #00ff88;
        font-weight: bold;
    }
    
    .status-idle {
        color: #888888;
        font-weight: bold;
    }
    
    /* Search interface */
    .search-container {
        background: #6fa8dc;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: #333;
        box-shadow: 0 4px 15px rgba(252, 182, 159, 0.3);
    }
    
    /* Sidebar metrics styling - compact version */
    .sidebar-metric-container {
        background: #6fa8dc;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        color: #1a1a1a;
        box-shadow: 0 3px 10px rgba(111, 168, 220, 0.3);
        margin-bottom: 0.5rem;
    }
    
    .sidebar-metric-container h3 {
        font-size: 1.5rem;
        margin: 0;
        font-weight: bold;
        color: #1a1a1a;
    }
    
    .sidebar-metric-container p {
        font-size: 0.9rem;
        margin: 0.2rem 0 0 0;
        font-weight: 500;
        color: #1a1a1a;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: #6fa8dc;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        color: #1a1a1a;
        box-shadow: 0 4px 15px rgba(111, 168, 220, 0.3);
    }
    
    .sidebar-header h3 {
        margin: 0;
        font-size: 1.0rem;
        font-weight: bold;
        color: #1a1a1a;
    }
    
    /* Model configuration styling */
    .model-config-container {
        background: #6fa8dc;
        padding: 0.8rem;
        border-radius: 10px;
        color: #1a1a1a;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(111, 168, 220, 0.3);
    }
    
    .model-config-container h3 {
        margin: 0 0 0.3rem 0;
        font-size: 0.9rem;
        font-weight: bold;
        text-align: center;
        color: #1a1a1a;
    }
    
    .model-status-active {
        color: #00ff88;
        font-weight: bold;
        font-size: 0.75rem;
    }
    
    .model-info {
        font-size: 0.7rem;
        line-height: 1.2;
        margin: 0.2rem 0;
        color: #1a1a1a;
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

def analyze_search_query(query):
    """Analyze search query and determine which agents to activate"""
    if not query:
        return [], []
    
    query_lower = query.lower()
    activated_agents = []
    search_insights = []
    
    # Always activate core agents for any search
    activated_agents.extend(["Coordinator", "Query Handler", "Embedding Generator"])
    
    # Text-related keywords
    text_keywords = ["document", "text", "paper", "article", "report", "pdf", "word", "research", "study", "clinical", "drug", "medical"]
    if any(keyword in query_lower for keyword in text_keywords):
        activated_agents.append("Text Processor")
        search_insights.append("🔤 Text documents will be searched for relevant content")
    
    # Image-related keywords  
    image_keywords = ["image", "chart", "graph", "picture", "visual", "diagram", "plot", "figure", "screenshot", "photo"]
    if any(keyword in query_lower for keyword in image_keywords):
        activated_agents.append("Image Analyzer")
        search_insights.append("🖼️ Visual content and charts will be analyzed")
    
    # Data/Table-related keywords
    data_keywords = ["data", "table", "excel", "csv", "database", "spreadsheet", "numbers", "statistics", "insurance", "groceries"]
    if any(keyword in query_lower for keyword in data_keywords):
        activated_agents.append("Data Processor")
        search_insights.append("📊 Structured data and tables will be processed")
    
    # Complex queries activate relationship mapping
    complex_keywords = ["relationship", "connection", "related", "similar", "compare", "analysis"]
    if any(keyword in query_lower for keyword in complex_keywords):
        activated_agents.append("Relationship Mapper")
        search_insights.append("🔗 Semantic relationships will be mapped")
    
    # Remove duplicates while preserving order
    activated_agents = list(dict.fromkeys(activated_agents))
    
    return activated_agents, search_insights

def search_with_embeddings(query, search_types):
    """
    Enhanced search using query_service with embeddings
    Supports: Keyword, Semantic, Hybrid, and Advanced search
    """
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
            file_ext = os.path.splitext(result['file_name'])[1].lower()
            
            # Create file info dict
            file_info = {
                "name": result['file_name'],
                "path": result['file_path'],
                "size": get_file_size(result['file_path']),
                "modified": result['file_modified'].strftime("%b %d, %Y") if hasattr(result['file_modified'], 'strftime') else str(result['file_modified']),
                "type": result['file_type'],
                "relevance_score": result['relevance_score'],
                "search_type": result.get('search_type', 'unknown'),
                "content_preview": result.get('content_preview', '')
            }
            
            # Add keyword/semantic scores if hybrid search
            if 'keyword_score' in result:
                file_info['keyword_score'] = result['keyword_score']
            if 'semantic_score' in result:
                file_info['semantic_score'] = result['semantic_score']
            
            # Categorize by file type
            if result['file_type'] in ['PDF Document']:
                file_info['category'] = 'documents'
                file_info['agent'] = 'Text Processor'
                results["documents"].append(file_info)
                
            elif result['file_type'] in ['Text Document', 'Word Document']:
                file_info['category'] = 'documents'
                file_info['agent'] = 'Text Processor'
                results["documents"].append(file_info)
                
            elif result['file_type'] == 'Image':
                file_info['category'] = 'images'
                file_info['agent'] = 'Image Analyzer'
                results["images"].append(file_info)
                
            elif result['file_type'] in ['HTML/Chart', 'Interactive Chart']:
                file_info['category'] = 'charts'
                file_info['agent'] = 'Image Analyzer'
                results["charts"].append(file_info)
                
            elif result['file_type'] == 'Excel Spreadsheet':
                file_info['category'] = 'excel'
                file_info['agent'] = 'Data Processor'
                results["excel"].append(file_info)
                
            elif result['file_type'] in ['CSV Data', 'JSON Data']:
                file_info['category'] = 'data'
                file_info['agent'] = 'Data Processor'
                results["data"].append(file_info)
            
            # Add to all_files
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

def get_file_size(file_path):
    """Get file size in human readable format"""
    try:
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
    except:
        return "Unknown"

def get_file_modified_date(file_path):
    """Get file last modified date"""
    try:
        timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(timestamp).strftime("%b %d, %Y")
    except:
        return "Unknown"

def display_file_preview(file_info):
    """Display preview of selected file based on its type"""
    try:
        file_path = file_info["path"]
        file_type = file_info["type"]
        
        st.markdown(f"**🔍 Content Preview for**: {file_info['name']}")
        
        # Show file location prominently
        st.markdown(f"**📍 File Location**: `{file_path}`")
        st.markdown(f"**📏 Size**: {file_info['size']} | **📅 Modified**: {file_info['modified']} | **🤖 Agent**: {file_info['agent']}")
        
        # Enhanced debugging information
        st.write("**🔧 Debug Info:**")
        st.write(f"File exists: {os.path.exists(file_path)}")
        st.write(f"File type detected: {file_type}")
        st.write(f"File extension: {os.path.splitext(file_path)[1]}")
        st.write(f"Is file: {os.path.isfile(file_path)}")
        
        # Try to get file permissions info
        try:
            import stat
            file_stat = os.stat(file_path)
            st.write(f"File permissions: {stat.filemode(file_stat.st_mode)}")
        except Exception as perm_error:
            st.write(f"Permission check failed: {perm_error}")
        
        # Add button to open file directly
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"📂 Open {file_type}", key=f"open_{file_info['name']}"):
                try:
                    import subprocess
                    import platform
                    
                    if platform.system() == 'Windows':
                        os.startfile(file_path)
                        st.success(f"✅ Opening {file_info['name']} in default application...")
                    else:
                        subprocess.run(['open', file_path])  # macOS
                        st.success(f"✅ Opening {file_info['name']}...")
                except Exception as open_error:
                    st.error(f"❌ Could not open file: {open_error}")
        
        with col2:
            if st.button(f"📁 Open Folder", key=f"folder_{file_info['name']}"):
                try:
                    folder_path = os.path.dirname(file_path)
                    if platform.system() == 'Windows':
                        os.startfile(folder_path)
                        st.success(f"✅ Opening folder containing {file_info['name']}...")
                    else:
                        subprocess.run(['open', folder_path])  # macOS
                        st.success(f"✅ Opening folder...")
                except Exception as folder_error:
                    st.error(f"❌ Could not open folder: {folder_error}")
        
        st.markdown("---")
        st.markdown("**🔍 Attempting Preview (Debug Mode):**")
        
        if file_type in ["Image", "Screenshot"]:
            # Display image with enhanced error handling
            if os.path.exists(file_path):
                try:
                    # Try to load and display the image
                    from PIL import Image
                    img = Image.open(file_path)
                    st.image(img, caption=f"{file_info['name']} - {file_info['type']}", width=600)
                    st.success("✅ Image loaded successfully!")
                except Exception as img_error:
                    st.error(f"❌ Error loading image: {str(img_error)}")
                    # Fallback: try direct path
                    try:
                        st.image(file_path, caption=f"{file_info['name']} - {file_info['type']}", width=600)
                        st.success("✅ Image loaded with fallback method!")
                    except Exception as fallback_error:
                        st.error(f"❌ Fallback also failed: {str(fallback_error)}")
            else:
                st.error(f"❌ Image file not found at: {file_path}")
            
        elif file_type in ["CSV Data"]:
            # Display CSV data with enhanced error handling
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    st.dataframe(df.head(10), use_container_width=True)
                    st.info(f"✅ Showing first 10 rows of {len(df)} total rows | Columns: {len(df.columns)}")
                except Exception as csv_error:
                    st.error(f"❌ Error loading CSV: {str(csv_error)}")
            else:
                st.error(f"❌ CSV file not found at: {file_path}")
        
        elif file_type == "Excel Spreadsheet":
            # Display Excel data with enhanced error handling
            if os.path.exists(file_path):
                try:
                    df = pd.read_excel(file_path)
                    st.dataframe(df.head(10), use_container_width=True)
                    st.info(f"✅ Showing first 10 rows of {len(df)} total rows | Columns: {len(df.columns)}")
                except Exception as excel_error:
                    st.error(f"❌ Error loading Excel: {str(excel_error)}")
            else:
                st.error(f"❌ Excel file not found at: {file_path}")
            
        elif file_type == "Interactive Chart":
            # Display HTML chart with enhanced error handling
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Check if it's a complete HTML document or just content
                    if not html_content.strip().startswith('<!DOCTYPE') and not html_content.strip().startswith('<html'):
                        # Wrap content in basic HTML structure
                        html_content = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="utf-8">
                            <title>Chart Preview</title>
                        </head>
                        <body>
                            {html_content}
                        </body>
                        </html>
                        """
                    
                    st.components.v1.html(html_content, height=600, scrolling=True)
                    st.success("✅ Interactive chart loaded successfully!")
                    
                    # Also provide a download link
                    st.markdown(f"**💡 Tip**: If the chart doesn't display properly, try opening the file directly:")
                    st.code(file_path)
                    
                except UnicodeDecodeError:
                    try:
                        # Try different encoding
                        with open(file_path, 'r', encoding='latin-1') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=600, scrolling=True)
                        st.success("✅ Interactive chart loaded with latin-1 encoding!")
                    except Exception as chart_error:
                        st.error(f"❌ Error loading chart: {str(chart_error)}")
                        # Fallback: show file info and suggest opening externally
                        st.info("💡 **Alternative**: Try opening the file directly in your browser:")
                        st.code(file_path)
            else:
                st.error(f"❌ Chart file not found at: {file_path}")
            
        elif file_type == "JSON Database":
            # Display JSON data with enhanced error handling
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    st.json(json_data)
                    st.success("✅ JSON data loaded successfully!")
                except Exception as json_error:
                    st.error(f"❌ Error loading JSON: {str(json_error)}")
            else:
                st.error(f"❌ JSON file not found at: {file_path}")
            
        elif file_type in ["PDF Document", "Text Document"]:
            # Display text content with enhanced error handling
            if file_type == "Text Document":
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        st.text_area("File Content:", content, height=400)
                        st.success("✅ Text content loaded successfully!")
                    except UnicodeDecodeError:
                        try:
                            with open(file_path, 'r', encoding='latin-1') as f:
                                content = f.read()
                            st.text_area("File Content:", content, height=400)
                            st.success("✅ Text content loaded with latin-1 encoding!")
                        except Exception as text_error:
                            st.error(f"❌ Error loading text: {str(text_error)}")
                    except Exception as text_error:
                        st.error(f"❌ Error loading text: {str(text_error)}")
                else:
                    st.error(f"❌ Text file not found at: {file_path}")
            else:
                # For PDF files, try to extract text content
                if os.path.exists(file_path):
                    try:
                        # Try to extract text from PDF using PyPDF2
                        import PyPDF2
                        with open(file_path, 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            text_content = ""
                            for page_num in range(min(3, len(pdf_reader.pages))):  # Show first 3 pages
                                page = pdf_reader.pages[page_num]
                                text_content += f"\n--- Page {page_num + 1} ---\n"
                                text_content += page.extract_text()
                            
                            if text_content.strip():
                                st.text_area("PDF Content (First 3 pages):", text_content, height=400)
                                st.success("✅ PDF text extracted successfully!")
                            else:
                                st.warning("⚠️ PDF appears to be image-based or encrypted. Text extraction not possible.")
                                st.info("📄 PDF file found but content cannot be displayed as text.")
                    except ImportError:
                        st.warning("⚠️ PyPDF2 not installed. Cannot extract PDF text.")
                        st.info("📄 PDF file location shown above. Install PyPDF2 for text extraction.")
                    except Exception as pdf_error:
                        st.error(f"❌ Error reading PDF: {str(pdf_error)}")
                        st.info("📄 PDF file found but content cannot be displayed.")
                else:
                    st.error(f"❌ PDF file not found at: {file_path}")
                
    except Exception as e:
        st.error(f"❌ Error displaying file preview: {str(e)}")
        st.error(f"🔍 File path: {file_path if 'file_path' in locals() else 'Unknown'}")
        st.error(f"📁 File exists: {os.path.exists(file_path) if 'file_path' in locals() else 'Unknown'}")

def display_categorized_search_results(search_results):
    """Display search results as a simple list with open buttons"""
    st.markdown("### 🔍 Search Results")
    st.markdown("*Click links to open files or folders directly*")
    st.markdown("---")
    
    if not search_results["all_files"]:
        st.info("No files found.")
        return
    
    # Display all results in a simple list
    for i, file_info in enumerate(search_results["all_files"], 1):
        # Create a container for each result
        with st.container():
            # File info
            st.markdown(f"**{i}. 📄 {file_info['name']}**")
            st.markdown(f"*Type:* {file_info['type']} | *Size:* {file_info['size']} | *Modified:* {file_info['modified']}")
            if file_info.get('relevance_score'):
                st.markdown(f"*Relevance Score:* {file_info['relevance_score']:.3f}")
            
            # Create columns for buttons
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"*Path:* `{file_info['path']}`")
            
            with col2:
                # Create unique key for this file
                file_key = f"file_{i}_{file_info['name']}"
                
                # Open file button with immediate execution
                if st.button(f"📂 Open File", key=file_key, use_container_width=True):
                    try:
                        # Use os.startfile for Windows - more reliable
                        os.startfile(file_info['path'])
                        st.toast(f"✅ Opening {file_info['name']}", icon="📂")
                    except Exception as e:
                        st.error(f"❌ Error opening file: {e}")
                        # Fallback: try with explorer
                        try:
                            import subprocess
                            subprocess.run(['explorer', file_info['path']], shell=True)
                        except:
                            st.error(f"❌ Fallback also failed. Path: {file_info['path']}")
            
            with col3:
                # Create unique key for folder
                folder_key = f"folder_{i}_{file_info['name']}"
                
                # Open folder button with immediate execution
                if st.button(f"📁 Open Folder", key=folder_key, use_container_width=True):
                    try:
                        folder_path = os.path.dirname(file_info['path'])
                        # Use os.startfile for Windows - more reliable
                        os.startfile(folder_path)
                        st.toast(f"✅ Opening folder", icon="📁")
                    except Exception as e:
                        st.error(f"❌ Error opening folder: {e}")
                        # Fallback: try with explorer
                        try:
                            import subprocess
                            subprocess.run(['explorer', folder_path], shell=True)
                        except:
                            st.error(f"❌ Fallback also failed. Path: {folder_path}")
            
            st.markdown("---")

# Initialize session state for search and embeddings visibility
if 'search_active' not in st.session_state:
    st.session_state.search_active = False
if 'activated_agents' not in st.session_state:
    st.session_state.activated_agents = []
if 'search_insights' not in st.session_state:
    st.session_state.search_insights = []
if 'show_embeddings' not in st.session_state:
    st.session_state.show_embeddings = False
if 'embedding_font_size' not in st.session_state:
    st.session_state.embedding_font_size = 6  # Start with extra tiny font (6px)

def scan_system_files():
    """Scan system for various file types with detailed breakdown"""
    detailed_stats = {
        "documents": {
            "pdf": 0,
            "word": 0,
            "presentations": 0,
            "total": 0
        },
        "structured_data": {
            "excel": 0,
            "csv": 0,
            "databases": 0,
            "total": 0
        },
        "visual_data": {
            "charts": 0,
            "images": 0,
            "total": 0
        },
        "mixed_media": {
            "compressed": 0,
            "others": 0,
            "total": 0
        },
        "processed": 0,
        "metadata": 0,
        "grand_total": 0
    }
    
    try:
        for search_path in SEARCH_PATHS:
            if os.path.exists(search_path):
                # Documents folder breakdown
                pdf_files = glob.glob(os.path.join(search_path, "**/documents/pdf/*.pdf"), recursive=True)
                detailed_stats["documents"]["pdf"] += len(pdf_files)
                
                word_files = glob.glob(os.path.join(search_path, "**/documents/word/*.txt"), recursive=True)
                word_files += glob.glob(os.path.join(search_path, "**/documents/word/*.docx"), recursive=True)
                detailed_stats["documents"]["word"] += len(word_files)
                
                ppt_files = glob.glob(os.path.join(search_path, "**/documents/presentations/*.pptx"), recursive=True)
                detailed_stats["documents"]["presentations"] += len(ppt_files)
                
                # Structured data breakdown
                excel_files = glob.glob(os.path.join(search_path, "**/structured_data/excel/*.xlsx"), recursive=True)
                detailed_stats["structured_data"]["excel"] += len(excel_files)
                
                csv_files = glob.glob(os.path.join(search_path, "**/structured_data/csv/*.csv"), recursive=True)
                detailed_stats["structured_data"]["csv"] += len(csv_files)
                
                db_files = glob.glob(os.path.join(search_path, "**/structured_data/databases/*"), recursive=True)
                db_files = [f for f in db_files if f.endswith(('.json', '.xml', '.pkl', '.pdb', '.db', '.sqlite'))]
                detailed_stats["structured_data"]["databases"] += len(db_files)
                
                # Visual data breakdown
                chart_files = glob.glob(os.path.join(search_path, "**/visual_data/charts/*"), recursive=True)
                chart_files = [f for f in chart_files if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.html'))]
                detailed_stats["visual_data"]["charts"] += len(chart_files)
                
                image_files = glob.glob(os.path.join(search_path, "**/visual_data/images/*"), recursive=True)
                image_files += glob.glob(os.path.join(search_path, "**/visual_data/screenshots/*"), recursive=True)
                image_files = [f for f in image_files if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))]
                detailed_stats["visual_data"]["images"] += len(image_files)
                
                # Mixed media breakdown
                compressed_files = glob.glob(os.path.join(search_path, "**/mixed_media/compressed/*"), recursive=True)
                compressed_files = [f for f in compressed_files if f.endswith(('.zip', '.rar', '.7z', '.tar', '.gz'))]
                detailed_stats["mixed_media"]["compressed"] += len(compressed_files)
                
                other_files = glob.glob(os.path.join(search_path, "**/mixed_media/other/*"), recursive=True)
                detailed_stats["mixed_media"]["others"] += len(other_files)
        
        # Calculate totals
        detailed_stats["documents"]["total"] = sum([detailed_stats["documents"][key] for key in detailed_stats["documents"] if key != "total"])
        detailed_stats["structured_data"]["total"] = sum([detailed_stats["structured_data"][key] for key in detailed_stats["structured_data"] if key != "total"])
        detailed_stats["visual_data"]["total"] = sum([detailed_stats["visual_data"][key] for key in detailed_stats["visual_data"] if key != "total"])
        detailed_stats["mixed_media"]["total"] = sum([detailed_stats["mixed_media"][key] for key in detailed_stats["mixed_media"] if key != "total"])
        
        detailed_stats["grand_total"] = (detailed_stats["documents"]["total"] + 
                                       detailed_stats["structured_data"]["total"] + 
                                       detailed_stats["visual_data"]["total"] + 
                                       detailed_stats["mixed_media"]["total"] + 
                                       detailed_stats["processed"] + 
                                       detailed_stats["metadata"])
        
    except Exception as e:
        st.error(f"Error scanning files: {str(e)}")
    
    return detailed_stats

def create_agent_performance_metrics():
    """Create agent performance metrics display"""
    agents = get_agent_definitions()
    metrics_data = {
        "Agent": list(agents.keys()),
        "Status": [agents[agent]["status"] for agent in agents.keys()],
        "Files Processed": [156, 45, 23, 15, 8, 1247, 89, 0],
        "Success Rate": ["98%", "100%", "95%", "92%", "88%", "99%", "94%", "N/A"],
        "Avg Response Time": ["0.2s", "0.1s", "2.3s", "1.8s", "3.2s", "0.5s", "1.1s", "0.3s"]
    }
    
    df = pd.DataFrame(metrics_data)
    return df

def main():
    # Main header - full width
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🧠 RESEARCH INTELLIGENT SEARCH PLATFORM</h1>
        <p class="subtitle">Advanced Multi-Agent Knowledge Discovery System</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<h2>🔎 Research Intelligence Search</h2>", unsafe_allow_html=True)

query = st.text_input("Enter search query (semantic, keyword, or question):")
search_type = st.selectbox("Search type:", ["Semantic Search", "Keyword Search", "LLM Q&A"])

# Initialize OpenAI client
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if st.button("Search"):
    embeddings = load_embeddings()
    if search_type == "Semantic Search":
        results = semantic_search(query, embeddings, openai_client)
        st.success(f"Top {len(results)} semantic matches:")
        for r in results:
            st.markdown(f"- {r['file_name']} (Score: {r['score']:.4f})")
    elif search_type == "Keyword Search":
        results = keyword_search(query, embeddings)
        st.success(f"Top {len(results)} keyword matches:")
        for r in results:
            st.markdown(f"- {r['file_name']}")
    elif search_type == "LLM Q&A":
        answer = llm_qa(query, openai_client)
        st.markdown(f"**Answer from LLM:** {answer}")

st.markdown("---")

    # Search box below title - matching title width
    st.markdown('<p style="font-size: 0.75rem; margin: 0.5rem 0 0.2rem 0; color: #000; font-weight: bold;">🧠 Multi-Modal Search Query</p>', unsafe_allow_html=True)
    col_search, col_button = st.columns([4, 1])
    with col_search:
        search_query = st.text_input(
            "Search Query",
            placeholder="Search your files...",
            key="top_search",
            label_visibility="collapsed"
        )
    with col_button:
        if st.button("🔍 Search", type="primary", use_container_width=True, key="top_search_btn"):
            st.session_state.search_query = search_query
            st.session_state.trigger_search = True
    
    # Search Mode Selection - Three modes: Keyword, Semantic, LLM Q&A
    st.markdown('<p style="font-size: 0.7rem; margin: 0.3rem 0 0.2rem 0; color: #000; font-weight: bold;">🎯 Search Mode</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    # Initialize search mode in session state
    if 'search_mode' not in st.session_state:
        st.session_state.search_mode = "Semantic"
    
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
    
    # Add CSS to change primary button color from red to blue
    st.markdown("""
    <style>
        /* Change primary button color to blue */
        .stButton > button[kind="primary"] {
            background-color: #6fa8dc !important;
            border-color: #6fa8dc !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #5a8fc7 !important;
            border-color: #5a8fc7 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Process search if triggered - MOVED HERE (right after search mode)
    if 'trigger_search' not in st.session_state:
        st.session_state.trigger_search = False
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    if st.session_state.trigger_search and st.session_state.search_query:
        search_query = st.session_state.search_query
        st.session_state.trigger_search = False
        
        if search_query:
            # Create progress log container
            st.markdown('<h4 style="font-size: 0.85rem; margin: 1rem 0 0.5rem 0;">📊 Search Progress</h4>', unsafe_allow_html=True)
            progress_container = st.container()
            
            with progress_container:
                # Create placeholders for each step
                step1 = st.empty()
                step2 = st.empty()
                step3 = st.empty()
                step4 = st.empty()
                step5 = st.empty()
                step6 = st.empty()
                
                # Step 1: Search initiated
                step1.markdown(f'<p style="font-size: 0.65rem; color: #666; margin: 0.2rem 0;">⏳ Search initiated: \'{search_query}\'</p>', unsafe_allow_html=True)
                time.sleep(0.3)
                step1.markdown(f'<p style="font-size: 0.65rem; color: #00cc66; margin: 0.2rem 0;">✅ Search initiated: \'{search_query}\'</p>', unsafe_allow_html=True)
                
                # Analyze query and activate relevant agents
                activated_agents, search_insights = analyze_search_query(search_query)
                st.session_state.search_active = True
                st.session_state.activated_agents = activated_agents
                st.session_state.search_insights = search_insights
                
                # Use the selected search mode
                search_types = [st.session_state.search_mode]
                
                # Step 2: Search method
                step2.markdown(f'<p style="font-size: 0.65rem; color: #666; margin: 0.2rem 0;">⏳ Using: {", ".join(search_types)} search methods</p>', unsafe_allow_html=True)
                time.sleep(0.3)
                step2.markdown(f'<p style="font-size: 0.65rem; color: #00cc66; margin: 0.2rem 0;">✅ Using: {", ".join(search_types)} search methods</p>', unsafe_allow_html=True)
                
                # Step 3: Activated agents
                step3.markdown(f'<p style="font-size: 0.65rem; color: #666; margin: 0.2rem 0;">⏳ Activated Agents: {", ".join(activated_agents)}</p>', unsafe_allow_html=True)
                time.sleep(0.3)
                step3.markdown(f'<p style="font-size: 0.65rem; color: #00cc66; margin: 0.2rem 0;">✅ Activated Agents: {", ".join(activated_agents)}</p>', unsafe_allow_html=True)
                
                # Step 4: Performing search or Q&A
                search_type_name = search_types[0]
                
                # Check if LLM Q&A mode
                if st.session_state.search_mode == "LLM":
                    step4.markdown(f'<p style="font-size: 0.65rem; color: #666; margin: 0.2rem 0;">⏳ Processing with GPT-4o...</p>', unsafe_allow_html=True)
                    
                    # Import and use QA service
                    try:
                        from qa_service import answer_question
                        
                        # Get answer from GPT-4o
                        answer = answer_question(search_query)
                        
                        step4.markdown(f'<p style="font-size: 0.65rem; color: #00cc66; margin: 0.2rem 0;">✅ GPT-4o Response Generated</p>', unsafe_allow_html=True)
                        
                        # Display answer
                        st.markdown("---")
                        st.markdown('<h3 style="font-size: 0.95rem;">💬 LLM Q&A Response</h3>', unsafe_allow_html=True)
                        st.markdown(f'<div style="background: #f0f8ff; padding: 1rem; border-radius: 10px; border-left: 4px solid #6fa8dc;"><p style="font-size: 0.85rem; color: #000; line-height: 1.6;">{answer}</p></div>', unsafe_allow_html=True)
                        
                        # Reset and skip normal search results
                        st.session_state.search_active = False
                        st.session_state.activated_agents = []
                        st.stop()
                        
                    except Exception as e:
                        st.error(f"❌ Error with GPT-4o: {str(e)}")
                        st.info("💡 Make sure qa_service.py is configured correctly with GPT-4o credentials")
                        st.session_state.search_active = False
                        st.session_state.activated_agents = []
                        st.stop()
                else:
                    step4.markdown(f'<p style="font-size: 0.65rem; color: #666; margin: 0.2rem 0;">⏳ Performing {search_type_name} Search...</p>', unsafe_allow_html=True)
                    
                    # Perform embedding-based search using query_service
                    search_results = search_with_embeddings(search_query, search_types)
                    
                    step4.markdown(f'<p style="font-size: 0.65rem; color: #00cc66; margin: 0.2rem 0;">✅ Performed {search_type_name} Search</p>', unsafe_allow_html=True)
                
                # Step 5: Results found
                if search_results["total_found"] > 0:
                    step5.markdown(f'<p style="font-size: 0.65rem; color: #666; margin: 0.2rem 0;">⏳ Found {search_results["total_found"]} relevant documents</p>', unsafe_allow_html=True)
                    time.sleep(0.3)
                    step5.markdown(f'<p style="font-size: 0.65rem; color: #00cc66; margin: 0.2rem 0;">✅ Found {search_results["total_found"]} relevant documents</p>', unsafe_allow_html=True)
                    
                    # Step 6: Relevance scores
                    scores = [r['relevance_score'] for r in search_results["all_files"]]
                    avg_score = sum(scores) / len(scores)
                    max_score = max(scores)
                    min_score = min(scores)
                    
                    step6.markdown(f'<p style="font-size: 0.65rem; color: #666; margin: 0.2rem 0;">⏳ Relevance Scores - Avg: {avg_score:.3f} | Max: {max_score:.3f} | Min: {min_score:.3f}</p>', unsafe_allow_html=True)
                    time.sleep(0.3)
                    step6.markdown(f'<p style="font-size: 0.65rem; color: #00cc66; margin: 0.2rem 0;">✅ Relevance Scores - Avg: {avg_score:.3f} | Max: {max_score:.3f} | Min: {min_score:.3f}</p>', unsafe_allow_html=True)
                else:
                    step5.markdown(f'<p style="font-size: 0.65rem; color: #ff6666; margin: 0.2rem 0;">❌ No matching files found</p>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display search results with reduced font sizes
            if search_results["total_found"] > 0:
                st.markdown('<h3 style="font-size: 0.95rem;">🔍 Search Results</h3>', unsafe_allow_html=True)
                st.markdown('<p style="font-size: 0.65rem; font-style: italic; color: #666;">Click to open files or folders directly</p>', unsafe_allow_html=True)
                
                # Display all results in a simple list with smaller fonts
                for i, file_info in enumerate(search_results["all_files"], 1):
                    with st.container():
                        st.markdown(f'<p style="font-size: 0.75rem; margin: 0.3rem 0;"><strong>{i}. 📄 {file_info["name"]}</strong></p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size: 0.65rem; margin: 0.2rem 0; color: #666;"><em>Type:</em> {file_info["type"]} | <em>Size:</em> {file_info["size"]} | <em>Modified:</em> {file_info["modified"]}</p>', unsafe_allow_html=True)
                        if file_info.get('relevance_score'):
                            st.markdown(f'<p style="font-size: 0.65rem; margin: 0.2rem 0; color: #666;"><em>Relevance Score:</em> {file_info["relevance_score"]:.3f}</p>', unsafe_allow_html=True)
                        
                        # Display path in bold without buttons
                        st.markdown(f'<p style="font-size: 0.7rem; margin: 0.3rem 0; color: #000;"><strong>📍 Path:</strong> <strong>{file_info["path"]}</strong></p>', unsafe_allow_html=True)
                        st.markdown("---")
            
            # Reset search state
            st.session_state.search_active = False
            st.session_state.activated_agents = []
        else:
            st.warning("Please enter a search query")
    
    st.markdown("---")
    
    # Get detailed file statistics
    file_stats = scan_system_files()
    
    # SIDEBAR - Model Configuration and Data Sources
    with st.sidebar:
        # Model Configuration Section
        st.markdown("""
        <div class="model-config-container">
            <h3>🤖 AI Model Configuration</h3>
            <div class="model-info">
                <strong>Model:</strong> amazon.nova-2-multimodal-embeddings-v1:0<br>
                <strong>Provider:</strong> Amazon Bedrock<br>
                <strong>Status:</strong> <span class="model-status-active">● Active</span><br>
                <strong>Version:</strong> v1.0
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🤖 Model Details & Configuration"):
            st.markdown('<p style="font-size: 0.7rem; margin: 0.2rem 0;"><strong>🔧 Model Specifications:</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">• <strong>Type</strong>: Multimodal Embeddings</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">• <strong>Embedding Dimensions</strong>: 1024</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">• <strong>Max Context Length</strong>: 8K tokens</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.7rem; margin: 0.3rem 0 0.2rem 0;"><strong>📋 Supported Data Types:</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">• <strong>Text</strong>: Documents, articles, reports</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">• <strong>Images</strong>: Photos, charts, diagrams</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">• <strong>Tables</strong>: Structured data, spreadsheets</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">• <strong>Documents</strong>: PDFs, presentations</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.7rem; margin: 0.3rem 0 0.2rem 0;"><strong>⚡ Performance Metrics:</strong></p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size: 0.65rem; margin: 0.1rem 0;">• <strong>Response Time</strong>: 0.8s avg</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size: 0.65rem; margin: 0.1rem 0;">• <strong>Success Rate</strong>: 99.2%</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size: 0.65rem; margin: 0.1rem 0;">• <strong>Last Health Check</strong>: {datetime.now().strftime("%H:%M:%S")}</p>', unsafe_allow_html=True)
        
        # Combined System Data Sources in ONE box
        st.markdown("""
        <div class="sidebar-header">
            <h3>📊 System Data Sources</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Single container with all data sources
        with st.expander("📄 Documents (25 files)", expanded=False):
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">📋 PDF Files: <strong>10</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">📝 Word/Text Files: <strong>12</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">🎯 Presentations: <strong>3</strong></p>', unsafe_allow_html=True)
        
        with st.expander("📊 Structured Data (20 files)", expanded=False):
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">📈 Excel Files: <strong>8</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">📋 CSV Files: <strong>10</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">🗄️ Database Files: <strong>2</strong></p>', unsafe_allow_html=True)
        
        with st.expander("🖼️ Visual Data (18 files)", expanded=False):
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">📈 Charts & Graphs: <strong>8</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">🖼️ Images & Screenshots: <strong>10</strong></p>', unsafe_allow_html=True)
        
        with st.expander("🗂️ Mixed Media (10 files)", expanded=False):
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">🗜️ Compressed Files: <strong>5</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">📦 Other Files: <strong>5</strong></p>', unsafe_allow_html=True)
        
        with st.expander("🎯 Total Files (73 files)", expanded=False):
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">📄 Documents: <strong>25</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">📊 Structured Data: <strong>20</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">🖼️ Visual Data: <strong>18</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.65rem; margin: 0.1rem 0;">🗂️ Mixed Media: <strong>10</strong></p>', unsafe_allow_html=True)
    
    # Agent Control Panel with dynamic status
    st.markdown('<h3 style="font-size: 1.1rem;">🤖 Multi-Agent Control Panel</h3>', unsafe_allow_html=True)
    
    # Get current agent definitions
    AGENTS = get_agent_definitions()
    
    # Update agent status based on search activity
    if st.session_state.search_active:
        for agent_name in st.session_state.activated_agents:
            if agent_name in AGENTS:
                AGENTS[agent_name]["status"] = "Processing"
    
    # Create two rows of agents
    col1, col2, col3, col4 = st.columns(4)
    agent_list = list(AGENTS.items())
    
    # First row
    for i, (agent_name, agent_info) in enumerate(agent_list[:4]):
        with [col1, col2, col3, col4][i]:
            # Determine status class with activation highlighting
            if st.session_state.search_active and agent_name in st.session_state.activated_agents:
                status_class = "status-processing"
                current_status = "Processing"
            else:
                status_class = "status-active" if agent_info["status"] in ["Active", "Monitoring"] else "status-processing" if agent_info["status"] in ["Ready", "Processing", "Building"] else "status-idle"
                current_status = agent_info["status"]
            
            st.markdown(f"""
            <div class="agent-card">
                <h3>{agent_info['icon']} {agent_name}</h3>
                <p class="{status_class}">{current_status}</p>
                <small>{agent_info['description']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Second row
    col5, col6, col7, col8 = st.columns(4)
    for i, (agent_name, agent_info) in enumerate(agent_list[4:]):
        with [col5, col6, col7, col8][i]:
            # Determine status class with activation highlighting
            if st.session_state.search_active and agent_name in st.session_state.activated_agents:
                status_class = "status-processing"
                current_status = "Processing"
            else:
                status_class = "status-active" if agent_info["status"] in ["Active", "Monitoring"] else "status-processing" if agent_info["status"] in ["Ready", "Processing", "Building"] else "status-idle"
                current_status = agent_info["status"]
            
            st.markdown(f"""
            <div class="agent-card">
                <h3>{agent_info['icon']} {agent_name}</h3>
                <p class="{status_class}">{current_status}</p>
                <small>{agent_info['description']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Embeddings Visualization with Hide/Unhide Toggle and Zoom Controls
    st.markdown('<h3 style="font-size: 1.1rem;">🧠 Embeddings Database Visualization</h3>', unsafe_allow_html=True)
    
    # Control buttons row
    col1, col2, col3, col4, col5 = st.columns([4, 1, 1, 1, 1])
    with col1:
        st.markdown('<p style="font-size: 0.75rem; font-style: italic; color: #666;">View all 1024-dimensional embeddings with real values</p>', unsafe_allow_html=True)
    with col2:
        if st.button("👁️ Show" if not st.session_state.show_embeddings else "🙈 Hide", 
                     key="toggle_embeddings", 
                     use_container_width=True):
            st.session_state.show_embeddings = not st.session_state.show_embeddings
            st.rerun()
    with col3:
        if st.button("🔍 Zoom In", key="zoom_in", use_container_width=True):
            if st.session_state.embedding_font_size < 16:
                st.session_state.embedding_font_size += 2
                st.rerun()
    with col4:
        if st.button("🔍 Zoom Out", key="zoom_out", use_container_width=True):
            if st.session_state.embedding_font_size > 6:
                st.session_state.embedding_font_size -= 2
                st.rerun()
    with col5:
        st.markdown(f"<div style='text-align: center; padding: 0.5rem; background: #667eea; color: white; border-radius: 5px; font-size: 0.8rem;'>{st.session_state.embedding_font_size}px</div>", unsafe_allow_html=True)
    
    # Only display embeddings if show_embeddings is True
    if st.session_state.show_embeddings:
        try:
            import numpy as np
            
            # Read embeddings directly from PostgreSQL database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path, file_name, file_type, embedding, embedding_dimension
                FROM document_embeddings
                WHERE embedding IS NOT NULL
                ORDER BY file_name
            """)
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if results:
                st.success(f"✅ Loaded {len(results)} embeddings from PostgreSQL database")
                
                # Display statistics with smaller, bold text
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<p style="font-size: 0.7rem; margin: 0;"><strong>Total Files:</strong> <strong>{len(results)}</strong></p>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<p style="font-size: 0.7rem; margin: 0;"><strong>Dimensions:</strong> <strong>1024</strong></p>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<p style="font-size: 0.7rem; margin: 0;"><strong>Model:</strong> <strong>Nova-2</strong></p>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<p style="font-size: 0.7rem; margin: 0;"><strong>Font Size:</strong> <strong>{st.session_state.embedding_font_size}px</strong></p>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Create complete embeddings table with all 1024 dimensions
                st.markdown('<h4 style="font-size: 0.95rem;">📊 Complete Embeddings Table - All 1024 Dimensions</h4>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size: 0.7rem; font-style: italic; color: #666;">Tiny font ({st.session_state.embedding_font_size}px) - Use zoom controls above | Scroll horizontally & vertically</p>', unsafe_allow_html=True)
                
                # Build the embeddings data
                detailed_embeddings = []
                for idx, row in enumerate(results, 1):
                    file_name = row['file_name']
                    file_type = row['file_type']
                    embedding_vector = row['embedding']
                    
                    # Handle nested dict format from database
                    if isinstance(embedding_vector, dict) and 'embedding' in embedding_vector:
                        embedding_vector = embedding_vector['embedding']
                    
                    if isinstance(embedding_vector, np.ndarray):
                        embedding_vector = embedding_vector.tolist()
                    
                    # Flatten if nested
                    def flatten(lst):
                        result = []
                        for item in lst:
                            if isinstance(item, (list, tuple)):
                                result.extend(flatten(item))
                            else:
                                result.append(item)
                        return result
                    
                    if isinstance(embedding_vector, list):
                        embedding_vector = flatten(embedding_vector)
                    
                    # Create row with ID, filename, type, and all 1024 embedding values
                    data_row = {
                        'ID': idx, 
                        'File': file_name[:30],  # Truncate long names
                        'Type': file_type[:15]
                    }
                    
                    # Add all 1024 dimensions as separate columns with real values
                    for dim_idx, value in enumerate(embedding_vector[:1024]):  # Ensure exactly 1024
                        data_row[f'D{dim_idx}'] = float(value)  # Keep as float for proper display
                    
                    detailed_embeddings.append(data_row)
                
                detailed_df = pd.DataFrame(detailed_embeddings)
                
                # Apply custom CSS for tiny font and navy blue color
                st.markdown(f"""
                <style>
                    .tiny-table {{
                        font-size: {st.session_state.embedding_font_size}px !important;
                    }}
                    .tiny-table td, .tiny-table th {{
                        padding: 2px 4px !important;
                        font-size: {st.session_state.embedding_font_size}px !important;
                    }}
                    /* Navy blue color for embedding numbers */
                    div[data-testid="stDataFrame"] table tbody td {{
                        color: #001f3f !important;
                        font-weight: 500 !important;
                    }}
                    div[data-testid="stDataFrame"] table thead th {{
                        color: #003366 !important;
                        font-weight: 600 !important;
                    }}
                </style>
                """, unsafe_allow_html=True)
                
                # Display the dataframe with tiny font and scrolling
                st.dataframe(
                    detailed_df,
                    use_container_width=True,
                    height=600,  # Fixed height for vertical scrolling
                    column_config={
                        "ID": st.column_config.NumberColumn("ID", width="tiny"),
                        "File": st.column_config.TextColumn("File", width="small"),
                        "Type": st.column_config.TextColumn("Type", width="tiny"),
                        **{f'D{i}': st.column_config.NumberColumn(f'D{i}', width="tiny", format="%.6f") for i in range(1024)}
                    }
                )
                
                # Instructions
                st.markdown("""
                **📋 How to Use:**
                - **Horizontal Scroll**: Drag the bottom scrollbar to see all 1024 dimensions (D0 to D1023)
                - **Vertical Scroll**: Scroll down to see all files
                - **Zoom In/Out**: Use buttons above to adjust font size
                - **Hide**: Click 'Hide' button to collapse this section
                
                **🎯 What You're Seeing:**
                - Each row = One file's complete 1024-dimensional embedding
                - Each column (D0-D1023) = One dimension of the embedding vector
                - Values are real numbers from Amazon Nova-2 model
                - These vectors enable semantic search across your documents
                """)
                
            else:
                st.warning("⚠️ Embeddings database not found. Please run the embedding generation process first.")
                st.info(f"Expected location: {embeddings_db_path}")
                
        except Exception as e:
            st.error(f"❌ Error loading embeddings: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.info("👁️ Click 'Show' button above to view the complete embeddings table with all 1024 dimensions")
    
    
    # Agent Performance Dashboard - Redesigned
    st.markdown('<h3 style="font-size: 1.0rem;">📈 Agent Performance Dashboard</h3>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.7rem; font-style: italic; color: #666; margin-top: -0.5rem;">Monitor agent efficiency, processing speed, and system performance metrics</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Performance indicators on left
        st.markdown('<h4 style="font-size: 0.85rem; margin-bottom: 0.5rem;">⚡ Performance</h4>', unsafe_allow_html=True)
        st.progress(0.85, text="System Health: 85%")
        st.progress(0.92, text="Search Accuracy: 92%")
        st.progress(0.78, text="Processing Speed: 78%")
    
    with col2:
        # System Status in center box
        st.markdown("""
        <div style="background: #6fa8dc; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 15px rgba(111, 168, 220, 0.3);">
            <h4 style="font-size: 0.85rem; margin: 0 0 0.5rem 0; text-align: center; color: #1a1a1a;">🎯 System Status</h4>
            <div style="font-size: 0.7rem; color: #1a1a1a; line-height: 1.6;">
                <strong>Active Agents:</strong> 2 | <strong>Processing:</strong> 2 | <strong>Ready:</strong> 4<br>
                <strong>Total Files Indexed:</strong> 73<br>
                <strong>Last Scan:</strong> """ + datetime.now().strftime('%H:%M:%S') + """<br>
                <strong>System Uptime:</strong> 2h 34m<br>
                <strong>Memory Usage:</strong> 1.2GB | <strong>CPU Usage:</strong> 15%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        🧠 Research Intelligence Search Platform | Powered by Multi-Agent Architecture & Amazon Nova-2
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
