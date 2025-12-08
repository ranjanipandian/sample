"""
Q&A Service for Research Intelligence Search Platform
Uses GPT-4o to answer questions based on document database
"""

import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

# Database Configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME', 'research_kb')

# Initialize Azure OpenAI client
try:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    print("✅ Azure OpenAI client initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize Azure OpenAI client: {e}")
    client = None

def get_db_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def get_relevant_context(query, max_docs=5):
    """
    Retrieve relevant document content from database to provide context for GPT-4o
    Uses semantic search for better relevance
    """
    context_docs = []
    
    try:
        # Import semantic_search from query_service
        from query_service import semantic_search
        
        # Use semantic search to find relevant documents
        results = semantic_search(query, top_k=max_docs, min_score=0.5)
        
        for result in results:
            if result['content_preview']:
                context_docs.append(
                    f"Document: {result['file_name']} ({result['file_type']}) [Relevance: {result['relevance_score']:.2f}]\n"
                    f"Content: {result['content_preview'][:1500]}\n"
                )
        
    except Exception as e:
        print(f"❌ Error retrieving context: {e}")
        import traceback
        traceback.print_exc()
    
    return "\n---\n".join(context_docs) if context_docs else "No relevant documents found in database."

def extract_insights(answer):
    """
    Extract key insights from the GPT-4o answer
    """
    if not client:
        return []
    
    try:
        insight_prompt = f"""Analyze the following answer and extract 3-5 key insights or important points as bullet points.
Be concise and focus on the most important information.

Answer:
{answer}

Provide only the bullet points, one per line, starting with a dash (-)."""
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "user", "content": insight_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        insights_text = response.choices[0].message.content
        # Parse bullet points
        insights = [line.strip().lstrip('-').strip() for line in insights_text.split('\n') if line.strip().startswith('-')]
        return insights
        
    except Exception as e:
        print(f"❌ Error extracting insights: {e}")
        return []

def answer_question(question):
    """
    Use GPT-4o to answer questions based on document database
    Returns a dictionary with answer and insights
    """
    if not client:
        return {
            'answer': "❌ Error: Azure OpenAI client not initialized. Please check your credentials in .env file.",
            'insights': []
        }
    
    try:
        # Get relevant context from database
        context = get_relevant_context(question, max_docs=5)
        
        # Create system prompt
        system_prompt = """You are a helpful research assistant with access to a document database. 
Your task is to answer questions directly and concisely based on the provided document context.
- Provide direct, factual answers without meta-commentary about what the documents contain or don't contain
- If the documents have relevant information, use it to answer the question directly
- Cite document names when referencing specific information
- If documents lack information, supplement with your general knowledge while noting what came from documents vs. general knowledge
- Be concise and informative"""
        
        # Create user prompt with context
        user_prompt = f"""Question: {question}

Relevant Documents from Database:
{context}

Please answer the question based on the above documents. If the documents don't contain relevant information, please state that clearly."""
        
        # Call GPT-4o
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Extract insights from the answer
        insights = extract_insights(answer)
        
        return {
            'answer': answer,
            'insights': insights
        }
        
    except Exception as e:
        print(f"❌ Error in Q&A service: {e}")
        import traceback
        traceback.print_exc()
        return {
            'answer': f"❌ Error generating answer: {str(e)}",
            'insights': []
        }

# -------------------- Main (for testing) --------------------
if __name__ == "__main__":
    print("=" * 70)
    print("💬 Q&A Service Test (GPT-4o)")
    print("=" * 70)
    
    # Test database connection
    conn = get_db_connection()
    if conn:
        print("✅ Database connection successful")
        conn.close()
    else:
        print("❌ Database connection failed")
        exit(1)
    
    # Test Azure OpenAI
    if not client:
        print("❌ Azure OpenAI client not initialized")
        exit(1)
    
    print("✅ Azure OpenAI client ready")
    print("\n" + "=" * 70)
    
    # Interactive Q&A loop
    while True:
        question = input("\nAsk a question (or 'exit' to quit): ").strip()
        if question.lower() == 'exit':
            break
        
        print("\n💭 Thinking...")
        answer = answer_question(question)
        print("\n📝 Answer:")
        print(answer)
        print("\n" + "=" * 70)
