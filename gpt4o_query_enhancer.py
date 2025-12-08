"""
GPT-4o Query Enhancer for Semantic Search
Improves query understanding before semantic search
Handles: synonyms, spelling mistakes, abbreviations, context
"""

import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

# Initialize Azure OpenAI client
try:
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    print("✅ GPT-4o Query Enhancer initialized")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize GPT-4o: {e}")
    client = None

def enhance_query_with_gpt4o(query):
    """
    Use GPT-4o to enhance query for better semantic search
    
    Handles:
    - Spelling mistakes: "coronery" → "coronary"
    - Abbreviations: "CHD" → "coronary heart disease"
    - Synonyms: "brain tumor" → "glioblastoma, brain cancer, brain neoplasm"
    - Context: Adds related medical terms
    
    Returns enhanced query string
    """
    
    if not client:
        print("⚠️ GPT-4o not available, using original query")
        return query
    
    try:
        print(f"🧠 Enhancing query with GPT-4o: '{query}'")
        
        system_prompt = """You are a medical research query enhancement expert.
Your job is to improve search queries for better semantic search results.

Tasks:
1. Fix spelling mistakes
2. Expand medical abbreviations (e.g., CHD → coronary heart disease)
3. Add relevant synonyms and related medical terms
4. Add context-appropriate terms

IMPORTANT: Return ONLY the enhanced query terms, separated by spaces or commas.
Do NOT add explanations or extra text."""

        user_prompt = f"""Original query: {query}

Enhance this medical/research query by:
1. Fixing any spelling mistakes (e.g., "coronery" → "coronary", "diabetis" → "diabetes")
2. Expanding abbreviations (e.g., CHD → coronary heart disease, MI → myocardial infarction, CAD → coronary artery disease)
3. Adding 3-5 relevant synonyms and related medical terms
4. Adding context-appropriate medical terminology
5. Including common variations and alternative names

Examples:
- "brain tumor" → "brain tumor glioblastoma brain cancer brain neoplasm intracranial tumor cerebral tumor"
- "CHD" → "CHD coronary heart disease cardiac disease heart condition coronary artery disease cardiovascular disease"
- "clinical studies" → "clinical studies clinical trials research studies medical research clinical research patient studies"
- "coronery" → "coronary heart disease CHD cardiac disease cardiovascular disease coronary artery disease"
- "diabetis" → "diabetes diabetes mellitus diabetic condition blood sugar disorder glucose metabolism"

Be comprehensive - include ALL relevant medical terms, synonyms, and variations.
Return ONLY the enhanced query (one line, space or comma separated terms)."""

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        enhanced_query = response.choices[0].message.content.strip()
        
        # Clean up the response (remove quotes, extra formatting)
        enhanced_query = enhanced_query.replace('"', '').replace("'", "").strip()
        
        print(f"✅ Enhanced query: '{enhanced_query}'")
        return enhanced_query
        
    except Exception as e:
        print(f"❌ Error enhancing query: {e}")
        return query  # Fallback to original query

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 GPT-4o Query Enhancer Test")
    print("=" * 70)
    
    test_queries = [
        "brain tumor",
        "CHD",
        "clinical studies",
        "de novo drug design",
        "coronery heart disease",  # Spelling mistake
        "diabetis",  # Spelling mistake
    ]
    
    for query in test_queries:
        print(f"\n📝 Original: {query}")
        enhanced = enhance_query_with_gpt4o(query)
        print(f"✨ Enhanced: {enhanced}")
        print("-" * 70)
