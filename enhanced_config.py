import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

# ============= API KEYS =============
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # REQUIRED - Get from https://makersuite.google.com/app/apikey
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # Optional - for web search
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")  # Optional - alternative search

# ============= LANGCHAIN CONFIG =============
# Using Gemini Models Only
LLM_MODEL = "gemini-2.0-flash"  # Options: "gemini-2.0-flash", "gemini-pro", "gemini-1.5-pro"
TEMPERATURE = 0.7
MAX_TOKENS = 2000
AGENT_TIMEOUT = 60

# ============= LANGUAGE CONFIG =============
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "flag": "🇬🇧"},
    "es": {"name": "Español", "flag": "🇪🇸"},
    "fr": {"name": "Français", "flag": "🇫🇷"},
    "de": {"name": "Deutsch", "flag": "🇩🇪"},
    "it": {"name": "Italiano", "flag": "🇮🇹"},
    "pt": {"name": "Português", "flag": "🇵🇹"},
    "ru": {"name": "Русский", "flag": "🇷🇺"},
    "ja": {"name": "日本語", "flag": "🇯🇵"},
    "zh": {"name": "中文", "flag": "🇨🇳"},
    "ko": {"name": "한국어", "flag": "🇰🇷"},
    "ar": {"name": "العربية", "flag": "🇸🇦"},
    "hi": {"name": "हिंदी", "flag": "🇮🇳"},
}

# ============= VECTOR STORE CONFIG =============
VECTOR_DB_PATH = "vector_store"
CHUNK_SIZE = 500
OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # For local embeddings

# ============= MEMORY CONFIG =============
MAX_CONVERSATION_HISTORY = 50
MAX_SESSIONS = 1000
SESSION_TIMEOUT = 86400  # 24 hours

# ============= FEATURE FLAGS =============
ENABLE_WEB_SEARCH = True
ENABLE_ORDER_TRACKING = True
ENABLE_PRODUCT_RECOMMENDATIONS = True
ENABLE_SENTIMENT_ANALYSIS = True
ENABLE_MULTILINGUAL_DETECTION = True
ENABLE_ANALYTICS = True
ENABLE_FEEDBACK = True
ENABLE_DOCUMENT_UPLOAD = True

# ============= SUPPORT CATEGORIES =============
SUPPORT_CATEGORIES = {
    "shipping": {"name": "Shipping & Delivery", "emoji": "📦"},
    "returns": {"name": "Returns & Refunds", "emoji": "↩️"},
    "products": {"name": "Product Information", "emoji": "🛍️"},
    "billing": {"name": "Billing & Payments", "emoji": "💳"},
    "technical": {"name": "Technical Support", "emoji": "🔧"},
    "account": {"name": "Account & Profile", "emoji": "👤"},
    "orders": {"name": "Order Status", "emoji": "📋"},
    "general": {"name": "General Inquiry", "emoji": "❓"},
}

# ============= STORAGE CONFIG =============
FEEDBACK_FILE = "data/feedback.json"
ANALYTICS_FILE = "data/analytics.json"
SESSIONS_FILE = "data/sessions.json"
DOCUMENTS_UPLOAD_DIR = "uploads/documents"
CACHE_DIR = "cache"

# ============= RATE LIMITING =============
RATE_LIMIT_ENABLED = True
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_PERIOD = 3600

# ============= APP CONFIG =============
DEBUG = os.getenv("DEBUG", "False") == "True"
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# ============= FRONTEND CONFIG =============
FRONTEND_TITLE = "GlobalTech AI Support"
FRONTEND_DESCRIPTION = "Multilingual AI-powered customer support system"
COMPANY_NAME = "GlobalTech"
COMPANY_LOGO = "/static/images/logo.png"
PRIMARY_COLOR = "#3B82F6"
SECONDARY_COLOR = "#10B981"

# ============= AGENT TOOLS =============
AGENT_TOOLS = [
    "product_search",
    "order_tracker",
    "faq_search",
    "web_search",
    "email_support",
    "product_recommendations",
    "sentiment_analyzer",
    "document_retriever",
]

# ============= DATABASE CONFIG =============
USE_VECTOR_DB = True  # Set to False if using traditional DB
VECTOR_DB_TYPE = "chromadb"  # or "faiss", "pinecone"

# ============= SECURITY =============
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB