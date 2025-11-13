import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool, tool
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools import TavilySearchResults
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langdetect import detect, DetectorFactory
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb
from chromadb.config import Settings

from enhanced_config import (
    GEMINI_API_KEY, TAVILY_API_KEY, SERPAPI_API_KEY,
    LLM_MODEL, TEMPERATURE, MAX_TOKENS, AGENT_TIMEOUT,
    SUPPORTED_LANGUAGES, ENABLE_WEB_SEARCH, ENABLE_ORDER_TRACKING,
    MAX_CONVERSATION_HISTORY, SESSION_TIMEOUT, FEEDBACK_FILE,
    DOCUMENTS_UPLOAD_DIR, CACHE_DIR
)

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= KNOWLEDGE BASE =============
KNOWLEDGE_BASE = {
    "products": [
        {
            "id": "P001",
            "name": "Premium Wireless Headphones",
            "description": "High-quality wireless headphones with noise cancellation",
            "price": 199.99,
            "specs": {
                "battery": "40 hours",
                "connectivity": "Bluetooth 5.0",
                "noise_cancellation": "Active ANC",
                "weight": "250g"
            },
            "in_stock": True,
            "category": "Electronics"
        },
        {
            "id": "P002",
            "name": "Eco-Friendly Water Bottle",
            "description": "Sustainable water bottle made from recycled materials",
            "price": 29.99,
            "specs": {
                "capacity": "750ml",
                "material": "Recycled plastic",
                "insulated": "Double-wall",
                "colors": ["Blue", "Green", "Red"]
            },
            "in_stock": True,
            "category": "Lifestyle"
        },
        {
            "id": "P003",
            "name": "Nike Air Jordan 1 Retro",
            "description": "Classic Nike Air Jordan 1 Retro High basketball shoes",
            "price": 175.00,
            "specs": {
                "brand": "Nike",
                "type": "Basketball Shoes",
                "model": "Air Jordan 1 Retro High",
                "colors": ["Black/Red", "White/Blue", "Chicago"]
            },
            "in_stock": True,
            "category": "Shoes"
        },
        {
            "id": "P004",
            "name": "Nike Air Force 1",
            "description": "Iconic Nike Air Force 1 casual sneakers for everyday wear",
            "price": 120.00,
            "specs": {
                "brand": "Nike",
                "type": "Casual Sneakers",
                "model": "Air Force 1",
                "colors": ["White", "Black", "Grey"]
            },
            "in_stock": True,
            "category": "Shoes"
        },
        {
            "id": "P005",
            "name": "Nike LeBron 21",
            "description": "Latest Nike LeBron 21 professional basketball shoes",
            "price": 185.00,
            "specs": {
                "brand": "Nike",
                "type": "Basketball Shoes",
                "model": "LeBron 21",
                "colors": ["Purple/Gold", "Black/Gold"]
            },
            "in_stock": True,
            "category": "Shoes"
        },
    ],
    "faqs": [
        {"question": "What is your return policy?", "answer": "We offer 30-day returns on most items."},
        {"question": "Do you ship internationally?", "answer": "Yes, we ship to 150+ countries worldwide."},
        {"question": "How long does shipping take?", "answer": "Typically 3-7 business days for standard shipping."},
    ],
    "company_info": {
        "name": "GlobalTech",
        "support_email": "support@globaltech.com",
        "support_phone": "+1-800-GLOBALTECH",
        "business_hours": "24/7 Support",
        "website": "www.globaltech.com"
    }
}

# ============= MULTILINGUAL SUPPORT SYSTEM =============
class AdvancedMultilingualSupportAgent:
    def __init__(self):
        self.conversation_history = defaultdict(list)
        self.session_metadata = defaultdict(dict)
        self.user_preferences = defaultdict(dict)
        self.order_database = self._load_mock_orders()
        self.feedback_data = []
        
        # Initialize LLM with Gemini
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required!")
        
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL or "gemini-2.0-flash",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=GEMINI_API_KEY,
            convert_system_message_to_human=True
        )
        
        # Sentiment analyzer disabled - using simple heuristic instead
        self.sentiment_analyzer = None
        
        # Initialize vector database
        self._init_vector_db()
        
        # Initialize memory manager
        self.memory_manager = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Build knowledge vectors
        self._build_knowledge_vectors()
        
        # Create agent with tools
        self.agent_executor = self._create_agent()
        
        logger.info("Advanced Multilingual Support Agent initialized successfully")
    
    def _init_vector_db(self):
        """Initialize Chroma vector database"""
        try:
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            )
            self.vector_client = chromadb.Client(settings)
            self.vector_collection = self.vector_client.get_or_create_collection(
                name="globaltech_support",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector database initialized")
        except Exception as e:
            logger.error(f"Vector DB initialization error: {e}")
            self.vector_collection = None
    
    def _build_knowledge_vectors(self):
        """Build vector embeddings for knowledge base"""
        product_texts = [
            f"Product: {p['name']}, Price: ${p['price']}, Description: {p['description']}, Specs: {json.dumps(p['specs'])}"
            for p in KNOWLEDGE_BASE['products']
        ]
        faq_texts = [
            f"Q: {f['question']} A: {f['answer']}"
            for f in KNOWLEDGE_BASE['faqs']
        ]
        
        self.documents = product_texts + faq_texts
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    
    # ============= AGENT TOOLS =============
    @tool
    def product_search(query: str) -> str:
        """Search for products - local first, then web search"""
        query_lower = query.lower()
        results = []
        
        # First try local knowledge base
        for product in KNOWLEDGE_BASE['products']:
            if query_lower in product['name'].lower() or query_lower in product['description'].lower():
                results.append({
                    "id": product['id'],
                    "name": product['name'],
                    "price": product['price'],
                    "in_stock": product['in_stock'],
                    "source": "catalog"
                })
        
        # If found locally, return results
        if results:
            return json.dumps(results)
        
        # If not found locally, search web
        if ENABLE_WEB_SEARCH and TAVILY_API_KEY:
            try:
                search = TavilySearchResults(max_results=5, api_key=TAVILY_API_KEY)
                web_results = search.run(f"{query} price")
                return f"Found online results for '{query}':\n\n{str(web_results)}"
            except Exception as e:
                logger.error(f"Web search error: {e}")
                return f"No results found for '{query}' in our catalog. Try searching on major e-commerce sites."
        
        return f"No products found matching '{query}'. Enable web search for online results."
    
    @tool
    def order_tracker(order_id: str) -> str:
        """Track customer order status"""
        if order_id in self.order_database:
            order = self.order_database[order_id]
            return json.dumps(order)
        return f"Order {order_id} not found. Please provide a valid order ID."
    
    @tool
    def faq_retriever(query: str) -> str:
        """Retrieve FAQ answers"""
        query_lower = query.lower()
        results = []
        
        for faq in KNOWLEDGE_BASE['faqs']:
            if query_lower in faq['question'].lower():
                results.append(faq)
        
        return json.dumps(results) if results else "No matching FAQs found."
    
    @tool
    def web_search(query: str) -> str:
        """Search the internet for information"""
        if not ENABLE_WEB_SEARCH:
            return "Web search is not enabled."
        
        if not TAVILY_API_KEY:
            return "Web search API key not configured."
        
        try:
            search = TavilySearchResults(max_results=5, api_key=TAVILY_API_KEY)
            results = search.run(query)
            return str(results)
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search failed: {str(e)}"
    
    @tool
    def product_recommendation(preferences: str) -> str:
        """Get product recommendations based on preferences"""
        recommendations = []
        for product in KNOWLEDGE_BASE['products']:
            recommendations.append({
                "id": product['id'],
                "name": product['name'],
                "price": product['price'],
                "category": product['category']
            })
        
        return json.dumps(recommendations[:5])
    
    @tool
    def email_support(query: str) -> str:
        """Escalate to email support"""
        return f"Your request has been escalated to our support team. Email: {KNOWLEDGE_BASE['company_info']['support_email']}"
    
    @tool
    def company_info(query: str) -> str:
        """Get company information"""
        return json.dumps(KNOWLEDGE_BASE['company_info'])
    
    def _create_agent(self) -> AgentExecutor:
        """Create LangChain agent with tools"""
        tools = [
            Tool(
                name="product_search",
                func=self.product_search,
                description="Search for products in our catalog"
            ),
            Tool(
                name="order_tracker",
                func=self.order_tracker,
                description="Track order status by order ID"
            ),
            Tool(
                name="faq_retriever",
                func=self.faq_retriever,
                description="Retrieve FAQ answers"
            ),
            Tool(
                name="web_search",
                func=self.web_search,
                description="Search the internet for information"
            ),
            Tool(
                name="product_recommendation",
                func=self.product_recommendation,
                description="Get product recommendations"
            ),
            Tool(
                name="email_support",
                func=self.email_support,
                description="Escalate to email support team"
            ),
            Tool(
                name="company_info",
                func=self.company_info,
                description="Get company information"
            ),
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful, multilingual customer support AI assistant for GlobalTech. 
You have access to various tools to help answer customer questions accurately and efficiently.
Be friendly, professional, and thorough in your responses.
Always try to use the available tools to find accurate information.
If you can't find the answer, suggest contacting support."""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=self.memory_manager,
            verbose=True,
            max_iterations=10,
            timeout=AGENT_TIMEOUT
        )
    
    # ============= LANGUAGE HANDLING =============
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            lang = detect(text)
            return lang if lang in SUPPORTED_LANGUAGES else 'en'
        except:
            return 'en'
    
    def translate_text(self, text: str, target_language: str, source_language: str = 'en') -> str:
        """Translate text using Gemini LLM"""
        if source_language == target_language:
            return text
        
        try:
            translate_prompt = f"Translate the following {source_language} text to {target_language}. Only provide the translation, no explanations:\n\n{text}"
            response = self.llm.invoke([HumanMessage(content=translate_prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    # ============= SENTIMENT & ANALYTICS =============
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using simple heuristic (no model download needed)"""
        try:
            text_lower = text.lower()
            
            # Simple sentiment heuristic
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'awesome', 'happy', 'thanks']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'problem', 'issue', 'error', 'sad', 'angry']
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                label = "POSITIVE"
                score = min(0.5 + (positive_count * 0.1), 1.0)
            elif negative_count > positive_count:
                label = "NEGATIVE"
                score = max(0.5 - (negative_count * 0.1), 0.0)
            else:
                label = "NEUTRAL"
                score = 0.5
            
            return {
                "label": label,
                "score": score,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"label": "NEUTRAL", "score": 0.5, "timestamp": datetime.now().isoformat()}
    
    # ============= MAIN RESPONSE GENERATION =============
    def generate_response(
        self,
        user_message: str,
        session_id: str = 'default',
        user_language: Optional[str] = None,
        response_language: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate multilingual response using LangChain agent"""
        try:
            # Detect input language
            detected_language = self.detect_language(user_message)
            
            # Use provided language or detected language
            input_language = user_language or detected_language
            output_language = response_language or input_language
            
            # Store user preference
            self.user_preferences[session_id]['language'] = output_language
            
            # Translate to English for processing
            english_query = user_message if input_language == 'en' else self.translate_text(user_message, 'en', input_language)
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(user_message)
            
            # Add category context if provided
            context_message = f"Category: {category}\n" if category else ""
            final_query = context_message + english_query
            
            # Generate response using agent
            logger.info(f"Processing query: {english_query[:100]}...")
            agent_response = self.agent_executor.invoke({
                "input": final_query,
                "chat_history": self._get_chat_history(session_id)
            })
            
            # Extract response text
            response_text = agent_response.get('output', 'Sorry, I could not generate a response.')
            
            # Translate response to output language
            final_response = response_text if output_language == 'en' else self.translate_text(response_text, output_language, 'en')
            
            # Update conversation history
            self._update_memory(session_id, user_message, final_response, input_language, output_language, sentiment)
            
            return {
                "success": True,
                "response": final_response,
                "input_language": input_language,
                "output_language": output_language,
                "sentiment": sentiment,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "category": category,
            }
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return {
                "success": False,
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # ============= SESSION & HISTORY MANAGEMENT =============
    def _update_memory(self, session_id: str, user_msg: str, bot_resp: str, input_lang: str, output_lang: str, sentiment: Dict):
        """Update conversation memory"""
        self.conversation_history[session_id].append({
            "user_message": user_msg,
            "bot_response": bot_resp,
            "input_language": input_lang,
            "output_language": output_lang,
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if too long
        if len(self.conversation_history[session_id]) > MAX_CONVERSATION_HISTORY:
            self.conversation_history[session_id] = self.conversation_history[session_id][-MAX_CONVERSATION_HISTORY:]
    
    def _get_chat_history(self, session_id: str, limit: int = 5) -> List[Tuple]:
        """Get recent chat history"""
        history = []
        for msg in self.conversation_history[session_id][-limit:]:
            history.append(("human", msg["user_message"]))
            history.append(("ai", msg["bot_response"]))
        return history
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get full conversation history"""
        return self.conversation_history.get(session_id, [])
    
    def clear_history(self, session_id: str):
        """Clear conversation history"""
        self.conversation_history[session_id] = []
        self.session_metadata[session_id] = {}
    
    def save_feedback(self, session_id: str, feedback_data: Dict):
        """Save user feedback"""
        feedback_entry = {
            "session_id": session_id,
            "rating": feedback_data.get("rating", 0),
            "comment": feedback_data.get("comment", ""),
            "timestamp": datetime.now().isoformat()
        }
        self.feedback_data.append(feedback_entry)
        self._persist_feedback()
    
    def _persist_feedback(self):
        """Persist feedback to file"""
        try:
            os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
            with open(FEEDBACK_FILE, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting feedback: {e}")
    
    # ============= ORDER MANAGEMENT =============
    def _load_mock_orders(self) -> Dict[str, Dict]:
        """Load mock order database"""
        return {
            "ORD001": {
                "id": "ORD001",
                "status": "shipped",
                "tracking": "TRK123456",
                "estimated_delivery": (datetime.now() + timedelta(days=3)).isoformat(),
                "items": [{"name": "Premium Wireless Headphones", "qty": 1}]
            },
            "ORD002": {
                "id": "ORD002",
                "status": "processing",
                "tracking": None,
                "estimated_delivery": (datetime.now() + timedelta(days=5)).isoformat(),
                "items": [{"name": "Eco-Friendly Water Bottle", "qty": 2}]
            },
        }
    
    # ============= ANALYTICS =============
    def get_analytics(self) -> Dict[str, Any]:
        """Get system analytics"""
        total_conversations = len(self.conversation_history)
        total_messages = sum(len(msgs) for msgs in self.conversation_history.values())
        
        languages_used = defaultdict(int)
        for session_msgs in self.conversation_history.values():
            for msg in session_msgs:
                lang = msg.get('input_language', 'en')
                languages_used[lang] += 1
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "languages_used": dict(languages_used),
            "average_sentiment": np.mean([msg['sentiment']['score'] for msgs in self.conversation_history.values() for msg in msgs]) if total_messages > 0 else 0.5,
            "total_feedback": len(self.feedback_data),
            "timestamp": datetime.now().isoformat()
        }


# Initialize the agent
support_agent = AdvancedMultilingualSupportAgent()