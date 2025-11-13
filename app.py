from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import uuid
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any

from langchain_support_agent import support_agent
from enhanced_config import (
    HOST, PORT, DEBUG, CORS_ORIGINS, SUPPORTED_LANGUAGES,
    SUPPORT_CATEGORIES, RATE_LIMIT_ENABLED, RATE_LIMIT_REQUESTS, RATE_LIMIT_PERIOD
)

# ============= APP INITIALIZATION =============
app = Flask(__name__, template_folder='templates', static_folder='static')

# CORS configuration
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})

# Rate limiting
if RATE_LIMIT_ENABLED:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[f"{RATE_LIMIT_REQUESTS} per {RATE_LIMIT_PERIOD} seconds"],
        storage_uri="memory://"
    )
else:
    limiter = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= SESSION MANAGEMENT =============
active_sessions = {}

def get_or_create_session(session_id: str = None) -> str:
    """Get or create a user session"""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "language": "en",
            "category": "general"
        }
    
    return session_id

# ============= STATIC ASSETS =============
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static assets"""
    return send_from_directory('static', filename)

# ============= HEALTH CHECK =============
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "GlobalTech AI Support"
    }), 200

# ============= FRONTEND PAGES =============
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/chat')
def chat_page():
    """Chat page"""
    return render_template('chat.html')

@app.route('/dashboard')
def dashboard_page():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/kb')
def knowledge_base_page():
    """Knowledge base page"""
    return render_template('knowledge_base.html')

@app.route('/analytics')
def analytics_page():
    """Analytics page"""
    return render_template('analytics.html')

@app.route('/faq')
def faq_page():
    """FAQ page"""
    return render_template('faq.html')

# ============= API ENDPOINTS =============

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get supported languages"""
    return jsonify({
        "success": True,
        "languages": SUPPORTED_LANGUAGES
    }), 200

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get support categories"""
    return jsonify({
        "success": True,
        "categories": SUPPORT_CATEGORIES
    }), 200

@app.route('/api/chat', methods=['POST'])
@limiter.limit("30 per minute") if RATE_LIMIT_ENABLED else lambda x: x
def handle_chat():
    """Handle chat requests"""
    try:
        data = request.json
        
        # Extract parameters
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', None)
        input_language = data.get('input_language', None)
        response_language = data.get('response_language', 'en')
        category = data.get('category', 'general')
        
        # Validate input
        if not user_message:
            return jsonify({
                "success": False,
                "error": "Empty message provided"
            }), 400
        
        if len(user_message) > 5000:
            return jsonify({
                "success": False,
                "error": "Message too long (max 5000 characters)"
            }), 400
        
        # Get or create session
        session_id = get_or_create_session(session_id)
        
        # Generate response
        logger.info(f"Processing chat request for session: {session_id}")
        response = support_agent.generate_response(
            user_message=user_message,
            session_id=session_id,
            user_language=input_language,
            response_language=response_language,
            category=category
        )
        
        if response.get('success'):
            return jsonify(response), 200
        else:
            return jsonify(response), 500
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get conversation history"""
    try:
        session_id = request.args.get('session_id', 'default')
        history = support_agent.get_conversation_history(session_id)
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "history": history,
            "message_count": len(history)
        }), 200
        
    except Exception as e:
        logger.error(f"History fetch error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat_history():
    """Clear chat history"""
    try:
        data = request.json or {}
        session_id = data.get('session_id', 'default')
        
        support_agent.clear_history(session_id)
        
        return jsonify({
            "success": True,
            "message": "Chat history cleared",
            "session_id": session_id
        }), 200
        
    except Exception as e:
        logger.error(f"Clear history error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        feedback_data = {
            "rating": data.get('rating', 0),
            "comment": data.get('comment', ''),
            "message_id": data.get('message_id', '')
        }
        
        support_agent.save_feedback(session_id, feedback_data)
        
        return jsonify({
            "success": True,
            "message": "Thank you for your feedback!"
        }), 200
        
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get system analytics"""
    try:
        analytics = support_agent.get_analytics()
        return jsonify({
            "success": True,
            "analytics": analytics
        }), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new session"""
    try:
        session_id = get_or_create_session()
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "created_at": active_sessions[session_id]['created_at']
        }), 200
        
    except Exception as e:
        logger.error(f"Session creation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session_info(session_id):
    """Get session information"""
    try:
        if session_id in active_sessions:
            return jsonify({
                "success": True,
                "session": active_sessions[session_id]
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Session not found"
            }), 404
            
    except Exception as e:
        logger.error(f"Session info error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ============= ERROR HANDLERS =============
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

@app.errorhandler(429)
def rate_limit_error(error):
    """Handle rate limit errors"""
    return jsonify({
        "success": False,
        "error": "Too many requests. Please try again later."
    }), 429

# ============= VERCEL DEPLOYMENT HANDLER =============
# For Vercel serverless deployment
from waitress import serve

if __name__ == '__main__':
    if os.getenv('VERCEL') or os.getenv('ENVIRONMENT') == 'production':
        # Production mode for Vercel
        logger.info("Running in Vercel production mode")
        serve(app, host=HOST, port=PORT, _quiet=True)
    else:
        # Development mode
        logger.info(f"Running in development mode on {HOST}:{PORT}")
        app.run(host=HOST, port=PORT, debug=DEBUG)