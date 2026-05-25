"""
📋 INTEGRATION GUIDE — How to Use Error Handling in Services

This file shows how to integrate the error handling & fallback mechanisms
into your existing services (recommendation, chatbot, sentiment).
"""

# ═══════════════════════════════════════════════════════════════════
# 1️⃣ SENTIMENT SERVICE INTEGRATION
# ═══════════════════════════════════════════════════════════════════

"""
IN: backend-ai/services/sentiment_service.py

# ADD IMPORT
from services.error_handling import SentimentErrorHandler

# IN THE predict() METHOD, wrap with error handling:

def predict(self, text: str) -> SentimentResult:
    text = self._validate_input(text)
    if not text:
        return SentimentResult(
            label="neutral", confidence=1.0,
            scores={"positive": 0.0, "neutral": 1.0, "negative": 0.0}
        )

    cache_key = text.lower()
    cached = self._get_cached(cache_key)
    if cached is not None:
        return cached

    self._ensure_loaded()

    # Try BERT with timeout monitoring
    if _HAS_TORCH and self._model is not None:
        try:
            t0 = time.perf_counter()
            result = self._predict_bert(text)
            elapsed = (time.perf_counter() - t0) * 1000
            
            # ✅ ADDED: Check if slow and handle
            if elapsed > 2000:  # > 2 seconds = slow
                logger.warning(f"Sentiment slow: {elapsed:.0f}ms")
                return SentimentErrorHandler.handle_slow_sentiment(
                    text, elapsed, threshold_ms=2000
                )
            
            self._set_cached(cache_key, result)
            return result
        except Exception as e:
            # ✅ ADDED: Handle BERT failure with Groq fallback
            logger.error(f"BERT prediction failed: {e}")
            result = self._groq_fallback([text])[0]
            self._set_cached(cache_key, result)
            return result
    
    # Try Groq with error handling
    try:
        result = self._groq_fallback([text])[0]
        self._set_cached(cache_key, result)
        return result
    except Exception as e:
        # ✅ ADDED: Final fallback when everything fails
        logger.error(f"Sentiment prediction completely failed: {e}")
        return SentimentErrorHandler.handle_failed_sentiment(
            e, language="ar" if "ar" in text else "en"
        )
"""

# ═══════════════════════════════════════════════════════════════════
# 2️⃣ CHATBOT / INTENT SERVICE INTEGRATION
# ═══════════════════════════════════════════════════════════════════

"""
IN: backend-ai/services/intent_service.py

# ADD IMPORT
from services.error_handling import ChatbotErrorHandler

# IN THE detect_intent_async() or detect_intent() METHOD:

async def detect_intent_async(text: str, user_id: str = "", language: str = "en") -> dict:
    \"\"\"Detect intent with error handling and fallback.\"\"\"
    
    if not text or not text.strip():
        # ✅ ADDED: Handle empty input
        return ChatbotErrorHandler.handle_unclear_intent(
            "", language, confidence=0.0
        )
    
    try:
        # Try LLM classification
        intent = await self._llm_classify(text, language)
        
        if not intent or intent not in VALID_INTENTS:
            # ✅ ADDED: Handle unclear/invalid intent
            return ChatbotErrorHandler.handle_unclear_intent(
                text, language, confidence=0.3
            )
        
        return {
            "intent": intent,
            "language": language,
            "confidence": 0.9,
            "status": "success",
        }
    
    except RateLimitError:
        # ✅ ADDED: Handle rate limit
        logger.warning(f"Rate limit hit for user {user_id}")
        return ChatbotErrorHandler.handle_rate_limit(language)
    
    except LLMTimeout:
        # ✅ ADDED: Fallback to keyword classification
        logger.warning(f"LLM timeout, using keyword fallback")
        intent = _keyword_fallback(text)
        return {
            "intent": intent,
            "language": language,
            "confidence": 0.6,  # Lower confidence = fallback
            "source": "keyword_fallback",
        }
    
    except Exception as e:
        # ✅ ADDED: Handle complete service failure
        logger.error(f"Intent classification failed: {e}")
        return ChatbotErrorHandler.handle_intent_service_error(e, language)
"""

# ═══════════════════════════════════════════════════════════════════
# 3️⃣ RECOMMENDATION SERVICE INTEGRATION
# ═══════════════════════════════════════════════════════════════════

"""
IN: backend-ai/services/recommendation_service.py

# ADD IMPORT
from services.error_handling import RecommendationErrorHandler

# IN THE get_recommendations() METHOD:

async def get_recommendations(self, user_id: int | str) -> list[dict]:
    \"\"\"Get recommendations with database fallback.\"\"\"
    
    mode = settings.RECOMMENDER_MODE.lower().strip()
    results = []
    
    try:
        if mode == "model":
            results = self._from_model(user_id)
        elif mode == "api":
            results = await self._from_api(user_id)
            if not results:
                results = self._from_db(user_id)
        else:
            results = self._from_db(user_id)
    
    except ModelTimeoutError:
        # ✅ ADDED: Model timeout → fallback to DB
        logger.warning(f"Model timeout for user {user_id}, using DB fallback")
        results = RecommendationErrorHandler.get_database_fallback_recommendations(
            str(user_id), limit=10
        )
    
    except ValueError as e:
        # ✅ ADDED: Model validation error
        logger.error(f"Model validation failed: {e}")
        results = RecommendationErrorHandler.get_database_fallback_recommendations(
            str(user_id), limit=10
        )
    
    except DatabaseAccessError as e:
        # ✅ ADDED: Database error → return empty with guidance
        logger.error(f"Database error: {e}")
        return RecommendationErrorHandler.handle_recommendation_failure(
            str(user_id), e, language="en"
        )
    
    except Exception as e:
        # ✅ ADDED: Unknown error → database fallback
        logger.error(f"Recommendation failed: {e}")
        return RecommendationErrorHandler.handle_recommendation_failure(
            str(user_id), e, language="en"
        )
    
    if results:
        _recommendation_memory.set(str(user_id), results)
    
    return results
"""

# ═══════════════════════════════════════════════════════════════════
# 4️⃣ ROUTE-LEVEL ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════

"""
IN: backend-ai/routes/recommend.py

# ALREADY HAS ERROR HANDLING, but enhance with fallback:

@router.get("/recommend")
async def recommend(user_id: str = Query(..., description="User UUID")) -> dict:
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id required")
    
    user_id = user_id.strip()
    
    try:
        recommendations = await recommendation_service.get_recommendations(user_id=user_id)
        return {"recommendations": [RecommendationItem(**item).model_dump() for item in recommendations]}
    
    except HTTPException:
        raise  # Re-raise HTTP errors
    
    except Exception as exc:
        logger.exception(f"Recommendation failed for {user_id}")
        
        # ✅ TRY DATABASE FALLBACK ONE MORE TIME
        from services.error_handling import RecommendationErrorHandler
        fallback = RecommendationErrorHandler.get_database_fallback_recommendations(
            user_id, limit=5
        )
        
        if fallback:
            return {
                "recommendations": fallback,
                "status": "fallback_mode",
                "error": "Model unavailable, showing top available mentors"
            }
        
        # ✅ IF FALLBACK ALSO FAILS, SHOW ERROR
        return {
            "recommendations": [],
            "error": "Failed to load recommendations",
            "status": "service_unavailable",
            "contact": "support@mentora.platform"
        }
"""

# ═══════════════════════════════════════════════════════════════════
# 5️⃣ TESTING THE EDGE CASES & FALLBACKS
# ═══════════════════════════════════════════════════════════════════

"""
Verify all edge cases are handled by running these test commands:

# SENTIMENT EDGE CASES (from tests/test_sentiment_live.py)
pytest tests/test_sentiment_live.py::TestSentimentEdgeCases -v -s
pytest tests/test_sentiment_live.py::TestSentimentPerformance::test_single_prediction_latency -v -s

# RECOMMENDATION EDGE CASES
pytest tests/test_recommendations_live.py::TestRecommendationEdgeCases -v -s
pytest tests/test_recommendations_live.py::TestRecommendationPerformance::test_latency_within_slo -v -s

# CHATBOT EDGE CASES
pytest tests/test_chatbot_live.py::TestMoreEdgeCases -v -s
pytest tests/test_chatbot_live.py::TestChatbotConsistency -v -s

# RUN CACHING VALIDATION SCRIPT
python check_caching.py
"""

# ═══════════════════════════════════════════════════════════════════
# 6️⃣ FALLBACK PRIORITY CHAIN
# ═══════════════════════════════════════════════════════════════════

"""
SENTIMENT ANALYSIS — Fallback Chain:
    1️⃣ BERT (Primary) — Fast, fine-tuned for Arabic
       ↓ [if SLOW > 2s]
    2️⃣ Groq LLM — More reliable, slower
       ↓ [if FAILS]
    3️⃣ Keyword Fallback — Pattern matching, always works
       ↓ [if ERROR]
    4️⃣ Neutral Default — Safe default when nothing works

CHATBOT / INTENT — Fallback Chain:
    1️⃣ Groq LLM (Primary) — Fast model (llama-3.1-8b-instant)
       ↓ [if RATE_LIMITED or TIMEOUT]
    2️⃣ Keyword Fallback — Pattern matching, local
       ↓ [if FAILS]
    3️⃣ General Question — Safe default, ask for clarification
       ↓ [if ERROR]
    4️⃣ Support Email — Last resort, show contact info

RECOMMENDATION — Fallback Chain:
    1️⃣ Model Inference (Primary) — LightGBM trained model
       ↓ [if SLOW or ERROR]
    2️⃣ API Call — External recommender service
       ↓ [if FAILS]
    3️⃣ Database Query — Direct SQL, top mentors by rating
       ↓ [if ALL FAIL]
    4️⃣ Error Message — Show support contact info
"""

# ═══════════════════════════════════════════════════════════════════
# 7️⃣ MONITORING & ALERTING
# ═══════════════════════════════════════════════════════════════════

"""
Add these logging/monitoring points:

# In error_handling.py or services:

logger.warning(f"Sentiment analysis slow: {elapsed:.0f}ms (threshold: 2000ms)")
# → Alert if > 10% of requests are slow

logger.error(f"Sentiment prediction completely failed: {error}")
# → Alert if > 5% of requests fail

logger.warning(f"Intent classification using keyword fallback")
# → Track fallback usage — if > 20% using fallback, investigate

logger.warning(f"Recommendation model timeout for user {user_id}")
# → Track model failures — if > 5% fail, check model/system

logger.info(f"Using database fallback for recommendations")
# → Monitor when we fallback to database
"""

print("""
═══════════════════════════════════════════════════════════════════
✅ INTEGRATION GUIDE COMPLETE

1. Copy error handling code into your services
2. Update exception handling in each service
3. Test edge cases: pytest tests/ -m live -k "EdgeCase"
4. Monitor fallback usage in logs
5. Verify SLOs: pytest tests/ --durations=10 -m live

═══════════════════════════════════════════════════════════════════
""")
