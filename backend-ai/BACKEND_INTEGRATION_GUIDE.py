# ════════════════════════════════════════════════════════════════════════
#  AI Backend Integration Guide — For .NET Backend Team
# ════════════════════════════════════════════════════════════════════════
#
#  This file documents all AI service APIs available for the .NET backend
#  to consume. The AI backend runs as a separate FastAPI service.
#
#  BASE URL:  http://localhost:8000
#  PREFIX:    /api/v1
#  FULL:      http://localhost:8000/api/v1/{endpoint}
#
# ════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
#  1. HEALTH CHECK — Verify AI service is running
# ──────────────────────────────────────────────────────────────────────
#
#  GET /health
#
#  Response:
#  {
#      "status": "ok",
#      "service": "Mentorship Platform AI Assistant"
#  }
#
#  USE IN .NET:
#  Call this endpoint on startup to verify the AI backend is reachable.
#  If it fails, fallback to non-AI recommendations.


# ──────────────────────────────────────────────────────────────────────
#  2. MENTOR RECOMMENDATIONS — Get top 20 mentor recommendations
# ──────────────────────────────────────────────────────────────────────
#
#  GET /api/v1/recommend?user_id={userId}
#
#  Parameters:
#    - user_id (string, required): The mentee's GUID from the Users table
#
#  Response:
#  {
#      "recommendations": [
#          {
#              "mentor_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
#              "mentor_name": "Ahmed Ali",
#              "domain": "Software Engineering",
#              "score": 87.5,            // similarity score 0-100
#              "reason": "Skills match: React, Node.js | Same subdomain | High rating"
#          },
#          // ... up to 20 items
#      ]
#  }
#
#  RANKING PRIORITY (how the AI ranks mentors):
#    1. Skill Overlap    — Mentor's skills match mentee's interests
#    2. Subdomain Match  — Same specialization (e.g., "Web Development")
#    3. Following Signal — Mentee follows this mentor (intent signal)
#    4. Domain Match     — Same broad domain (e.g., "IT")
#    5. Quality/Popularity — Mentor's rating and engagement
#
#  IMPORTANT FOR FRONTEND:
#    - The API returns up to 20 recommendations sorted by relevance
#    - Frontend should paginate: show 5 at a time (page 1: items 1-5, page 2: 6-10, etc.)
#    - Each item includes a "reason" field for displaying "Why this mentor?"
#
#  USE IN .NET BACKEND:
#
#  // In your RecommendationController or Service:
#  public async Task<List<MentorRecommendation>> GetRecommendationsAsync(string userId)
#  {
#      var client = _httpClientFactory.CreateClient("AIBackend");
#      var response = await client.GetAsync($"/api/v1/recommend?user_id={userId}");
#      response.EnsureSuccessStatusCode();
#      var result = await response.Content.ReadFromJsonAsync<RecommendationResponse>();
#      return result.Recommendations;  // List of 20 items
#  }
#
#  // In your .NET model:
#  public class MentorRecommendation
#  {
#      public string MentorId { get; set; }
#      public string MentorName { get; set; }
#      public string Domain { get; set; }
#      public double Score { get; set; }       // 0-100
#      public string Reason { get; set; }      // "Skills match: ..."
#  }
#
#  // Frontend pagination (React/Angular):
#  // GET /api/recommendations?userId=xxx&page=1&pageSize=5
#  // Your .NET controller fetches all 20 from AI, then paginates for the frontend.


# ──────────────────────────────────────────────────────────────────────
#  3. CHATBOT — Conversational AI assistant
# ──────────────────────────────────────────────────────────────────────
#
#  POST /api/v1/chat
#
#  Request body:
#  {
#      "user_id": "3fa85f64-...",     // optional, for personalized responses
#      "message": "أريد مرشد في React",
#      "language": "ar"               // optional, auto-detected if null
#  }
#
#  Response:
#  {
#      "language": "ar",
#      "intent": "find_mentor",        // classified intent
#      "response_type": "text",        // "text" | "recommendation" | "materials" | "roadmap" | "stats"
#      "answer": "لقد وجدت هؤلاء المرشدين المتخصصين في React...",
#      "recommendations": [],          // filled when intent = "ask_mentor_recommendation"
#      "materials": [],                // filled when intent = "roadmap_request" or "materials"
#      "stats": []                     // filled when intent = "stats"
#  }
#
#  SUPPORTED INTENTS:
#    - "greeting"                → Welcome message
#    - "find_mentor"             → Search for mentors by criteria (uses DB)
#    - "ask_mentor_recommendation" → AI-powered personalized recommendations
#    - "task_help"               → Help with assignments/tasks
#    - "submit_task"             → Task submission guidance
#    - "roadmap_request"         → Learning roadmap generation
#    - "faq"                     → Platform FAQ answers
#    - "general_question"        → General Q&A (with guardrails)
#    - "off_topic"               → Rejected — not related to mentoring
#
#  USE IN .NET BACKEND:
#
#  // In your ChatController:
#  [HttpPost("chat")]
#  public async Task<IActionResult> Chat([FromBody] ChatRequest request)
#  {
#      var client = _httpClientFactory.CreateClient("AIBackend");
#      var response = await client.PostAsJsonAsync("/api/v1/chat", new
#      {
#          user_id = request.UserId,
#          message = request.Message,
#          language = request.Language    // "ar" or "en", or null for auto-detect
#      });
#      var result = await response.Content.ReadFromJsonAsync<ChatResponse>();
#      return Ok(result);
#  }
#
#  // Response types determine what the frontend renders:
#  //   "text"           → Plain text answer (show in chat bubble)
#  //   "recommendation" → Mentor cards (show carousel of mentor profiles)
#  //   "materials"      → Learning resources (show link cards)
#  //   "roadmap"        → Roadmap + materials (show timeline + link cards)
#  //   "stats"          → Dashboard cards (show stat cards)


# ──────────────────────────────────────────────────────────────────────
#  4. SENTIMENT ANALYSIS — Analyze feedback text sentiment
# ──────────────────────────────────────────────────────────────────────
#
#  4a. Single prediction
#  POST /api/v1/sentiment/predict
#
#  Request:
#  { "text": "المرشد ممتاز وساعدني كتير" }
#
#  Response:
#  {
#      "label": "positive",         // "positive" | "neutral" | "negative"
#      "confidence": 0.9823,        // 0.0 to 1.0
#      "scores": {
#          "negative": 0.0052,
#          "neutral": 0.0125,
#          "positive": 0.9823
#      }
#  }
#
#  ──────────────────────────────────────────────────────
#  4b. Batch prediction (up to 32 texts)
#  POST /api/v1/sentiment/predict-batch
#
#  Request:
#  { "texts": ["Great mentor!", "Average experience", "Not helpful at all"] }
#
#  Response:
#  {
#      "results": [
#          { "label": "positive", "confidence": 0.95, "scores": {...} },
#          { "label": "neutral",  "confidence": 0.78, "scores": {...} },
#          { "label": "negative", "confidence": 0.91, "scores": {...} }
#      ],
#      "count": 3
#  }
#
#  ──────────────────────────────────────────────────────
#  4c. Mentor feedback summary
#  GET /api/v1/sentiment/mentor-summary/{mentorId}
#
#  Response:
#  {
#      "mentor_id": "ABC-123",
#      "mentor_name": "Ahmed Ali",
#      "satisfaction_rate": 87.5,       // percentage of positive feedbacks
#      "average_rating": 4.3,           // star rating average (0-5)
#      "breakdown": {
#          "positive": 14,
#          "neutral": 2,
#          "negative": 0,
#          "total": 16
#      },
#      "summary": "المتدربون أجمعوا على أن المرشد ممتاز في الشرح",
#      "top_positive_themes": ["شرح ممتاز", "تواصل فعال"],
#      "top_negative_themes": []
#  }
#
#  USE IN .NET BACKEND:
#
#  // Call when user submits feedback (in FeedbackService):
#  public async Task<SentimentResult> AnalyzeFeedbackAsync(string feedbackText)
#  {
#      var client = _httpClientFactory.CreateClient("AIBackend");
#      var response = await client.PostAsJsonAsync("/api/v1/sentiment/predict", new
#      {
#          text = feedbackText
#      });
#      var result = await response.Content.ReadFromJsonAsync<SentimentResult>();
#
#      // Store the sentiment in the Feedbacks table:
#      // feedback.SentimentLabel = result.Label;         // "positive"
#      // feedback.SentimentConfidence = result.Confidence; // 0.98
#      return result;
#  }
#
#  // Call for mentor profile page (show satisfaction rate):
#  public async Task<MentorSummary> GetMentorSummaryAsync(string mentorId)
#  {
#      var client = _httpClientFactory.CreateClient("AIBackend");
#      var response = await client.GetAsync($"/api/v1/sentiment/mentor-summary/{mentorId}");
#      return await response.Content.ReadFromJsonAsync<MentorSummary>();
#  }


# ──────────────────────────────────────────────────────────────────────
#  5. .NET BACKEND SETUP — How to register the AI HttpClient
# ──────────────────────────────────────────────────────────────────────
#
#  // In Program.cs or Startup.cs:
#  builder.Services.AddHttpClient("AIBackend", client =>
#  {
#      client.BaseAddress = new Uri("http://localhost:8000");
#      client.Timeout = TimeSpan.FromSeconds(30);
#      client.DefaultRequestHeaders.Add("Accept", "application/json");
#  });
#
#  // For production, replace localhost with the deployed AI service URL.
#  // Environment variable: AI_BACKEND_URL=http://ai-service:8000


# ──────────────────────────────────────────────────────────────────────
#  6. FRONTEND PAGINATION — Recommendations (5 per page)
# ──────────────────────────────────────────────────────────────────────
#
#  The AI API returns TOP 20 recommendations in one call.
#  The .NET backend should cache these and paginate for the frontend:
#
#  // .NET Controller:
#  [HttpGet("recommendations")]
#  public async Task<IActionResult> GetRecommendations(
#      [FromQuery] string userId,
#      [FromQuery] int page = 1,
#      [FromQuery] int pageSize = 5)
#  {
#      // 1. Check cache first
#      var cacheKey = $"recommendations:{userId}";
#      var allRecs = await _cache.GetOrCreateAsync(cacheKey, async entry =>
#      {
#          entry.SlidingExpiration = TimeSpan.FromMinutes(15);
#          var client = _httpClientFactory.CreateClient("AIBackend");
#          var resp = await client.GetAsync($"/api/v1/recommend?user_id={userId}");
#          var result = await resp.Content.ReadFromJsonAsync<RecommendationResponse>();
#          return result.Recommendations;  // 20 items
#      });
#
#      // 2. Paginate for frontend
#      var paged = allRecs
#          .Skip((page - 1) * pageSize)
#          .Take(pageSize)
#          .ToList();
#
#      return Ok(new
#      {
#          recommendations = paged,
#          pagination = new
#          {
#              page,
#              pageSize,
#              totalItems = allRecs.Count,
#              totalPages = (int)Math.Ceiling(allRecs.Count / (double)pageSize)
#          }
#      });
#  }
#
#  // Frontend (React/Angular) calls:
#  // GET /api/recommendations?userId=xxx&page=1&pageSize=5  → items 1-5
#  // GET /api/recommendations?userId=xxx&page=2&pageSize=5  → items 6-10
#  // GET /api/recommendations?userId=xxx&page=3&pageSize=5  → items 11-15
#  // GET /api/recommendations?userId=xxx&page=4&pageSize=5  → items 16-20


# ──────────────────────────────────────────────────────────────────────
#  7. COMPLETE API SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────
#
#  | Method | Endpoint                              | Purpose                      | Request                      | Response Key Fields                    |
#  |--------|---------------------------------------|------------------------------|------------------------------|----------------------------------------|
#  | GET    | /health                               | Health check                 | —                            | status, service                        |
#  | GET    | /api/v1/recommend?user_id=X            | Mentor recommendations (20)  | user_id (query)              | recommendations[{mentor_id, score...}] |
#  | POST   | /api/v1/chat                          | Chatbot conversation         | {user_id, message, language} | {intent, answer, recommendations...}   |
#  | POST   | /api/v1/sentiment/predict             | Single sentiment             | {text}                       | {label, confidence, scores}            |
#  | POST   | /api/v1/sentiment/predict-batch       | Batch sentiment (max 32)     | {texts: [...]}               | {results: [...], count}                |
#  | GET    | /api/v1/sentiment/mentor-summary/{id} | Mentor feedback summary      | mentor_id (path)             | {satisfaction_rate, breakdown...}      |
#  | GET    | /api/v1/sentiment/health              | Sentiment model health       | —                            | {model_loaded, ...}                    |
#  | GET    | /db-health                            | Database connectivity check  | —                            | {connected, tables, missing_tables}    |


# ──────────────────────────────────────────────────────────────────────
#  8. RUNNING THE AI BACKEND
# ──────────────────────────────────────────────────────────────────────
#
#  Prerequisites:
#    - Python 3.10+
#    - pip install -r backend-ai/requirements.txt
#    - SQL Server with MentorshipPlatformDB populated
#    - .env file with GROQ_API_KEY (for chatbot LLM)
#
#  Start command:
#    cd backend-ai
#    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
#  The API docs are auto-generated at:
#    http://localhost:8000/docs    (Swagger UI)
#    http://localhost:8000/redoc   (ReDoc)
