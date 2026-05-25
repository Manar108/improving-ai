# Mentor Chatbot Architecture - Final Implementation

## Overview

The **Mentor Chatbot** is a lightweight, role-aware chatbot system built on the existing mentee chatbot infrastructure. It provides mentors with practical assistance for:
- Platform FAQs and how-to guidance
- Teaching material suggestions and resources
- Analytics and mentorship statistics
- Workflow help (managing mentees, applications, communications)
- Document upload and Q&A (Phase 2)

The implementation reuses ~80% of existing infrastructure (LLM services, intent classification, routing, caching, language detection, history handling) while adding mentor-specific intents, context loading, and analytics.

---

## Architecture Components

### 1. **Role-Aware Entry Point**

**Endpoint:** `POST /api/v1/mentor-chat` (new)

The chatbot now routes based on user role:
```python
# In routes/mentor_chat.py
- Detect language (Arabic/English)
- Load user context from DB
- VERIFY user is a mentor (role == "mentor" or "both")
- Classify intent using MENTOR-SPECIFIC classifier
- Route to appropriate handler
- Return structured response
```

**Security:** Non-mentors attempting to use `/mentor-chat` receive a friendly rejection.

---

### 2. **Intent Classification - Mentor-Specific**

**File:** `services/mentor_intent_service.py`

**Supported Intents (8 total):**

| Intent | Purpose | Examples |
|--------|---------|----------|
| `greeting` | Social greeting | "Hi", "مرحبا", "ازيك" |
| `faq` | Platform how-to questions | "How to create a program?", "ازاي أنشئ برنامج" |
| `materials_request` | Request teaching resources | "Interview questions", "تمارين للمبتدئين" |
| `mentor_analytics` | Statistics and engagement | "How many mentees?", "كام منتي في البرنامج" |
| `mentor_workflow_help` | Managing mentees and sessions | "How to contact mentees?", "ازاي أتواصل" |
| `ask_document` | Questions about uploaded files | "Summarize this PDF", "اعمل امتحان من الـ slides" |
| `general_question` | General learning Q&A | "Explain OOP", "ايه الفرق بين..." |
| `off_topic` | Unrelated/profanity | Jokes, weather, abuse |

**Mechanism:**
1. **LLM-primary** (llama-3.1-8b via Groq)
2. **Cache TTL:** 5 minutes (deduplicates repeated questions)
3. **Keyword fallback:** If API unavailable
4. **Default fallback:** `general_question`

**Key Difference from Mentee:**
- Mentee intents: recommendation-heavy (find_mentor, ask_mentor_recommendation, etc.)
- Mentor intents: administrative and guidance-heavy (analytics, workflow, materials)

---

### 3. **Mentor Context Loading**

**File:** `services/mentor_context_service.py`

Loads mentor-specific data from database without throwing exceptions:

**Methods:**
```python
get_mentor_context(mentor_id)
├── Returns: user_id, first_name, last_name, domain_name, 
│           years_of_experience, bio, is_verified, 
│           average_rating, total_reviews, program_count

get_mentor_programs(mentor_id, limit=10)
├── Returns: program_id, title, domain_name, capacity,
│           active_mentees, applications_count, created_at

get_mentor_active_mentees(mentor_id, limit=20)
├── Returns: mentee_id, first_name, last_name, domain_name,
│           start_date, status

get_mentor_pending_applications(mentor_id, limit=20)
├── Returns: application_id, program_id, program_title,
│           mentee_id, mentee_name, status, applied_at
```

**DB Queries:**
- SQL Server direct queries via `database.run_query_df()`
- All methods return safe defaults on failure (empty lists/dicts)
- No exceptions thrown - follows existing user_context_service pattern

---

### 4. **Mentor Response Generation**

**File:** `services/mentor_response_service.py`

Generates mentor-specific responses for each intent:

| Intent | Handler | Data Source | Output |
|--------|---------|-------------|--------|
| `greeting` | `respond_to_greeting()` | Static | Text |
| `faq` | `respond_to_faq()` | RAG (FAQ DB) + LLM fallback | Text |
| `materials_request` | `respond_to_materials_request()` | Search service + LLM | Text + Materials list |
| `mentor_analytics` | `respond_to_mentor_analytics()` | Context service | Text (LLM-summarized stats) |
| `mentor_workflow_help` | `respond_to_mentor_workflow_help()` | Context service + LLM | Text |
| `off_topic` | `respond_to_off_topic()` | Static | Text |
| `general_question` | `respond_to_general_question()` | LLM service | Text |

**Key Pattern - Reuse Existing Services:**
- LLM service: `chat_with_system_prompt()` (NEW generic method added)
- RAG service: `answer_platform_question()`, `find_materials()` (reused)
- Search service: Material search (reused)
- Context service: Load data from DB (reused)

---

### 5. **Document Handling** (Phase 1 MVP)

**File:** `services/mentor_document_service.py`

Supports upload and basic text extraction for future Q&A:

**Supported Formats:** PDF, DOCX, PPTX, TXT
**Max File Size:** 10 MB

**Methods:**
```python
validate_file(file_name, file_size)
├── Returns: (is_valid, error_message)

save_uploaded_file(mentor_id, file_name, file_content)
├── Returns: (success, message, file_path)
├── Saves to: data/uploads/{mentor_id}/{timestamp}_{filename}

process_document(file_path)
├── Extract text via: PDF, DOCX, PPTX, TXT parsers
├── Chunk text with 1000 char chunks + 100 char overlap
├── Returns: (success, message, chunks)

get_document_summary(text, max_length=500)
├── Returns: truncated text preview
```

**Phase 1 MVP:** Storage + extraction only
**Phase 2:** Database storage + Q&A via RAG

---

### 6. **Analytics APIs**

**File:** `routes/mentor_analytics.py`

RESTful endpoints for mentor dashboard/UI:

```
GET /api/v1/mentor/analytics/profile/{mentor_id}
├── Returns: MentorProfileDto (profile info + stats)

GET /api/v1/mentor/analytics/programs/{mentor_id}?limit=10
├── Returns: list[ProgramStatsDto] (programs with mentee counts, applications)

GET /api/v1/mentor/analytics/mentees/{mentor_id}?limit=20
├── Returns: list[MenteeActivityDto] (active mentees)

GET /api/v1/mentor/analytics/applications/{mentor_id}?limit=20
├── Returns: list of pending applications for mentor's programs

GET /api/v1/mentor/analytics/overview/{mentor_id}
├── Returns: AnalyticsOverviewDto (complete dashboard data)
```

**Data Models:**
- `MentorProfileDto`: user_id, name, domain, experience, rating, verified, program_count
- `ProgramStatsDto`: program details + active_mentees, applications_count
- `MenteeActivityDto`: mentee info + status, start_date
- `AnalyticsOverviewDto`: aggregated metrics for dashboard

---

### 7. **Document Upload Routes**

**File:** `routes/mentor_documents.py`

Upload and retrieve documents:

```
POST /api/v1/mentor/documents/upload/{mentor_id}
├── File upload (PDF, DOCX, PPTX, TXT)
├── Returns: DocumentUploadResponseDto (file_name, size, text_preview)

GET /api/v1/mentor/documents/list/{mentor_id}?limit=10
├── Phase 2: List mentor's documents

GET /api/v1/mentor/documents/info/{document_id}
├── Phase 2: Get document metadata + full text

GET /api/v1/mentor/documents/formats
├── Supported formats + size limits
```

---

### 8. **Main Chatbot Route**

**File:** `routes/mentor_chat.py`

Single endpoint handles all mentor queries:

```python
@router.post("/mentor-chat", response_model=ChatResponse)
async def mentor_chat(payload: ChatRequest) -> ChatResponse:
    # 1. Detect language
    # 2. Classify intent (MENTOR-SPECIFIC)
    # 3. Load mentor context
    # 4. Verify user is a mentor
    # 5. Route to handler
    # 6. Return ChatResponse
```

**Request Schema:** (reused from mentee)
```json
{
  "message": "How many mentees do I have?",
  "user_id": "uuid-string",
  "language": "ar" | "en",  // auto-detected if omitted
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response Schema:** (reused from mentee)
```json
{
  "language": "ar",
  "intent": "mentor_analytics",
  "response_type": "text" | "materials" | "recommendation" | "roadmap",
  "answer": "string",
  "materials": [  // optional
    {
      "title": "...",
      "url": "...",
      "kind": "videos|courses|articles|docs|projects",
      "source": "..."
    }
  ],
  "recommendations": []  // unused for mentor
}
```

---

## Integration with Existing Infrastructure

### Reused Services:
```python
# LLM Service
await llm_service.chat_with_system_prompt(message, language, system_prompt)
await llm_service.chat_general(message, language, history)
await llm_service.chat_off_topic(language)

# RAG Service
await rag_service.answer_platform_question(message, language, intent="faq")
await rag_service.find_materials(query, language)

# Search Service
await search_service.find_materials(query, language)

# User Context Service
user_context_service.get_user_context(user_id)  # Returns role, domain, etc.

# Intent Service (Mentee - not used for mentor)
# Language detection reused via intent_service.detect_language()
```

### NOT Reused:
- Recommendation services (mentor-specific analytics instead)
- Program recommendation service (different logic)
- Mentee-specific prompts and handlers

---

## Flow Diagrams

### Message Flow
```
User (Mentor) sends message
    ↓
POST /api/v1/mentor-chat
    ↓
Detect Language (AR/EN)
    ↓
Classify Intent (8 mentor intents)
    ↓
Load Mentor Context from DB
    ↓
Verify User is Mentor
    ↓
Route to Handler:
├─ greeting → Static response
├─ faq → RAG + LLM
├─ materials_request → Search + Materials
├─ mentor_analytics → DB + LLM summary
├─ mentor_workflow_help → DB context + LLM
├─ ask_document → Placeholder (Phase 2)
├─ general_question → LLM
└─ off_topic → Rejection
    ↓
Return ChatResponse (JSON)
```

### Analytics Flow
```
Frontend requests mentor dashboard
    ↓
GET /api/v1/mentor/analytics/overview/{mentor_id}
    ↓
Load from DB:
├─ mentor_context (profile, rating, experience)
├─ mentor_programs (active programs, mentee counts)
├─ mentor_active_mentees (list of mentees)
├─ mentor_pending_applications (applications to review)
    ↓
Aggregate metrics:
├─ active_mentees_count
├─ pending_applications_count
├─ avg_mentees_per_program
    ↓
Return AnalyticsOverviewDto (JSON)
```

---

## Mentor Intent Definitions (Detailed)

### 1. **greeting**
- User is saying hello or opening conversation
- Examples: "hi", "مرحبا", "ازيك"
- Response: Mentor-specific greeting with available features

### 2. **faq**
- Questions about platform features, how to use platform
- Key signals: "how to", "إزاي", "feature", "قاعدة"
- Examples: "How to create a program?", "ازاي أنشئ برنامج"
- Handler: RAG (FAQ database) + LLM fallback

### 3. **materials_request**
- Mentor asks for teaching resources to give to mentees
- Key signals: "exercise", "تمرين", "interview", "مقابلة", "quiz", "project", "فكرة"
- Examples: "Interview questions for backend", "تمارين للمبتدئين"
- Handler: Search service + materials list

### 4. **mentor_analytics**
- Mentor wants statistics about programs/mentees
- Key signals: "how many", "كام", "statistic", "analytics", "performance"
- Examples: "How many mentees in my program?", "Top performing program?"
- Handler: Load from DB + LLM summarization

### 5. **mentor_workflow_help**
- Help managing mentorships (contact, feedback, sessions, applications)
- Key signals: "contact", "feedback", "session", "mentee", "manage"
- Examples: "How to give feedback?", "ازاي أتواصل مع المنتيز"
- Handler: Load context from DB + LLM guidance

### 6. **ask_document**
- Questions about uploaded lecture notes, slides, assignments
- Key signals: "document", "pdf", "file", "slide", "lecture"
- Examples: "Summarize this PDF", "اعمل امتحان من الـ slides"
- Handler: PLACEHOLDER for Phase 2

### 7. **general_question**
- General educational Q&A (mentor's own learning)
- Examples: "Explain OOP", "ايه الفرق بين Python و Java"
- Handler: LLM (reused chat_general)

### 8. **off_topic**
- Unrelated topics, profanity, jokes
- Handler: Polite rejection

---

## Database Schema (For Document Storage - Phase 2)

```sql
CREATE TABLE mentor_documents (
    document_id INT PRIMARY KEY IDENTITY(1,1),
    mentor_id UNIQUEIDENTIFIER NOT NULL,
    file_name NVARCHAR(255) NOT NULL,
    file_type NVARCHAR(20) NOT NULL,      -- pdf, docx, txt, pptx
    file_path NVARCHAR(500) NOT NULL,
    file_size INT NOT NULL,
    extracted_text NVARCHAR(MAX),
    uploaded_at DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (mentor_id) REFERENCES mentor_profile(user_id)
);

CREATE TABLE mentor_document_chunks (
    chunk_id INT PRIMARY KEY IDENTITY(1,1),
    document_id INT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text NVARCHAR(MAX) NOT NULL,
    FOREIGN KEY (document_id) REFERENCES mentor_documents(document_id)
);
```

---

## Configuration & Deployment

### Environment Variables (existing .env)
```
GROQ_API_KEY=...          # Used by intent classifier
MODEL_NAME=...            # LLM model (llama-3.3-70b-versatile)
DB_SERVER=...             # SQL Server connection
DB_DATABASE=...           # MentorshipPlatformDB
DB_TRUSTED_CONNECTION=... # True/False
```

### New Directories Created
```
backend-ai/
├── data/
│   └── uploads/          # Document upload directory
├── services/
│   ├── mentor_intent_service.py
│   ├── mentor_context_service.py
│   ├── mentor_response_service.py
│   ├── mentor_document_service.py
│   └── ...existing...
├── routes/
│   ├── mentor_chat.py
│   ├── mentor_analytics.py
│   ├── mentor_documents.py
│   └── ...existing...
└── ...existing...
```

---

## Testing Checklist

### Unit Tests (Ready to Implement)
```python
# mentor_intent_service.py
- test_detect_language()
- test_detect_intent_async_greeting()
- test_detect_intent_async_faq()
- test_detect_intent_async_analytics()
- test_keyword_fallback()

# mentor_context_service.py
- test_get_mentor_context_valid_id()
- test_get_mentor_context_invalid_id()
- test_get_mentor_programs()
- test_get_mentor_active_mentees()

# mentor_response_service.py
- test_respond_to_greeting()
- test_respond_to_faq()
- test_respond_to_materials_request()
- test_respond_to_mentor_analytics()

# mentor_chat.py (route)
- test_mentor_chat_greeting()
- test_mentor_chat_faq()
- test_mentor_chat_analytics()
- test_mentor_chat_non_mentor_blocked()
```

### Integration Tests
```python
# Full flow tests
- test_mentor_chat_analytics_flow()
- test_mentor_chat_materials_flow()
- test_mentor_chat_off_topic()
- test_mentor_analytics_api_overview()
- test_mentor_analytics_api_programs()
```

### Manual Testing
```
1. Message from mentor → Greeting → Verify mentor-specific response
2. Message from mentor → FAQ ("How to create a program?") → Verify RAG/LLM response
3. Message from mentor → Analytics ("How many mentees?") → Verify stats loaded from DB
4. Message from mentor → Materials ("Interview questions") → Verify materials list returned
5. Document upload → POST /mentor/documents/upload → Verify extraction works
6. Non-mentor → /mentor-chat → Verify rejection
```

---

## API Summary

### Chatbot
```
POST /api/v1/mentor-chat
├── Input: ChatRequest (message, user_id, language, history)
└── Output: ChatResponse (intent, answer, materials)
```

### Analytics
```
GET /api/v1/mentor/analytics/profile/{mentor_id}
GET /api/v1/mentor/analytics/programs/{mentor_id}?limit=10
GET /api/v1/mentor/analytics/mentees/{mentor_id}?limit=20
GET /api/v1/mentor/analytics/applications/{mentor_id}?limit=20
GET /api/v1/mentor/analytics/overview/{mentor_id}
```

### Documents
```
POST /api/v1/mentor/documents/upload/{mentor_id}
    ├── File: PDF, DOCX, PPTX, TXT
    └── Response: DocumentUploadResponseDto

GET /api/v1/mentor/documents/list/{mentor_id}?limit=10          [Phase 2]
GET /api/v1/mentor/documents/info/{document_id}                 [Phase 2]
GET /api/v1/mentor/documents/formats
```

---

## Known Limitations & Phase 2 Items

### Phase 1 (Current - MVP)
✅ Mentor chatbot with 8 intents
✅ FAQ handling (RAG + LLM)
✅ Materials request
✅ Analytics from DB
✅ Workflow help
✅ Document upload & extraction
✅ Analytics APIs
✅ Language support (Arabic/English)

### Phase 2 (Future)
⚠️ Document Q&A (ask questions about uploaded files)
⚠️ Document database storage
⚠️ Vector embedding for semantic search
⚠️ Abstract summarization (currently truncation-based)
⚠️ Quiz/exercise generation from documents
⚠️ Mentor feedback suggestions
⚠️ Program performance analytics

---

## Maintenance & Future Improvements

### Quick Wins
1. Add more FAQ content to knowledge base
2. Improve system prompts based on mentor feedback
3. Add mentor testimonials to FAQ responses
4. Add contextual help widgets

### Medium-term
1. Implement document Q&A (Phase 2)
2. Add mentor ratings/reviews analytics
3. Predictive analytics (which programs underperform)
4. Mentor skill recommendations

### Long-term
1. Multi-turn contextual conversations
2. Mentor-to-mentor knowledge sharing
3. Program recommendation engine
4. Automated feedback templates

---

## Files Modified/Created

### New Files (10)
- `services/mentor_intent_service.py` (220 lines)
- `services/mentor_context_service.py` (230 lines)
- `services/mentor_response_service.py` (320 lines)
- `services/mentor_document_service.py` (250 lines)
- `routes/mentor_chat.py` (280 lines)
- `routes/mentor_analytics.py` (220 lines)
- `routes/mentor_documents.py` (200 lines)

### Modified Files (2)
- `main.py` (added 3 router imports + 3 router includes)
- `services/llm_service.py` (added `chat_with_system_prompt()` method)

### Total New Code
~1500 lines of production code
~0 breaking changes to existing code

---

## Conclusion

The Mentor Chatbot is a **lightweight, practical, and well-integrated** system that:
- Reuses 80% of existing infrastructure
- Adds mentor-specific intents and handlers
- Provides clean APIs for analytics and documents
- Maintains code quality and no breaking changes
- Sets up groundwork for Phase 2 features (document Q&A, advanced analytics)

The system is **ready for integration** with the frontend and can handle realistic mentor workflows today while being extensible for future enhancements.
