# Mentor Chatbot - Quick Start Guide

## Using the Mentor Chatbot

### Basic Flow: Chatbot Message

**Endpoint:** `POST /api/v1/mentor-chat`

**Request Example:**
```json
{
  "message": "How many mentees do I have?",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "language": "ar"
}
```

**Response Example (Analytics Intent):**
```json
{
  "language": "ar",
  "intent": "mentor_analytics",
  "response_type": "text",
  "answer": "لديك حالياً 5 منتيز نشطين في برامجك. البرنامج الأساسي فيه 3 منتيز والبرنامج المتقدم فيه 2 منتي. معدل الانخراط بتاعك جيد جداً! 📊"
}
```

---

## Mentor Chatbot Intents & Examples

### 1. Greeting
**Input:** "أهلا بك"  
**Output:** Mentor-specific welcome message

### 2. FAQ - How to Create a Program
**Input:** "إزاي أنشئ برنامج؟"  
**Output:** Step-by-step guide from FAQ database

### 3. Materials Request - Interview Questions
**Input:** "أعطني أسئلة مقابلات لـ backend"  
**Output:**
```json
{
  "response_type": "materials",
  "answer": "لقد جمعت أسئلة مقابلات مفيدة لتقييم منتيزك",
  "materials": [
    {
      "title": "50 Backend Interview Questions",
      "url": "https://example.com/backend-questions",
      "kind": "articles",
      "source": "dev.to"
    },
    {...}
  ]
}
```

### 4. Analytics - Mentee Count
**Input:** "كام منتي عندي؟"  
**Output:**
```json
{
  "response_type": "text",
  "answer": "أنت حالياً بتدرس 8 منتيز في برامجك:\n- برنامج Python: 4 منتيز\n- برنامج Web Dev: 4 منتيز\nمعدل الإكمال: 75% 📈"
}
```

### 5. Workflow Help - Contact Mentees
**Input:** "إزاي أتواصل مع المنتيز بتوعي؟"  
**Output:**
```json
{
  "response_type": "text",
  "answer": "تقدر تتواصل مع منتيزك بـ عدة طرق:\n1. في صفحة البرنامج الخاص بك، عندك قائمة المنتيز\n2. اضغط على كل منتي لترسل رسالة خاصة\n3. يمكنك إرسال رسائل جماعية قبل كل جلسة\n..."
}
```

---

## Analytics API Usage

### Get Overview (Complete Dashboard Data)
```bash
GET /api/v1/mentor/analytics/overview/{mentor_id}
```

**Response:**
```json
{
  "mentor_profile": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "first_name": "محمد",
    "last_name": "أحمد",
    "domain_name": "Backend Development",
    "years_of_experience": 5,
    "is_verified": true,
    "average_rating": 4.8,
    "total_reviews": 32,
    "program_count": 3
  },
  "programs": [
    {
      "program_id": 1,
      "title": "Python Fundamentals",
      "domain_name": "Backend",
      "capacity": 10,
      "active_mentees": 6,
      "applications_count": 8,
      "created_at": "2026-01-15T00:00:00"
    },
    {...}
  ],
  "active_mentees_count": 15,
  "pending_applications_count": 3,
  "average_mentees_per_program": 5.0
}
```

### Get Programs Only
```bash
GET /api/v1/mentor/analytics/programs/{mentor_id}?limit=10
```

**Response:**
```json
[
  {
    "program_id": 1,
    "title": "Python Fundamentals",
    "domain_name": "Backend",
    "capacity": 10,
    "active_mentees": 6,
    "applications_count": 8,
    "created_at": "2026-01-15T00:00:00"
  },
  {...}
]
```

### Get Mentees
```bash
GET /api/v1/mentor/analytics/mentees/{mentor_id}?limit=20
```

**Response:**
```json
[
  {
    "mentee_id": "660e8400-e29b-41d4-a716-446655440001",
    "first_name": "علي",
    "last_name": "محمود",
    "domain_name": "Backend",
    "start_date": "2026-01-20T00:00:00",
    "status": "Active"
  },
  {...}
]
```

### Get Pending Applications
```bash
GET /api/v1/mentor/analytics/applications/{mentor_id}?limit=20
```

**Response:**
```json
[
  {
    "application_id": 1,
    "program_id": 1,
    "program_title": "Python Fundamentals",
    "mentee_id": "770e8400-e29b-41d4-a716-446655440002",
    "mentee_name": "فاطمة علي",
    "status": "Pending",
    "applied_at": "2026-05-15T10:30:00"
  },
  {...}
]
```

---

## Document Upload API

### Upload a Document
```bash
POST /api/v1/mentor/documents/upload/{mentor_id}
Content-Type: multipart/form-data

file: <PDF/DOCX/PPTX/TXT file>
```

**Response:**
```json
{
  "success": true,
  "message": "Document uploaded and processed successfully",
  "file_name": "lecture_01.pdf",
  "file_size": 2048576,
  "extracted_text_preview": "Chapter 1: Introduction to Python\n\nPython is a high-level programming language known for its simplicity and readability..."
}
```

### Get Supported Formats
```bash
GET /api/v1/mentor/documents/formats
```

**Response:**
```json
{
  "supported_formats": [".pdf", ".docx", ".txt", ".pptx"],
  "max_file_size_mb": 10.0
}
```

---

## Frontend Integration Example

### Using Fetch API
```javascript
// Send message to mentor chatbot
async function sendMentorMessage(message, userId) {
  const response = await fetch('/api/v1/mentor-chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: message,
      user_id: userId,
      language: 'ar'
    })
  });
  
  const data = await response.json();
  console.log(data.intent);
  console.log(data.answer);
  
  if (data.materials) {
    // Display materials list
  }
}

// Get mentor analytics
async function getMentorAnalytics(mentorId) {
  const response = await fetch(
    `/api/v1/mentor/analytics/overview/${mentorId}`
  );
  
  const data = await response.json();
  console.log(`Programs: ${data.mentor_profile.program_count}`);
  console.log(`Active Mentees: ${data.active_mentees_count}`);
}

// Upload document
async function uploadDocument(mentorId, file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(
    `/api/v1/mentor/documents/upload/${mentorId}`,
    {
      method: 'POST',
      body: formData
    }
  );
  
  const data = await response.json();
  console.log(data.extracted_text_preview);
}
```

---

## Real-World Mentor Workflows

### Workflow 1: Mentor Checks Daily Statistics
```
1. Mentor opens dashboard
2. Dashboard calls: GET /api/v1/mentor/analytics/overview/{id}
3. Shows: Program count, active mentees, pending applications
4. Mentor can click to view details or applications to review
```

### Workflow 2: Mentor Wants Teaching Materials
```
1. Mentor opens chatbot
2. Types: "تمارين صعبة للمتقدمين في Python"
3. Intent detected: materials_request
4. Returns: List of exercises + articles + projects
5. Mentor can share link with mentees
```

### Workflow 3: Mentor Needs Platform Help
```
1. Mentor opens chatbot
2. Types: "ازاي أراجع واجبات المنتيز؟"
3. Intent detected: faq
4. Returns: Step-by-step guide from FAQ database
5. Mentor follows steps
```

### Workflow 4: Mentor Uploads Lecture Notes
```
1. Mentor opens documents section
2. Uploads: lecture_01.pdf
3. System extracts: text + chunks
4. Returns: Text preview
5. [Phase 2] Mentor can ask: "اعمل امتحان من الـ pdf دي"
```

---

## Error Handling

### Non-Mentor Attempts to Use Mentor Chat
```json
{
  "detail": "This chatbot is for mentors only 📚"
}
```

### Invalid User ID
```json
{
  "language": "ar",
  "intent": "off_topic",
  "answer": "لا يمكن عرض الإحصائيات بدون تسجيل دخول 😊"
}
```

### File Upload Too Large
```json
{
  "detail": "File too large. Max size: 10.0 MB"
}
```

### Unsupported File Type
```json
{
  "detail": "Unsupported file type. Supported: .pdf, .docx, .txt, .pptx"
}
```

---

## Performance Notes

### Caching
- Intent classification results cached for 5 minutes
- LLM responses cached for 5 minutes
- Reduces API calls to Groq by ~70% for repeated questions

### Database Queries
- Analytics queries use `TOP N` with limits
- No N+1 queries
- Direct SQL Server queries (no ORM overhead)
- Typical response time: <500ms

### Chat History
- Trimmed to last 6 messages
- Each message capped at 1000 characters
- Prevents token overflow in LLM

---

## Next Steps

### For Frontend Integration
1. Add mentor chatbot section to dashboard
2. Implement analytics visualization
3. Add document upload UI
4. Style responses (materials list, text formatting)

### For Backend Enhancement (Phase 2)
1. Implement document Q&A via RAG
2. Add document database storage
3. Create quiz/exercise generator
4. Add mentor skill recommendations

### For Data & Analytics
1. Track mentor usage patterns
2. Measure chatbot effectiveness
3. Improve FAQ based on mentor questions
4. Monitor document uploads

---

## Support

For issues or questions:
- Check `MENTOR_CHATBOT_ARCHITECTURE.md` for detailed docs
- Review `services/mentor_*.py` for implementation details
- Check logs for error messages
- Email: mentora.help@gmail.com
