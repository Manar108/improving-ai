"""Fetch real test data from the database for manual testing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from database.db import database

# 1. Real mentees who have applications
mentees = database.run_query_df("""
    SELECT TOP 5 u.user_id, u.first_name, u.last_name
    FROM users u
    WHERE u.user_id IN (
        SELECT DISTINCT a.MenteeProfileId FROM applications a
    )
""")
print("=== REAL MENTEES (use for /recommend & /chat) ===")
for _, r in mentees.iterrows():
    print(f"  user_id: {r['user_id']}  |  Name: {r['first_name']} {r['last_name']}")

# 2. Mentors with feedbacks
mentors = database.run_query_df("""
    SELECT TOP 3 mp.user_id, u.first_name, u.last_name, COUNT(f.FeedbackId) as fb_count
    FROM mentor_profile mp
    INNER JOIN users u ON u.user_id = mp.user_id
    INNER JOIN feedbacks f ON f.MentorProfileId = mp.user_id
    GROUP BY mp.user_id, u.first_name, u.last_name
    ORDER BY COUNT(f.FeedbackId) DESC
""")
print("\n=== MENTORS WITH FEEDBACKS (use for /sentiment/mentor-summary) ===")
for _, r in mentors.iterrows():
    print(f"  mentor_id: {r['user_id']}  |  Name: {r['first_name']} {r['last_name']}  |  {r['fb_count']} feedbacks")

# 3. Sample feedbacks
fb = database.run_query_df("""
    SELECT TOP 8 f.Comment, f.Rating
    FROM feedbacks f
    WHERE f.Comment IS NOT NULL AND LEN(CAST(f.Comment AS NVARCHAR(MAX))) > 5
""")
print("\n=== SAMPLE FEEDBACK TEXTS (use for /sentiment/predict) ===")
for _, r in fb.iterrows():
    print(f"  [{r['Rating']}*] {str(r['Comment'])[:100]}")

# 4. Chatbot test messages
print("\n=== CHATBOT TEST MESSAGES (POST /api/v1/chat) ===")
msgs = [
    ("ar", "مرحبا"),
    ("ar", "عايز مرشد في React و Node.js"),
    ("ar", "ازاي اقدم على mentorship؟"),
    ("ar", "عايز roadmap لتعلم Machine Learning"),
    ("en", "Can you recommend a mentor for Python?"),
    ("en", "What are the top mentors on the platform?"),
    ("en", "Help me with my task about database design"),
    ("ar", "ايه هو الطقس النهارده؟"),
]
for lang, msg in msgs:
    print(f"  [{lang}] {msg}")

# 5. Sentiment test texts
print("\n=== SENTIMENT TEST TEXTS (POST /api/v1/sentiment/predict) ===")
texts = [
    "المرشد ممتاز جدا وساعدني كتير في فهم React",
    "The mentor was incredibly helpful and supportive!",
    "تجربة عادية مش وحشة بس مش حلوة",
    "Average experience, nothing special",
    "المرشد مكانش متاح ابدا وضيع وقتي",
    "Terrible experience, the mentor never responded",
    "شكرا على المساعدة في المشروع",
    "I learned a lot about Python and APIs",
]
for t in texts:
    print(f'  "{t}"')

# Print curl commands
print("\n" + "=" * 70)
print("  COPY-PASTE CURL COMMANDS (run while backend is on port 8000)")
print("=" * 70)

if not mentees.empty:
    uid = mentees.iloc[0]["user_id"]
    print(f"\n# 1. Recommendations for {mentees.iloc[0]['first_name']}:")
    print(f"curl http://localhost:8000/api/v1/recommend?user_id={uid}")

    print(f'\n# 2. Chatbot - Find mentor (Arabic):')
    print(f'curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" ^')
    print(f'  -d "{{\\"user_id\\": \\"{uid}\\", \\"message\\": \\"عايز مرشد في React\\"}}"')

    print(f'\n# 3. Chatbot - Greeting:')
    print(f'curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" ^')
    print(f'  -d "{{\\"message\\": \\"مرحبا\\"}}"')

print(f'\n# 4. Sentiment - Positive:')
print(f'curl -X POST http://localhost:8000/api/v1/sentiment/predict -H "Content-Type: application/json" ^')
print(f'  -d "{{\\"text\\": \\"The mentor was excellent and very helpful\\"}}"')

print(f'\n# 5. Sentiment - Negative:')
print(f'curl -X POST http://localhost:8000/api/v1/sentiment/predict -H "Content-Type: application/json" ^')
print(f'  -d "{{\\"text\\": \\"Terrible experience the mentor never responded\\"}}"')

if not mentors.empty:
    mid = mentors.iloc[0]["user_id"]
    print(f"\n# 6. Mentor Feedback Summary for {mentors.iloc[0]['first_name']}:")
    print(f"curl http://localhost:8000/api/v1/sentiment/mentor-summary/{mid}")

print(f"\n# 7. Health:")
print(f"curl http://localhost:8000/health")
