#!/usr/bin/env python3
"""
Mentorship AI Testing Playground — Streamlit App
Tests Chatbot (with all 13 intents), Sentiment, and Recommendations.
"""

import os
import time
# pyrefly: ignore [missing-import]
import streamlit as st 
import requests
import json
from datetime import datetime

# ==================== CONFIG ====================
DEFAULT_BASE_URL = os.getenv("AI_BACKEND_BASE_URL", "http://localhost:8088/api/v1")
TIMEOUT = 30

def _root_url() -> str:
    """Derive the root URL (no /api/v1) from the configured base URL."""
    base = st.session_state.get("base_url", DEFAULT_BASE_URL)
    # Strip trailing /api/v1 or /api/v1/ to get the root
    for suffix in ["/api/v1/", "/api/v1"]:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Mentora AI Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        color: #374151;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.4rem;
    }
    .intent-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        margin-right: 0.3rem;
    }
    .intent-find_mentor { background: #3b82f6; }
    .intent-ask_mentor_recommendation { background: #8b5cf6; }
    .intent-ask_program_recommendation { background: #7c3aed; }
    .intent-recommendation_explanation { background: #6d28d9; }
    .intent-task_help { background: #f59e0b; }
    .intent-submit_task { background: #ef4444; }
    .intent-roadmap_request { background: #10b981; }
    .intent-materials_request { background: #f97316; }
    .intent-faq { background: #06b6d4; }
    .intent-complaint { background: #dc2626; }
    .intent-support_request { background: #ea580c; }
    .intent-general_question { background: #6366f1; }
    .intent-greeting { background: #22c55e; }
    .intent-off_topic { background: #9ca3af; }

    .chat-user {
        background: linear-gradient(135deg, #667eea20, #764ba220);
        border-left: 3px solid #667eea;
        padding: 0.8rem 1rem;
        border-radius: 0.5rem;
        margin: 0.4rem 0;
    }
    .chat-bot {
        background: #f8fafc;
        border-left: 3px solid #10b981;
        padding: 0.8rem 1rem;
        border-radius: 0.5rem;
        margin: 0.4rem 0;
    }
    .latency-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 0.8rem;
        font-size: 0.7rem;
        font-weight: 500;
        background: #f3f4f6;
        color: #6b7280;
    }
    .test-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: box-shadow 0.2s;
    }
    .test-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .status-ok { color: #10b981; font-weight: 600; }
    .status-fail { color: #ef4444; font-weight: 600; }

    /* Recommendation cards in chat */
    .rec-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .rec-match-high {
        color: #059669;
        background: #d1fae5;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .rec-match-mid {
        color: #d97706;
        background: #fef3c7;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .rec-match-low {
        color: #dc2626;
        background: #fee2e2;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .rec-reason {
        color: #374151;
        font-size: 0.9rem;
        font-style: italic;
        border-left: 3px solid #8b5cf6;
        padding-left: 0.6rem;
        margin-top: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== SESSION STATE ====================
if "base_url" not in st.session_state:
    st.session_state.base_url = DEFAULT_BASE_URL
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "test_results" not in st.session_state:
    st.session_state.test_results = []

BASE_URL = st.session_state.base_url


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚡ Connection")
    new_url = st.text_input("Backend API URL", value=BASE_URL, key="url_input")
    if st.button("Connect", use_container_width=True, type="primary"):
        st.session_state.base_url = new_url
        st.rerun()

    st.divider()

    # Quick health check
    st.markdown("### 🏥 Status")
    try:
        health_url = _root_url() + "/health"
        r = requests.get(health_url, timeout=5)
        if r.status_code == 200:
            st.markdown("<span class='status-ok'>● Connected</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-fail'>● Error</span>", unsafe_allow_html=True)
    except Exception:
        st.markdown("<span class='status-fail'>● Offline</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🏷️ Intent Legend")
    intents_info = {
        "greeting": "👋 Greeting",
        "find_mentor": "🔍 Find Mentor",
        "ask_mentor_recommendation": "📋 Recommend Mentor",
        "ask_program_recommendation": "🎓 Recommend Program",
        "recommendation_explanation": "💬 Why Recommended",
        "task_help": "📝 Task Help",
        "submit_task": "📤 Submit Task",
        "roadmap_request": "🗺️ Roadmap",
        "materials_request": "🎬 Materials",
        "faq": "❓ FAQ",
        "complaint": "🚨 Complaint",
        "support_request": "🔧 Support",
        "general_question": "💡 General Q",
        "off_topic": "🚫 Off-topic",
    }
    for intent, label in intents_info.items():
        st.markdown(
            f"<span class='intent-badge intent-{intent}'>{label}</span>",
            unsafe_allow_html=True,
        )


# ==================== HELPER FUNCTIONS ====================

def api_call(method: str, endpoint: str, data=None, params=None):
    """Make HTTP request. Returns (success, data, latency_ms)."""
    url = f"{st.session_state.base_url}{endpoint}"
    t0 = time.time()
    try:
        if method.upper() == "GET":
            r = requests.get(url, params=params, timeout=TIMEOUT)
        else:
            r = requests.post(url, json=data, timeout=TIMEOUT)
        latency = (time.time() - t0) * 1000
        r.raise_for_status()
        return True, r.json(), latency
    except requests.exceptions.Timeout:
        return False, {"error": f"Timeout (>{TIMEOUT}s)"}, 0
    except requests.exceptions.ConnectionError:
        return False, {"error": f"Cannot connect to {st.session_state.base_url}"}, 0
    except requests.exceptions.HTTPError as e:
        latency = (time.time() - t0) * 1000
        try:
            return False, e.response.json(), latency
        except Exception:
            return False, {"error": f"HTTP {e.response.status_code}"}, latency
    except Exception as e:
        return False, {"error": str(e)}, 0


def intent_badge(intent: str) -> str:
    """Return HTML for an intent badge."""
    return f"<span class='intent-badge intent-{intent}'>{intent}</span>"


def render_materials(materials):
    """Render material cards."""
    if not materials:
        return
    for i, m in enumerate(materials, 1):
        with st.container(border=True):
            title = m.get("title", "Untitled")
            url = m.get("url", "")
            kind = m.get("kind", "article")
            source = m.get("source", "")
            reason = m.get("reason", "")
            kind_emoji = {"videos": "🎬", "courses": "🎓", "docs": "📖", "article": "📄"}.get(kind, "📄")
            st.markdown(f"**{i}. {kind_emoji} [{title}]({url})**" if url else f"**{i}. {kind_emoji} {title}**")
            st.caption(f"{kind.upper()} • {source}")
            if reason:
                st.write(reason)


# ==================== MAIN ====================

st.markdown("<div class='main-header'>🧠 Mentora AI Playground</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Test all chatbot intents, sentiment analysis, and recommendations</div>", unsafe_allow_html=True)

# ==================== TABS ====================
tab_chat, tab_mentor_chat, tab_test, tab_sentiment, tab_recs, tab_info = st.tabs([
    "💬 Mentee Chat", "👨‍🏫 Mentor Chat", "🧪 Intent Tests", "🎭 Sentiment", "👥 Recommendations", "ℹ️ System"
])


# ==================== TAB: CHAT ====================
with tab_chat:
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-user'>👤 <strong>You:</strong> {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                intent = msg.get("intent", "unknown")
                latency = msg.get("latency", 0)
                badge = intent_badge(intent)
                latency_html = f"<span class='latency-badge'>{latency:.0f}ms</span>" if latency else ""
                st.markdown(
                    f"<div class='chat-bot'>"
                    f"🤖 <strong>Mentora:</strong> {badge} {latency_html}<br><br>"
                    f"{msg['content']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Render materials if present
                if msg.get("materials"):
                    render_materials(msg["materials"])
                # Render recommendations if present
                if msg.get("recommendations"):
                    st.markdown("<div style='margin-top:0.5rem;font-weight:600;color:#4b5563;'>🎯 Recommendations for you</div>", unsafe_allow_html=True)
                    for i, rec in enumerate(msg["recommendations"], 1):
                        pct = int(rec.get('match_percentage', 75))
                        if pct >= 90:
                            match_class = "rec-match-high"
                        elif pct >= 75:
                            match_class = "rec-match-mid"
                        else:
                            match_class = "rec-match-low"
                        with st.container(border=True):
                            c1, c2 = st.columns([4, 1])
                            with c1:
                                st.markdown(f"**{i}. {rec.get('mentor_name', 'Unknown')}**  <span class='{match_class}'>{pct}% match</span>", unsafe_allow_html=True)
                                st.caption(f"Domain: {rec.get('domain', 'General')} • Score: {rec.get('score', 0):.3f}")
                                reason = rec.get('reason', '')
                                if reason:
                                    st.markdown(f"<div class='rec-reason'>💡 {reason}</div>", unsafe_allow_html=True)
                            with c2:
                                st.metric("Match", f"{pct}%", delta=f"#{i}", delta_color="off")
                                st.caption(f"ID: `{rec.get('mentor_id', '')[:8]}...`")

        # Input area
        col1, col2 = st.columns([5, 1])
        with col1:
            user_msg = st.text_input(
                "Your message:",
                placeholder="اكتب رسالتك هنا... (Arabic or English)",
                key="chat_input",
                label_visibility="collapsed",
            )
        with col2:
            send = st.button("📤 Send", use_container_width=True, type="primary")

        # Options
        with st.expander("⚙️ Chat Options & Recommendation Test"):
            chat_uid = st.text_input("User ID (for personalized recommendations)", "", key="chat_uid", help="Enter a UUID or integer user ID to test personalized mentor recommendations via chat")
            if st.button("🎯 Quick Test: Recommend mentors for this user", use_container_width=True):
                test_msg = "رشحلي أحسن مرشدين" if chat_uid.strip() else "recommend mentors for me"
                payload = {"message": test_msg}
                if chat_uid.strip():
                    payload["user_id"] = chat_uid.strip()
                with st.spinner("Getting recommendations via chat..."):
                    ok, resp, latency = api_call("POST", "/chat", data=payload)
                if ok:
                    st.session_state.chat_history.append({"role": "user", "content": test_msg + (f" (user={chat_uid.strip()})" if chat_uid.strip() else "")})
                    st.session_state.chat_history.append({
                        "role": "bot",
                        "content": resp.get("answer", "Here are your recommendations:"),
                        "intent": resp.get("intent", "unknown"),
                        "latency": latency,
                        "materials": resp.get("materials", []),
                        "recommendations": resp.get("recommendations", []),
                    })
                    st.rerun()
                else:
                    st.error(f"Failed: {resp.get('error')}")

        if send and user_msg.strip():
            payload = {"message": user_msg}
            if chat_uid.strip():
                payload["user_id"] = chat_uid.strip()

            with st.spinner("Processing..."):
                ok, resp, latency = api_call("POST", "/chat", data=payload)

            if ok:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_msg,
                })
                st.session_state.chat_history.append({
                    "role": "bot",
                    "content": resp.get("answer", "No response"),
                    "intent": resp.get("intent", "unknown"),
                    "latency": latency,
                    "materials": resp.get("materials", []),
                    "recommendations": resp.get("recommendations", []),
                })
                st.rerun()
            else:
                st.error(f"Error: {resp.get('error', 'Unknown')}")

    # pyrefly: ignore [parse-error]
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    # Quick test buttons
    # pyrefly: ignore [parse-error]
    st.markdown("---")
    # pyrefly: ignore [parse-error]
    st.markdown("**Quick tests:**")
    quick_tests = {
        "👋 Greeting": "مرحبا",
        "🔍 Find Mentor": "عايز mentor في AI",
        "📋 Recommend": "رشحلي أحسن مرشدين",
        "📝 Task Help": "مش فاهم التاسك",
        "📤 Submit": "إزاي أسلم التاسك؟",
        "🗺️ Roadmap": "عايز roadmap للـ machine learning",
        "🎬 Materials": "هات فيديو python",
        "🧠 Explain": "ايه machine learning",
        "❓ FAQ": "هل المنصة مجانية؟",
        "🚨 Complaint": "عايز أشتكي من المرشد",
        "🔧 Support": "مش بيرفع",
        "💡 General": "ايه الفرق بين Python و Java؟",
        "🚫 Off-topic": "الجو عامل ايه النهاردة؟",
    }
    cols = st.columns(4)
    for i, (label, msg) in enumerate(quick_tests.items()):
        with cols[i % 4]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                payload = {"message": msg}
                with st.spinner(f"Testing: {msg[:30]}..."):
                    ok, resp, latency = api_call("POST", "/chat", data=payload)
                if ok:
                    st.session_state.chat_history.append({"role": "user", "content": msg})
                    st.session_state.chat_history.append({
                        "role": "bot",
                        "content": resp.get("answer", "No response"),
                        "intent": resp.get("intent", "unknown"),
                        "latency": latency,
                        "materials": resp.get("materials", []),
                        "recommendations": resp.get("recommendations", []),
                    })
                    st.rerun()
                else:
                    st.error(f"Failed: {resp.get('error')}")







# ==================== TAB: MENTOR CHAT ====================
with tab_mentor_chat:
    st.markdown("<div class='section-header'>👨‍🏫 Mentor Chatbot</div>", unsafe_allow_html=True)
    st.markdown("Test the mentor-specific chatbot with mentor intents (analytics, workflow, materials, FAQs).")

    # Session state for mentor chat
    if "mentor_chat_history" not in st.session_state:
        st.session_state.mentor_chat_history = []

    # Mentor user selection
    MENTOR_USERS = {
        "Select a mentor...": "",
        "Hassan Mahmoud (AI & Data Science)": "45FE4901-84AA-5F02-A2E3-BED68B585311",
        "Aya Tarek (Software Engineering)": "CC1C07C2-F15F-549C-8832-74B4EBA0E517",
        "Ahmed Salem (Design)": "97887A88-7F04-5044-BD9E-4DAB25457BA7",
    }

    mentor_choice = st.selectbox("Select mentor:", list(MENTOR_USERS.keys()), key="mentor_chat_select")
    mentor_uid = st.text_input("Or enter Mentor ID:", value=MENTOR_USERS.get(mentor_choice, ""), key="mentor_chat_uid")

    # Display mentor chat history
    for msg in st.session_state.mentor_chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-user'>👤 <strong>Mentor:</strong> {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            intent = msg.get("intent", "unknown")
            latency = msg.get("latency", 0)
            badge = intent_badge(intent)
            latency_html = f"<span class='latency-badge'>{latency:.0f}ms</span>" if latency else ""
            st.markdown(
                f"<div class='chat-bot'>"
                f"🤖 <strong>Mentora:</strong> {badge} {latency_html}<br><br>"
                f"{msg['content']}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Input
    col1, col2 = st.columns([5, 1])
    with col1:
        mentor_msg = st.text_input(
            "Your message:",
            placeholder="اكتب رسالتك كمرشد... (Arabic or English)",
            key="mentor_chat_input",
            label_visibility="collapsed",
        )
    with col2:
        mentor_send = st.button("📤 Send", use_container_width=True, type="primary", key="mentor_send")

    if mentor_send and mentor_msg.strip():
        payload = {"message": mentor_msg}
        if mentor_uid.strip():
            payload["user_id"] = mentor_uid.strip()

        with st.spinner("Processing..."):
            ok, resp, latency = api_call("POST", "/mentor-chat", data=payload)

        if ok:
            st.session_state.mentor_chat_history.append({"role": "user", "content": mentor_msg})
            st.session_state.mentor_chat_history.append({
                "role": "bot",
                "content": resp.get("answer", "No response"),
                "intent": resp.get("intent", "unknown"),
                "latency": latency,
            })
            st.rerun()
        else:
            st.error(f"Error: {resp.get('error', 'Unknown')}")

    if st.session_state.mentor_chat_history:
        if st.button("🗑️ Clear Mentor Chat", type="secondary", key="clear_mentor_chat"):
            st.session_state.mentor_chat_history = []
            st.rerun()

    # Quick test buttons for all 8 mentor intents
    st.markdown("---")
    st.markdown("**Quick tests (Mentor Intents):**")
    mentor_quick_tests = {
        "👋 Greeting": "مرحبا",
        "❓ FAQ": "ازاي أنشئ برنامج جديد؟",
        "📚 Materials": "أعطني تمارين Python للمبتدئين",
        "📊 Analytics": "كام منتي في برنامجي؟",
        "🔧 Workflow": "ازاي أتواصل مع المنتيز؟",
        "📄 Document": "لخص الملف ده",
        "💡 General": "شرحلي الفرق بين REST و GraphQL",
        "🚫 Off-topic": "ايه الجو النهاردة؟",
    }
    cols = st.columns(4)
    for i, (label, msg) in enumerate(mentor_quick_tests.items()):
        with cols[i % 4]:
            if st.button(label, key=f"mentor_quick_{i}", use_container_width=True):
                payload = {"message": msg}
                if mentor_uid.strip():
                    payload["user_id"] = mentor_uid.strip()
                with st.spinner(f"Testing: {msg[:30]}..."):
                    ok, resp, latency = api_call("POST", "/mentor-chat", data=payload)
                if ok:
                    st.session_state.mentor_chat_history.append({"role": "user", "content": msg})
                    st.session_state.mentor_chat_history.append({
                        "role": "bot",
                        "content": resp.get("answer", "No response"),
                        "intent": resp.get("intent", "unknown"),
                        "latency": latency,
                    })
                    st.rerun()
                else:
                    st.error(f"Failed: {resp.get('error')}")

# ==================== TAB: INTENT TESTS ====================
with tab_test:
    st.markdown("<div class='section-header'>🧪 Automated Intent Classification Tests</div>", unsafe_allow_html=True)
    st.markdown("Run all test cases to verify the LLM correctly classifies each intent.")

    # Test cases
    TEST_CASES = [
        # Greetings
        ("مرحبا", "greeting"),
        ("hi there", "greeting"),
        ("السلام عليكم", "greeting"),
        ("hello", "greeting"),
        # Find mentor
        ("عايز mentor في AI", "find_mentor"),
        ("I need a mentor for web development", "find_mentor"),
        ("مين أحسن mentor في AI", "find_mentor"),
        ("best mentor in web dev", "find_mentor"),
        # Mentor recommendations
        ("رشحلي mentor", "ask_mentor_recommendation"),
        ("recommend mentors for me", "ask_mentor_recommendation"),
        ("اقترح عليا مرشدين مناسبين", "ask_mentor_recommendation"),
        # Task help
        ("مش فاهم التاسك", "task_help"),
        ("help me with the assignment", "task_help"),
        # Submit task
        ("إزاي أسلم التاسك", "submit_task"),
        ("how to submit my task?", "submit_task"),
        ("الديدلاين امتى", "submit_task"),
        # Roadmap request
        ("عايز roadmap للـ AI", "roadmap_request"),
        ("give me a learning path for backend", "roadmap_request"),
        ("كيف أبدأ أتعلم web dev", "roadmap_request"),
        # Materials request (NEW)
        ("هات فيديو python", "materials_request"),
        ("give me videos about React", "materials_request"),
        ("كورسات AI", "materials_request"),
        # General question (concept explanation)
        ("ايه machine learning", "general_question"),
        ("شرحلي OOP", "general_question"),
        ("what is REST API", "general_question"),
        # FAQ
        ("هل المنصة مجانية؟", "faq"),
        ("how to register?", "faq"),
        # Complaint (NEW)
        ("عايز أشتكي من المرشد", "complaint"),
        ("المرشد وحش", "complaint"),
        ("mentor is rude", "complaint"),
        # Support request (NEW)
        ("مش عارف أرفع التاسك", "support_request"),
        ("مش بيرفع", "support_request"),
        ("not working", "support_request"),
        # General question
        ("ايه الفرق بين Python و Java؟", "general_question"),
        # Off-topic
        ("الجو عامل ايه؟", "off_topic"),
        ("tell me a joke", "off_topic"),
    ]

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("▶️ Run All Tests", type="primary", use_container_width=True):
            results = []
            progress = st.progress(0, text="Running tests...")

            for i, (msg, expected) in enumerate(TEST_CASES):
                progress.progress((i + 1) / len(TEST_CASES), text=f"Testing: {msg[:40]}...")
                ok, resp, latency = api_call("POST", "/chat", data={"message": msg})
                if ok:
                    actual = resp.get("intent", "unknown")
                    passed = actual == expected
                    results.append({
                        "message": msg,
                        "expected": expected,
                        "actual": actual,
                        "passed": passed,
                        "latency": latency,
                    })
                else:
                    results.append({
                        "message": msg,
                        "expected": expected,
                        "actual": "ERROR",
                        "passed": False,
                        "latency": 0,
                    })

            progress.empty()
            st.session_state.test_results = results

    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.test_results = []
            st.rerun()

    # Show results
    if st.session_state.test_results:
        results = st.session_state.test_results
        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        avg_latency = sum(r["latency"] for r in results) / total if total else 0

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tests", total)
        c2.metric("Passed ✅", passed)
        c3.metric("Failed ❌", total - passed)
        c4.metric("Accuracy", f"{(passed/total)*100:.0f}%")

        st.metric("Avg Latency", f"{avg_latency:.0f}ms")

        # Detailed results
        for r in results:
            icon = "✅" if r["passed"] else "❌"
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1.5, 1.5, 0.5])
                with c1:
                    st.markdown(f"**{r['message']}**")
                with c2:
                    st.markdown(
                        f"Expected: {intent_badge(r['expected'])}",
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f"Got: {intent_badge(r['actual'])}",
                        unsafe_allow_html=True,
                    )
                with c4:
                    st.write(icon)


# ==================== TAB: SENTIMENT ====================
with tab_sentiment:
    st.markdown("<div class='section-header'>🎭 Sentiment Analysis</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        feedback_text = st.text_area(
            "Enter feedback text:",
            placeholder="e.g., المرشد كان ممتاز والبرنامج مفيد جداً",
            height=100,
            key="sent_input",
        )
    with col2:
        st.write("")
        analyze = st.button("🔍 Analyze", use_container_width=True, type="primary")

    if analyze and feedback_text.strip():
        with st.spinner("Analyzing..."):
            ok, resp, latency = api_call("POST", "/sentiment/predict", data={"text": feedback_text})

        if ok:
            label = resp.get("label", "unknown").upper()
            confidence = resp.get("confidence", 0)
            scores = resp.get("scores", {})
            emoji_map = {"POSITIVE": "😊", "NEUTRAL": "😐", "NEGATIVE": "😞"}
            emoji = emoji_map.get(label, "🤔")

            c1, c2, c3 = st.columns(3)
            c1.metric("Sentiment", f"{emoji} {label}")
            c2.metric("Confidence", f"{confidence:.1%}")
            c3.metric("Latency", f"{latency:.0f}ms")

            # Score bars
            st.markdown("**Score Breakdown:**")
            for lbl, score in sorted(scores.items(), key=lambda x: -x[1]):
                st.progress(score, text=f"{lbl.capitalize()}: {score:.4f}")

            with st.expander("Raw JSON"):
                st.json(resp)
        else:
            st.error(f"Error: {resp.get('error')}")

    # Batch
    with st.expander("📦 Batch Prediction"):
        batch = st.text_area(
            "One text per line (max 32):",
            placeholder="Line 1: Great experience\nLine 2: Not good\nLine 3: Okay",
            height=80,
            key="batch_input",
        )
        if st.button("🔍 Analyze Batch"):
            texts = [t.strip() for t in batch.split("\n") if t.strip()]
            if len(texts) > 32:
                st.error("Max 32 texts")
            elif texts:
                with st.spinner(f"Analyzing {len(texts)} texts..."):
                    ok, resp, _ = api_call("POST", "/sentiment/predict-batch", data={"texts": texts})
                if ok:
                    for i, (text, result) in enumerate(zip(texts, resp.get("results", [])), 1):
                        lbl = result.get("label", "?").upper()
                        conf = result.get("confidence", 0)
                        emoji = {"POSITIVE": "😊", "NEUTRAL": "😐", "NEGATIVE": "😞"}.get(lbl, "🤔")
                        st.write(f"**{i}.** {text[:60]}... → {emoji} {lbl} ({conf:.0%})")
                else:
                    st.error(f"Error: {resp.get('error')}")

    # Mentor Feedback Summary
    st.markdown("---")
    st.markdown("<div class='section-header'>📊 Mentor Feedback Summary & Satisfaction Rate</div>", unsafe_allow_html=True)
    st.markdown("Get a comprehensive AI-generated summary of all feedback for a specific mentor.")

    SAMPLE_MENTORS = {
        "Hassan Mahmoud (28 feedbacks)": "45FE4901-84AA-5F02-A2E3-BED68B585311",
        "Aya Tarek (27 feedbacks)": "CC1C07C2-F15F-549C-8832-74B4EBA0E517",
        "Ahmed Salem (26 feedbacks)": "97887A88-7F04-5044-BD9E-4DAB25457BA7",
        "Fatma Farag (25 feedbacks)": "064D8F4E-AA7B-55DE-876A-51B3C25949E0",
        "Laila Mahmoud (25 feedbacks)": "572AF556-59A4-5ED5-A5D9-791ABC11F3B8",
        "Select a mentor...": "",
    }
    # Note: Can be extended to fetch mentor list from DB dynamically

    col1, col2 = st.columns([3, 1])
    with col1:
        mentor_choice = st.selectbox("Select mentor or enter ID:", list(SAMPLE_MENTORS.keys()), key="mentor_select")
        mentor_id_input = st.text_input("Or enter Mentor ID manually:", value=SAMPLE_MENTORS.get(mentor_choice, ""), key="mentor_id_input")
    with col2:
        st.write("")
        get_summary = st.button("📊 Get Summary", use_container_width=True, type="primary")

    if get_summary and mentor_id_input.strip():
        with st.spinner("Analyzing feedbacks with AI..."):
            ok, resp, latency = api_call("GET", f"/sentiment/mentor-summary/{mentor_id_input.strip()}")

        if ok:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("👤 Mentor", resp.get("mentor_name", "Unknown"))
            c2.metric("😊 Satisfaction", f"{resp.get('satisfaction_rate', 0)}%")
            c3.metric("⭐ Avg Rating", f"{resp.get('average_rating', 0)}/5")
            c4.metric("⏱️ Latency", f"{latency:.0f}ms")

            breakdown = resp.get("breakdown", {})
            if breakdown:
                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric("✅ Positive", breakdown.get("positive", 0))
                bc2.metric("😐 Neutral", breakdown.get("neutral", 0))
                bc3.metric("❌ Negative", breakdown.get("negative", 0))
                bc4.metric("📝 Total", breakdown.get("total", 0))

            summary = resp.get("summary", "")
            if summary:
                st.info(f"**📝 AI Summary:** {summary}")

            pos_themes = resp.get("top_positive_themes", [])
            neg_themes = resp.get("top_negative_themes", [])
            if pos_themes:
                st.markdown("**👍 Positive:** " + " ".join(pos_themes))
            if neg_themes:
                st.markdown("**👎 Negative:** " + " ".join(neg_themes))

            with st.expander("Raw JSON"):
                st.json(resp)
        else:
            st.error(f"Error: {resp.get('error', 'Unknown')}")


# ==================== TAB: RECOMMENDATIONS ====================
with tab_recs:
    st.markdown("<div class='section-header'>👥 Mentor Recommendations — Deep Test</div>", unsafe_allow_html=True)
    st.markdown("Test a specific User ID and verify normalized match percentages (55–98%) and smart explanation sentences.")

    # Real user IDs from the database
    REAL_USERS = {
        "Select a real user...": "",
        "Fatma Hassan (AI & Data Science, 13 follows, 10 interests)": "DC537411-D831-5393-ABE8-6154CF0A6C0A",
        "Aya Mahmoud (Software Engineering, 14 follows, 11 interests)": "46BAD202-62BE-5E0B-9155-4C9BFB34EB95",
        "Nour Yasin (Software Engineering, 14 follows, 9 interests)": "86F43E98-7BB5-5CC3-B446-89ABA27DC179",
        "Fatma Mahmoud (Product & Business, 13 follows, 8 interests)": "BA5ABF7C-BC1B-5494-B37A-094C388BA88B",
        "Ali Mahmoud (Software Engineering, 13 follows, 4 interests)": "1B3A0B0B-5CA4-5B9E-975E-2C35CB40029C",
    }

    col1, col2 = st.columns([3, 1])
    with col1:
        user_choice = st.selectbox("Select a real user:", list(REAL_USERS.keys()), key="user_select")
        uid = st.text_input("Or enter User ID (UUID or integer):", value=REAL_USERS.get(user_choice, ""), key="rec_uid")
    with col2:
        st.write("")
        get_recs = st.button("🎯 Get Recs", use_container_width=True, type="primary")

    if get_recs and uid.strip():
        with st.spinner("Fetching recommendations..."):
            ok, resp, latency = api_call("GET", "/recommend", params={"user_id": uid})

        if ok:
            recs = resp if isinstance(resp, list) else resp.get("recommendations", [])
            c1, c2, c3 = st.columns(3)
            c1.metric("Results", len(recs))
            c2.metric("Latency", f"{latency:.0f}ms")
            if recs:
                avg_pct = sum(int(r.get('match_percentage', 75)) for r in recs) / len(recs)
                c3.metric("Avg Match %", f"{avg_pct:.0f}%")
            else:
                c3.metric("Avg Match %", "N/A")

            for i, rec in enumerate(recs, 1):
                pct = int(rec.get('match_percentage', 75))
                if pct >= 90:
                    match_badge = f"<span class='rec-match-high'>{pct}% match</span>"
                elif pct >= 75:
                    match_badge = f"<span class='rec-match-mid'>{pct}% match</span>"
                else:
                    match_badge = f"<span class='rec-match-low'>{pct}% match</span>"

                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"**#{i} — {rec.get('mentor_name', 'Unknown')}**  {match_badge}", unsafe_allow_html=True)
                        st.caption(f"Domain: `{rec.get('domain', 'General')}` • Mentor ID: `{rec.get('mentor_id', '')}`")
                        reason = rec.get('reason', '')
                        if reason:
                            st.markdown(f"<div class='rec-reason'>💡 {reason}</div>", unsafe_allow_html=True)
                        else:
                            st.warning("No smart reason generated — check backend logic.")
                    with c2:
                        st.metric("Match", f"{pct}%")
                        score = rec.get('score', 0)
                        st.metric("Model Score", f"{score:.4f}" if isinstance(score, (int, float)) else str(score))

            with st.expander("📦 Raw JSON Response"):
                st.json(resp)
        else:
            st.error(f"Error: {resp.get('error')}")

    st.markdown("---")
    st.markdown("<div class='section-header'>🧑‍🎓 Program Recommendations — Test</div>", unsafe_allow_html=True)
    st.markdown("Test program recommendations (program posts) for a given user ID.")

    col1, col2 = st.columns([3, 1])
    with col1:
        program_uid = st.text_input("User ID for program recs:", value=REAL_USERS.get(user_choice, ""), key="program_rec_uid")
        top_k_prog = st.number_input("Top K", min_value=1, max_value=50, value=10, key="prog_top_k")
    with col2:
        st.write("")
        get_prog = st.button("🎯 Get Program Recs", use_container_width=True, type="primary")

    if get_prog and program_uid.strip():
        with st.spinner("Fetching program recommendations..."):
            ok, resp, latency = api_call("GET", "/program-recommend", params={"user_id": program_uid, "top_k": top_k_prog})

        if ok:
            recs = resp.get("recommendations", []) if isinstance(resp, dict) else []
            c1, c2, c3 = st.columns(3)
            c1.metric("Results", len(recs))
            c2.metric("Latency", f"{latency:.0f}ms")
            if recs:
                avg_pct = sum(int(r.get('match_percentage', 75)) for r in recs) / len(recs)
                c3.metric("Avg Match %", f"{avg_pct:.0f}%")
            else:
                c3.metric("Avg Match %", "N/A")

            for i, rec in enumerate(recs, 1):
                pct = int(rec.get('match_percentage', 75))
                if pct >= 90:
                    match_badge = f"<span class='rec-match-high'>{pct}% match</span>"
                elif pct >= 75:
                    match_badge = f"<span class='rec-match-mid'>{pct}% match</span>"
                else:
                    match_badge = f"<span class='rec-match-low'>{pct}% match</span>"

                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        title = rec.get('title', 'Untitled Program')
                        st.markdown(f"**#{i} — {title}**  {match_badge}", unsafe_allow_html=True)
                        st.caption(f"Mentor: {rec.get('mentor_name','Unknown')} • Post ID: {rec.get('post_id','')}")
                        reason = rec.get('reason', '')
                        if reason:
                            st.markdown(f"<div class='rec-reason'>💡 {reason}</div>", unsafe_allow_html=True)
                    with c2:
                        st.metric("Match", f"{pct}%")
                        score = rec.get('pred_score', rec.get('score', 0))
                        st.metric("Model Score", f"{score:.4f}" if isinstance(score, (int, float)) else str(score))

            with st.expander("📦 Raw JSON Response"):
                st.json(resp)
        else:
            st.error(f"Error: {resp.get('error')}")


# ==================== TAB: SYSTEM INFO ====================
with tab_info:
    st.markdown("<div class='section-header'>ℹ️ System Information</div>", unsafe_allow_html=True)

    # Health checks
    st.markdown("#### 🏥 Health Checks")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Check API Health"):
            _t0 = time.time()
            try:
                _r = requests.get(_root_url() + "/health", timeout=10)
                latency = (time.time() - _t0) * 1000
                _r.raise_for_status()
                resp = _r.json()
                ok = True
            except Exception as _e:
                ok = False
                resp = {"error": str(_e)}
                latency = 0
            if ok:
                st.success(f"✅ API running ({latency:.0f}ms)")
                st.json(resp)
            else:
                st.error(f"❌ {resp.get('error')}")

    with c2:
        if st.button("Check DB Health"):
            _t0 = time.time()
            try:
                _r2 = requests.get(_root_url() + "/db-health", timeout=15)
                latency = (time.time() - _t0) * 1000
                _r2.raise_for_status()
                resp = _r2.json()
                ok = True
            except Exception as _e2:
                ok = False
                resp = {"error": str(_e2)}
                latency = 0
            if ok:
                summary = resp.get("summary", {})
                st.success(f"✅ DB connected ({latency:.0f}ms)")
                st.metric("Tables Found", summary.get("tables_found", 0))
                st.metric("Total Rows", summary.get("total_rows", 0))
                with st.expander("Full Report"):
                    st.json(resp)
            else:
                st.error(f"❌ {resp.get('error')}")

    # Endpoints
    st.markdown("#### 📚 Endpoints")
    endpoints = {
        "Chat (Mentee)": "POST /api/v1/chat",
        "Chat (Mentor)": "POST /api/v1/mentor-chat",
        "Recommendations": "GET /api/v1/recommend?user_id=...",
        "Program Recs": "GET /api/v1/program-recommend?user_id=...",
        "Sentiment": "POST /api/v1/sentiment/predict",
        "Batch Sentiment": "POST /api/v1/sentiment/predict-batch",
        "Mentor Summary": "GET /api/v1/sentiment/mentor-summary/{mentor_id}",
        "Health": "GET /health",
        "DB Health": "GET /db-health",
    }
    for name, ep in endpoints.items():
        st.code(f"{name}: {ep}", language="http")

    # Intent documentation
    st.markdown("#### 🏷️ Intent System (14 Intents)")
    intent_docs = {
        "greeting": "Hello, hi, مرحبا, etc.",
        "find_mentor": "User wants to FIND a mentor in a specific field (DB search)",
        "ask_mentor_recommendation": "User wants AI to RECOMMEND the best mentors (personalized)",
        "task_help": "User needs HELP understanding or completing a task",
        "submit_task": "User asks HOW TO SUBMIT work, deadlines, where to upload",
        "roadmap_request": "User wants a LEARNING ROADMAP or study path",
        "materials_request": "User wants specific MATERIALS: videos, courses, articles",
        "explanation_request": "User wants a CONCEPT EXPLAINED: what is X, explain Y",
        "faq": "Questions about PLATFORM RULES, registration, duration, system usage",
        "complaint": "User REPORTING an issue with a mentor or behavior",
        "support_request": "User has a TECHNICAL PROBLEM: upload fails, errors, bugs",
        "general_question": "Learning/education question that doesn't fit other categories",
        "off_topic": "Unrelated to education/mentorship (politics, weather, etc.)",
    }
    for intent, desc in intent_docs.items():
        st.markdown(f"- {intent_badge(intent)} **{intent}**: {desc}", unsafe_allow_html=True)

    st.markdown("#### ⚙️ Config")
    st.info(f"**Base URL:** `{st.session_state.base_url}`")
    st.info(f"**Timeout:** `{TIMEOUT}s`")


# ==================== FOOTER ====================
st.divider()
st.markdown(
    f"<div style='text-align:center;color:#9ca3af;font-size:0.8rem;'>"
    f"🧠 Mentora AI Playground v5.0 • Unified Intents (14) • Document Q\u0026A • Match % Badges • {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    f"</div>",
    unsafe_allow_html=True,
)
