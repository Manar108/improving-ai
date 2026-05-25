"""Smart Search & Materials Service.

Pipeline:
  User Query → LLM Understanding (Groq) → Intent/Topic/Level Extraction
  → Source Site Selection → Google Custom Search API
  → LLM Ranking + Summarization → Final Structured Response

Rules:
  - Links come ONLY from Google Custom Search — LLM never invents links.
  - Groq is used only for understanding, query optimization, ranking, and summarization.
"""

import json
import logging
import re
import time
from urllib.parse import urlparse

import httpx

from config import settings


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source-site mappings — picked by intent and topic
# ---------------------------------------------------------------------------

INTENT_SITES: dict[str, list[str]] = {
	"roadmap": [
		"roadmap.sh", "github.com", "medium.com", "coursera.org", "freecodecamp.org",
	],
	"course": [
		"coursera.org", "udemy.com", "youtube.com", "edx.org",
		"pluralsight.com", "linkedin.com/learning",
	],
	"materials": [
		"freecodecamp.org", "medium.com", "dev.to", "github.com",
		"geeksforgeeks.org", "w3schools.com",
	],
	"docs": [
		"developer.mozilla.org", "learn.microsoft.com", "docs.python.org",
		"react.dev", "devdocs.io",
	],
	"projects": [
		"github.com", "dev.to", "freecodecamp.org", "kaggle.com",
	],
	"general_search": [
		"stackoverflow.com", "github.com", "medium.com", "dev.to",
		"freecodecamp.org",
	],
}

TOPIC_SITES: dict[str, list[str]] = {
	"ai": ["kaggle.com", "huggingface.co", "tensorflow.org", "pytorch.org"],
	"machine learning": ["kaggle.com", "scikit-learn.org", "tensorflow.org"],
	"deep learning": ["pytorch.org", "tensorflow.org", "huggingface.co"],
	"data science": ["kaggle.com", "towardsdatascience.com", "datacamp.com"],
	"nlp": ["huggingface.co", "spacy.io"],
	"web": ["developer.mozilla.org", "web.dev", "css-tricks.com"],
	"react": ["react.dev", "nextjs.org"],
	"angular": ["angular.io", "angular.dev"],
	"vue": ["vuejs.org"],
	"python": ["docs.python.org", "realpython.com"],
	"javascript": ["developer.mozilla.org", "javascript.info"],
	"typescript": ["typescriptlang.org"],
	"java": ["docs.oracle.com", "baeldung.com"],
	"csharp": ["learn.microsoft.com", "dotnet.microsoft.com"],
	"dotnet": ["learn.microsoft.com", "dotnet.microsoft.com"],
	".net": ["learn.microsoft.com", "dotnet.microsoft.com"],
	"mobile": ["developer.android.com", "flutter.dev", "reactnative.dev"],
	"flutter": ["flutter.dev", "pub.dev"],
	"android": ["developer.android.com"],
	"ios": ["developer.apple.com"],
	"devops": ["docker.com", "kubernetes.io"],
	"cloud": ["aws.amazon.com", "cloud.google.com", "learn.microsoft.com"],
	"cybersecurity": ["owasp.org", "cybrary.it"],
	"design": ["figma.com", "dribbble.com", "behance.net"],
	"ui": ["figma.com", "dribbble.com", "material.io"],
	"ux": ["figma.com", "nngroup.com", "interaction-design.org"],
	"database": ["postgresql.org", "mysql.com", "use-the-index-luke.com"],
	"sql": ["learn.microsoft.com", "postgresql.org", "sqlzoo.net"],
	"backend": ["github.com", "medium.com", "roadmap.sh"],
	"frontend": ["developer.mozilla.org", "css-tricks.com", "web.dev"],
}

ARABIC_SITES = ["maharatech.gov.eg", "youtube.com", "harmash.com", "academy.hsoub.com"]

# Cap site: filters so Google query doesn't get too long
_MAX_SITE_FILTERS = 6

# Groq API endpoint
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SearchService:
	"""Orchestrates the full smart-search pipeline."""

	# ------------------------------------------------------------------ #
	# Public API                                                          #
	# ------------------------------------------------------------------ #

	async def find_materials(
		self,
		query: str,
		language: str = "en",
		history: list[dict] | None = None,
	) -> dict:
		"""Run the complete pipeline and return a structured response.

		Returns dict with keys: success, intent, topic, results, summary.
		"""
		t_start = time.perf_counter()
		logger.info("Search pipeline START | query='%s' lang=%s", query, language)

		# --- Step 1: LLM intent extraction --------------------------------
		t1 = time.perf_counter()
		understanding = await self._extract_intent(query, language)
		intent = understanding.get("intent", "general_search")
		topic = understanding.get("topic", query)
		level = understanding.get("level", "any")
		detected_lang = understanding.get("language", language)
		optimized_query = understanding.get("optimized_query", query)
		t1_elapsed = time.perf_counter() - t1

		logger.info(
			"LLM understanding (%.1fs) | intent=%s topic='%s' level=%s lang=%s opt_query='%s'",
			t1_elapsed, intent, topic, level, detected_lang, optimized_query,
		)

		# --- Step 2: Pick source sites ------------------------------------
		sites = self._get_source_sites(intent, topic, detected_lang)

		# --- Step 3: Build Google query -----------------------------------
		google_query = self._build_search_query(optimized_query, sites)
		logger.info("Google query: %s", google_query)

		# --- Step 4: Google Custom Search ---------------------------------
		t4 = time.perf_counter()
		raw_results = await self._search_google(google_query)
		t4_elapsed = time.perf_counter() - t4
		logger.info("Google CSE (%.1fs): %d results", t4_elapsed, len(raw_results))

		# Retry without site: filters if nothing came back
		if not raw_results and sites:
			logger.warning("No results with site filters — retrying without")
			raw_results = await self._search_google(optimized_query)

		# --- Step 5: LLM ranking + summary --------------------------------
		t5 = time.perf_counter()
		if raw_results:
			ranked = await self._rank_and_summarize(
				query=query, topic=topic, level=level,
				results=raw_results, language=detected_lang,
				history=history or [],
			)
		else:
			# Complete fallback — no Google results at all
			logger.warning("Google returned 0 results — falling back to LLM-only")
			ranked = await self._llm_fallback(query, detected_lang)
		t5_elapsed = time.perf_counter() - t5

		total_elapsed = time.perf_counter() - t_start
		logger.info(
			"Search pipeline END | %d results | total=%.1fs (intent=%.1fs, google=%.1fs, rank=%.1fs)",
			len(ranked.get("results", [])), total_elapsed, t1_elapsed, t4_elapsed, t5_elapsed,
		)
		return {
			"success": True,
			"intent": intent,
			"topic": topic,
			"results": ranked.get("results", []),
			"summary": ranked.get("summary", ""),
		}

	# ------------------------------------------------------------------ #
	# Step 1 — LLM Intent Extraction                                      #
	# ------------------------------------------------------------------ #

	async def _extract_intent(self, query: str, language: str) -> dict:
		"""Ask Groq to extract intent/topic/level from the user query."""
		system = (
			"You are a search-intent analyzer for a mentorship & learning platform.\n"
			"Given a user query, return ONLY valid JSON (no markdown):\n"
			'{"intent":"<roadmap|course|materials|docs|projects|general_search>",'
			'"topic":"<main topic>","level":"<beginner|intermediate|advanced|any>",'
			'"language":"<en|ar>",'
			'"optimized_query":"<optimized English search query for Google>"}\n\n'
			"Rules:\n"
			"- optimized_query must always be in English for best Google results\n"
			"- Detect language from the user message\n"
			"- Return ONLY the JSON object"
		)
		text = await self._call_groq(
			[{"role": "system", "content": system}, {"role": "user", "content": query}],
			temperature=0.1,
		)
		default = {
			"intent": "general_search", "topic": query,
			"level": "any", "language": language, "optimized_query": query,
		}
		return self._parse_json(text, default) if text else default

	# ------------------------------------------------------------------ #
	# Step 2 — Source Site Selection                                       #
	# ------------------------------------------------------------------ #

	def _get_source_sites(self, intent: str, topic: str, language: str) -> list[str]:
		"""Merge intent-based + topic-based + language-based site lists."""
		sites: list[str] = []
		sites.extend(INTENT_SITES.get(intent, INTENT_SITES["general_search"]))

		topic_lower = topic.lower()
		for key, extra in TOPIC_SITES.items():
			if key in topic_lower or topic_lower in key:
				sites.extend(extra)

		if language == "ar":
			sites.extend(ARABIC_SITES)

		# Deduplicate, preserve order, cap length
		seen: set[str] = set()
		unique: list[str] = []
		for s in sites:
			if s not in seen:
				seen.add(s)
				unique.append(s)
		return unique[:_MAX_SITE_FILTERS]

	# ------------------------------------------------------------------ #
	# Step 3 — Build Google Query                                          #
	# ------------------------------------------------------------------ #

	@staticmethod
	def _build_search_query(optimized_query: str, sites: list[str]) -> str:
		if not sites:
			return optimized_query
		site_filter = " OR ".join(f"site:{s}" for s in sites)
		return f"{optimized_query} {site_filter}"

	# ------------------------------------------------------------------ #
	# Step 4 — Google Custom Search                                        #
	# ------------------------------------------------------------------ #

	async def _search_google(self, query: str, num: int = 8) -> list[dict]:
		"""Call Google CSE. Returns list of {title, link, snippet, source}."""
		if not settings.GOOGLE_API_KEY or not settings.CSE_ID:
			logger.warning("Google API keys missing — skipping search")
			return []

		params = {
			"key": settings.GOOGLE_API_KEY,
			"cx": settings.CSE_ID,
			"q": query,
			"num": num,
		}
		try:
			async with httpx.AsyncClient(timeout=15) as client:
				resp = await client.get(
					"https://www.googleapis.com/customsearch/v1", params=params,
				)
			if resp.status_code != 200:
				logger.error("Google CSE %d: %s", resp.status_code, resp.text[:300])
				return []
			items = resp.json().get("items", [])
			return [
				{
					"title": it.get("title", "Untitled"),
					"link": it.get("link", ""),
					"snippet": it.get("snippet", ""),
					"source": self._extract_domain(it.get("link", "")),
				}
				for it in items
			]
		except httpx.TimeoutException:
			logger.error("Google CSE timed out")
			return []
		except Exception as exc:
			logger.error("Google CSE error: %s", exc)
			return []

	# ------------------------------------------------------------------ #
	# Step 5 — LLM Ranking + Summarization                                #
	# ------------------------------------------------------------------ #

	async def _rank_and_summarize(
		self, *, query: str, topic: str, level: str,
		results: list[dict], language: str, history: list[dict],
	) -> dict:
		"""Ask Groq to pick the best results, explain why, and summarize."""
		results_text = "\n".join(
			f"{i+1}. Title: {r['title']}\n   Link: {r['link']}\n   Snippet: {r['snippet']}"
			for i, r in enumerate(results)
		)
		lang_label = "Arabic (العربية)" if language == "ar" else "English"

		if language == "ar":
			system = (
				"أنت خبير في اختيار مصادر التعلم.\n"
				"أجب بالكامل بالعربية.\n"
				"استخدم فقط الروابط المذكورة أدناه — لا تخترع أو تعدل أي رابط.\n\n"
				"المهام:\n"
				"1. اختر أفضل 3-5 نتائج للمستخدم\n"
				"2. اكتب سبباً قصيراً لفائدة كل نتيجة\n"
				"3. اكتب ملخصاً مفيداً\n\n"
				"أرجع JSON فقط:\n"
				'{"results":[{"title":"...","link":"الرابط بالضبط","source":"domain","reason":"..."}],'
				'"summary":"..."}\n'
				"مهم: انسخ الروابط كما هي. بدون markdown."
			)
		else:
			system = (
				"You are an expert learning-resource curator.\n"
				f"Respond ENTIRELY in {lang_label}.\n"
				"You MUST use ONLY the links listed below — NEVER invent or modify a URL.\n\n"
				"Tasks:\n"
				"1. Pick the best 3-5 results for the user\n"
				"2. For each, give a short reason why it is useful\n"
				"3. Write a helpful summary paragraph\n\n"
				"Return ONLY valid JSON:\n"
				'{"results":[{"title":"...","link":"EXACT link","source":"domain","reason":"..."}],'
				'"summary":"..."}\n'
				"IMPORTANT: copy links EXACTLY. No markdown."
			)

		messages: list[dict] = [{"role": "system", "content": system}]
		# Include last few history messages for conversational context
		for msg in (history or [])[-6:]:
			role = "user" if msg.get("role") == "user" else "assistant"
			messages.append({"role": role, "content": str(msg.get("text", msg.get("content", "")))})

		messages.append({
			"role": "user",
			"content": f"Query: {query}\nTopic: {topic}\nLevel: {level}\n\nSearch Results:\n{results_text}",
		})

		text = await self._call_groq(messages, temperature=0.2)
		if not text:
			# Return raw Google results as-is
			return self._raw_fallback(results, language)

		parsed = self._parse_json(text, None)
		if parsed and "results" in parsed:
			return parsed

		return self._raw_fallback(results, language)

	# ------------------------------------------------------------------ #
	# Fallback — no Google results at all                                  #
	# ------------------------------------------------------------------ #

	async def _llm_fallback(self, query: str, language: str) -> dict:
		"""Generate a helpful text answer when Google returned nothing."""
		lang_label = "Arabic" if language == "ar" else "English"
		system = (
			f"You are a helpful learning assistant. Respond in {lang_label}.\n"
			"The user asked for learning resources but no search results were found.\n"
			"Give general guidance about the topic. Do NOT invent any URLs.\n"
			"Return ONLY JSON: {\"results\":[], \"summary\":\"...\"}"
		)
		text = await self._call_groq(
			[{"role": "system", "content": system}, {"role": "user", "content": query}],
			temperature=0.3,
		)
		parsed = self._parse_json(text, None) if text else None
		if parsed:
			return parsed
		fallback_msg = "لم أجد نتائج، حاول صياغة سؤالك بشكل مختلف." if language == "ar" else "No results found. Try rephrasing your query."
		return {"results": [], "summary": fallback_msg}

	# ------------------------------------------------------------------ #
	# Helpers                                                              #
	# ------------------------------------------------------------------ #

	def _raw_fallback(self, results: list[dict], language: str) -> dict:
		"""Return Google results without LLM ranking (parsing failed)."""
		summary = "إليك أفضل ما وجدته:" if language == "ar" else "Here are the best results I found:"
		return {
			"results": [
				{"title": r["title"], "link": r["link"],
				 "source": r.get("source", ""), "reason": r.get("snippet", "")}
				for r in results[:5]
			],
			"summary": summary,
		}

	async def _call_groq(self, messages: list[dict], temperature: float = 0.2) -> str | None:
		"""Call Groq chat completions API. Returns content string or None."""
		if not settings.GROQ_API_KEY:
			logger.warning("GROQ_API_KEY not set — skipping LLM call")
			return None

		payload = {
			"model": settings.MODEL_NAME,
			"messages": messages,
			"temperature": temperature,
		}
		headers = {
			"Authorization": f"Bearer {settings.GROQ_API_KEY}",
			"Content-Type": "application/json",
		}
		try:
			async with httpx.AsyncClient(timeout=30) as client:
				resp = await client.post(_GROQ_URL, headers=headers, json=payload)
			if resp.status_code != 200:
				logger.error("Groq API %d: %s", resp.status_code, resp.text[:300])
				return None
			return resp.json()["choices"][0]["message"]["content"].strip()
		except httpx.TimeoutException:
			logger.error("Groq API timed out")
			return None
		except Exception as exc:
			logger.error("Groq API error: %s", exc)
			return None

	@staticmethod
	def _parse_json(text: str, fallback):
		"""Try to parse JSON from LLM output, with regex fallback."""
		# Direct parse
		try:
			return json.loads(text)
		except json.JSONDecodeError:
			pass
		# Try to extract JSON block from markdown fences or surrounding text
		match = re.search(r"\{[\s\S]*\}", text)
		if match:
			try:
				return json.loads(match.group())
			except json.JSONDecodeError:
				pass
		logger.warning("Failed to parse LLM JSON: %s", text[:200])
		return fallback

	@staticmethod
	def _extract_domain(url: str) -> str:
		"""Extract clean domain from a URL."""
		try:
			host = urlparse(url).netloc
			return host.removeprefix("www.")
		except Exception:
			return ""


search_service = SearchService()