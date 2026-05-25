# Mentorship Recommendation System

A hybrid recommendation system that matches mentees with the most suitable mentors using content-based filtering and a LightGBM learning-to-rank model.

## Project Overview

The system:
1. Matches mentees to mentors based on skills, subdomains, experience, and engagement patterns
2. Uses a **LGBMRanker** (LambdaRank) model trained on historical application data
3. **Label definition**: `label=1` if the user **applied** to the mentor (proxy for user interest)
4. Handles cold-start users through content-based fallback and global popularity ranking
5. Produces explainable recommendations with similarity scores and human-readable reasons

## Architecture

```
Raw CSV Tables
      │
      ▼
┌─────────────────┐
│  Preprocessing   │  Load, clean, normalize, time-filter
│  preprocessing.py│  SCD Type 1 tables → no time filter
│                  │  Temporal tables → filter ≤ train_end
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Features      │  Build mentee, mentor, interaction,
│    features.py   │  and pair-level features
│                  │  Candidate pool generation (4-tier)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Ranking       │  LGBMRanker training with time-based
│    ranking.py    │  train/valid/test split
│                  │  Evaluation: NDCG@k, HitRate@k
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Pipeline      │  End-to-end orchestration
│    pipeline.py   │  Inference API (predict_for_user)
└─────────────────┘
```

## Data Flow

1. **Loading**: Raw CSV tables loaded from `data/raw/` (24 tables)
2. **Preprocessing**: Clean, normalize IDs, handle missing values, deduplicate
3. **Time Filtering**: All temporal tables filtered to `≤ train_end` to prevent data leakage
4. **Feature Engineering**: Build mentee features, mentor features (quality + reliability + popularity), interaction features, and pairwise content features
5. **Candidate Generation**: 4-tier matching (skill → subdomain → domain → global fallback)
6. **Dataset Building**: Assign labels per time split, hard negative sampling (train only)
7. **Model Training**: LGBMRanker with early stopping on validation set
8. **Evaluation**: NDCG@5 and HitRate@5 on held-out test set
9. **Artifact Saving**: Model, scaler, features, and manifest saved to `data/artifacts/`

## Model Features (11)

| Feature | Type | Description |
|---------|------|-------------|
| `skill_overlap_score` | Continuous [0,1] | Jaccard similarity between mentee interests and mentor expertise |
| `skill_coverage_score` | Continuous [0,1] | Fraction of mentee interests covered by mentor expertise |
| `subdomain_similarity` | Continuous [0,1] | Jaccard similarity between subdomain sets |
| `mentor_quality_score` | Continuous | Unified quality: 60% weighted rating + 25% sentiment + 15% positive feedback ratio |
| `mentor_weighted_rating` | Continuous [1-5] | Bayesian-smoothed average rating from mentor feedback |
| `interaction_score_log` | Continuous | Log of aggregated engagement score (likes=1, comments=2, saves=3, shares=4) |
| `experience_gap_abs` | Integer [0,3] | Absolute difference in experience levels |
| `mentor_more_experienced` | Binary {0,1} | Whether mentor has higher experience than mentee |
| `is_following` | Binary {0,1} | Whether mentee follows the mentor |
| `same_country` | Binary {0,1} | Whether mentee and mentor share the same country |
| `popularity_log` | Continuous | Weighted popularity score from enrollments, applications, programs, and engagement |

## How to Run

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Place raw data

Put all CSV files in `data/raw/`. Required tables:
- `mentee_profile.csv`, `mentor_profile.csv`
- `mentee_subdomains.csv`, `mentor_subdomains.csv`
- `mentee_interests.csv`, `mentor_expertise.csv`
- `mentorship_posts.csv`, `mentorships.csv`, `mentorship_applications.csv`
- Plus: feedback, follows, likes, comments, saves, shares, cancellation tables

### 3. Train the model

```bash
python run_pipeline.py
```

This runs the full pipeline and saves artifacts to `data/artifacts/`.

### 4. Test inference

```bash
python test_inference.py
```

### 5. Benchmark latency

```bash
python benchmark_existing_user.py
```

### 6. Use in code

```python
from src.hybrid_recommender import predict_for_user, load_inference_artifacts

# Load once
bundle = load_inference_artifacts(Path("data/artifacts"))

# Predict for any user
recs = predict_for_user(user_id=123, data=bundle, top_k=5)
```

## Model: LGBMRanker (LambdaRank)

- **Objective**: `lambdarank` — optimizes pairwise ranking directly
- **Groups**: Mentee ID (each mentee's candidates form one ranking group)
- **Validation**: Early stopping on NDCG@5 with 50-round patience
- **Hyperparameters**: 300 estimators, lr=0.05, 31 leaves, subsample=0.9

## Evaluation Metrics

Metrics measure: **"Did we recommend mentors the user actually applied to?"**

- **NDCG@k** (Normalized Discounted Cumulative Gain): Are applied-to mentors ranked near the top?
- **HitRate@k**: For what fraction of mentees does at least one applied-to mentor appear in top-k?

Groups with no positive labels or too few candidates are excluded from evaluation to avoid noise.

## Cold-Start Handling

| Scenario | Strategy |
|----------|----------|
| **Existing user** (has pre-computed features) | Model prediction using LGBMRanker |
| **Profile-only user** (has profile but no recommendation features) | Content-based scoring: skill overlap + subdomain similarity + experience gap |
| **Unknown user** (no profile at all) | Global popularity fallback: top mentors by weighted popularity |

## Design Decisions

### Label Definition
The recommendation label is based on **user applications**, not mentorship outcomes:
- `label=1` if the user applied to the mentor's post
- `label=0` otherwise

Applications are a high-quality signal because users cannot apply unless they meet the mentor's requirements (level, skills). This makes each application a strong indicator of genuine interest.

### Time-Based Splitting
- Train/valid/test split is based on mentorship `start_date` quantiles (70/15/15) for global boundaries
- Applications receive their own `time_split` based on `applied_at`
- `event_time_by_mentee` uses the first application date (not mentorship start)
- All features use data from `<= train_end` only — no future information leaks into training
- Applications used for features (popularity, reliability) are filtered to train-only
- Hard negative sampling is applied ONLY to train split; valid/test retain full candidate distribution for realistic evaluation

### SCD Type 1 Tables (Slowly-Changing Dimensions)
Profile-level tables are treated as latest-snapshot:
- `mentee_profile`, `mentor_profile`
- `mentee_subdomains`, `mentor_subdomains`
- `mentee_interests`, `mentor_expertise`

These are NOT time-filtered because they represent a user's current identity (declared skills, interests, subdomains) which change infrequently. If the platform tracks historical profile changes, these should be versioned and time-filtered.

### Experience Handling
- `junior`, `mid`, `senior` are valid aliases mapped to canonical values (`beginner`, `intermediate`, `advanced`)
- Overall experience comes from `mentee_profile.current_level` ONLY
- Skill-level experience (from `mentee_interests.experience_level`) is separate — it is used for preprocessing normalization but NOT for overriding the mentee's overall experience level
- `"None"` in `mentee_interests.experience_level` is treated as `"no_experience"` (intentional business meaning, not missing data)

### Sentiment Usage
Sentiment is used ONLY in the unified `mentor_quality_score`:
```
mentor_quality_score = 0.60 × mentor_weighted_rating
                     + 0.25 × mentor_sentiment_score
                     + 0.15 × mentor_positive_feedback_ratio
```
Sentiment is NEVER used in candidate generation or interaction features.

### Follow Signal
The follow relationship is used ONLY as the binary `is_following` feature. It is intentionally EXCLUDED from `interaction_score` to prevent double-counting.

### Popularity Features
Engagement counts (likes, comments, saves, shares) are computed from time-filtered interaction tables — NOT from DB-level counters on `mentorship_posts` (which would leak future data). The final `popularity_log` is a weighted sum of log1p values (no double-log).

### Scaling
Binary features (`is_following`, `same_country`, `mentor_more_experienced`) are excluded from MinMaxScaler to preserve their 0/1 semantics.

## Repository Structure

```
mentorship-recommendation/
├── run_pipeline.py              # Main entry point — train the model
├── test_inference.py            # Test inference for different user types
├── benchmark_existing_user.py   # Latency benchmarking
├── requirements.txt
├── config/
│   └── time_split_config.csv    # Auto-generated time boundaries
├── data/
│   ├── raw/                     # Source CSV tables
│   ├── processed/               # Cleaned intermediate data
│   ├── features/                # Engineered feature tables
│   └── artifacts/               # Model, scaler, feature manifests
└── src/
    └── hybrid_recommender/
        ├── __init__.py          # Public API exports
        ├── io.py                # I/O utilities (load/save)
        ├── preprocessing.py     # Data cleaning, normalization, time filtering
        ├── features.py          # Feature engineering, candidate generation
        ├── pipeline.py          # End-to-end pipeline, inference API
        ├── ranking.py           # Model training, evaluation, recommendations
        └── testing.py           # Dataset validation checks
```

## AI Functions & Usage

This project exposes (or expects) a small set of AI-related functions that services and routes call. Place implementations either in `ai/` (model artifacts + loaders) or in the service layer `backend-ai/services/`.

- Loader (recommended): `ai/models/loader.py`
    - `def load_model(bundle_path: Path) -> Tuple[Model, Scaler, Dict]:` — load model weights, scaler and metadata.

- Inference / public API: `src/hybrid_recommender/pipeline.py` (or `pipeline.py` at repo root)
    - `def load_inference_artifacts(path: Path) -> Dict:` — load model, scaler, feature manifest once at startup.
    - `def predict_for_user(user_id: int, data: Dict, top_k: int = 10) -> List[Dict]:` — return ranked recommendations (id, score, reason).

- Service-level helpers: `backend-ai/services/recommendation_service.py`
    - `def get_recommendations_for_user(user_id: int, top_k: int = 10) -> List[Dict]:` — wraps `predict_for_user` and applies post-processing and filtering.

- Search / Candidate generation: `backend-ai/services/search_service.py`
    - `def search_candidates(mentee_profile: Dict, limit: int=100) -> List[int]:` — return candidate mentor ids.

- RAG / LLM helpers: `backend-ai/services/rag_service.py`, `backend-ai/services/llm_service.py`
    - `def retrieve_documents(query: str, top_k: int=5) -> List[Dict]`
    - `def call_llm(prompt: str, **kwargs) -> str`

- Utility loaders for encoders/scalers: any `*.joblib` loader utility; recommended: `ai/models/loader.py` includes `load_scaler()` / `load_encoders()`.

Examples (quick):

```python
from pathlib import Path
from src.hybrid_recommender import load_inference_artifacts, predict_for_user

bundle = load_inference_artifacts(Path('data/artifacts'))
recs = predict_for_user(user_id=123, data=bundle, top_k=5)
```

Notes
- Keep heavy artifact files outside git (use `data/artifacts/` or git-lfs).
- Implement `loader.py` with safe fallbacks (dummy model) so tests and CI can run without large weights.

