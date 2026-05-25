# Program Recommender

This module implements a complete mentee → program recommender built on the same project patterns as the mentor recommender, but with program-specific semantics:

- program fit is driven by skills and required skill levels
- `target_level` and `education_level` are hard eligibility gates
- program availability and capacity are treated as serving constraints
- collaborative filtering uses enrollments and interactions, not follows

## Core files

- `io.py`: project paths, artifact I/O, and raw table loading helpers
- `features.py`: program features, candidate generation, CF embeddings
- `preprocessing.py`: scaling and feature-frame preparation
- `pipeline.py`: end-to-end training pipeline
- `ranking.py`: model training, scoring, reranking, evaluation
- `run_program_pipeline.py`: CLI entrypoint

## Required input schema

The normalized inputs used by the module are:

- `mentee_profile`: `user_id`, `education_status`, `current_level`, `country_code`
- `mentee_subdomains`: `mentee_id`, `subdomain_id`
- `mentee_interests`: `mentee_id`, `technology_id`, `experience_level`
- `mentorship_posts`: `post_id`, `mentor_id`, `target_level`, `education_level`, `availability`, `capacity`
- `mentorship_requirements`: `post_id`, `technology_id`, `required_experience_level`
- `mentorships` or `applications`: `post_id`, `mentee_id`, time column

## What the model uses

### Feature logic
- `requirement_coverage_score`
- `requirement_overlap_score`
- `required_skill_level_match_score`
- `target_level_gap`
- `education_level_gap`
- `target_level_pass`
- `education_level_pass`
- `availability_pass`
- `candidate_pre_score`
- `program_popularity_log`
- `program_difficulty_score`
- `cf_score`

### Sampling policy
- Train uses harder negatives and keeps signal-bearing mismatches.
- Valid/test keep the full eligible candidate pool for evaluation.

### Output
The recommender returns:
- `mentee_id`
- `post_id`
- `pred_score`
- `match_percentage`

No explanation text is added in the final serving output.

## Run

### 1) Run the synthetic smoke test

```bash
python test_program_recommendations.py
```

### 2) Run the full pipeline against SQL Server

```bash
python -m src.hybrid_recommender.program_recommender.run_program_pipeline
```

### 3) Run the pipeline on normalized CSVs

```bash
python -m src.hybrid_recommender.program_recommender.run_program_pipeline --raw-data "d:/path/to/folder"
```

If the folder contains DB-normalized files like `programs.csv`, `applications.csv`, `mentorships.csv`, and the related profile tables, the pipeline will use them.

## Validation

The included smoke test verifies:
- feature construction
- ranker training
- top-k recommendation generation
- `match_percentage` output in the 0..100 range
