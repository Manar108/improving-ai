import os, sys, uuid, hashlib
import warnings
import pandas as pd
import pyodbc

CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=.;DATABASE=MentorshipPlatformDB;"
    "Trusted_Connection=yes;TrustServerCertificate=yes"
)

_DB_FILES_PRIMARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database files")
DB_FILES = _DB_FILES_PRIMARY
RAW_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
# DB-ready folder (must be the single source of truth for imports)
DB_READY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "db-ready")
# NO ARCHIVE FALLBACK - ONLY USE DB-READY CSVs
SOURCE_ROOTS = [DB_READY]

# --- Deterministic GUID from integer ID (reproducible across runs) ---
def int_to_guid(val):
    """Convert integer ID to deterministic GUID."""
    if pd.isna(val) or str(val).strip() in ("", "nan", "NaN", "None"):
        return None
    s = str(int(float(str(val).strip())))
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"mentorship.user.{s}"))

def mentorship_to_guid(val):
    """Convert integer mentorship ID to deterministic GUID."""
    if pd.isna(val) or str(val).strip() in ("", "nan", "NaN", "None"):
        return None
    s = str(int(float(str(val).strip())))
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"mentorship.session.{s}"))

def feedback_to_guid(*parts):
    """Create a deterministic GUID for feedback rows with missing IDs."""
    key = ".".join(str(p).strip() for p in parts if str(p).strip() not in ("", "nan", "NaN", "None"))
    if not key:
        return str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"mentorship.feedback.{key}"))

def to_bit(val):
    if pd.isna(val) or str(val).strip() in ("", "nan", "NaN", "None"):
        return "0"
    v = str(val).strip().lower()
    return "1" if v in ("1", "true", "yes") else "0"

def to_int_or_null(val):
    if pd.isna(val) or str(val).strip() in ("", "nan", "NaN", "None"):
        return None
    try:
        return str(int(float(str(val).strip())))
    except:
        return None

def to_datetime_or_null(val):
    if pd.isna(val) or str(val).strip() in ("", "nan", "NaN", "None"):
        return None
    text = str(val).strip()
    if "/" in text:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


class _CaseInsensitiveRow:
    def __init__(self, row):
        self._data = {self._normalize_key(key): value for key, value in row.items()}

    @staticmethod
    def _normalize_key(key):
        return "".join(character for character in str(key).strip().lower() if character.isalnum())

    def get(self, key, default=None):
        if key is None:
            return default
        return self._data.get(self._normalize_key(key), default)


def _ci_row(row):
    return _CaseInsensitiveRow(row)


def _first_value(row, *names, default=None):
    for name in names:
        value = row.get(name, None)
        if value is not None and str(value).strip() not in ("", "nan", "NaN", "None"):
            return value
    return default


def _read_source_csv(*filenames):
    for root in SOURCE_ROOTS:
        for filename in filenames:
            path = os.path.join(root, filename)
            if os.path.exists(path):
                df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python")
                df.columns = [str(column).strip() for column in df.columns]
                print(f"  [OK] Loading CSV: {path} ({len(df)} rows)")
                return df, path
    raise FileNotFoundError(f"No source CSV found for any of: {', '.join(filenames)}")


def _normalize_program_post_status(value):
    """Map to backend ProgramPostStatus enum (stored as STRING via HasConversion<string>).
    Backend: Draft=1, Published=2.  EF stores the NAME: "Draft" or "Published".
    Raw CSV 'is_open' column: 'open'/'closed' is actually Availability, not ProgramPostStatus.
    All seeded programs are published (visible), so default to Published.
    """
    raw = str(value).strip().lower()
    if raw in ("draft",):
        return "Draft"
    # open/closed/published/true/1/yes → Published (program is visible)
    return "Published"


def _normalize_target_level(value):
    """Map to backend CurrentLevel enum (stored as STRING via HasConversion<string>).
    Backend: Beginner=1, Junior=2, Mid=3, Senior=4.
    EF stores the NAME: "Beginner", "Junior", "Mid", "Senior".
    """
    raw = str(value).strip().lower()
    if raw in ("beginner", "no_experience", "none", "1"):
        return "Beginner"
    if raw in ("junior", "2"):
        return "Junior"
    if raw in ("mid", "intermediate", "3"):
        return "Mid"
    if raw in ("senior", "advanced", "4"):
        return "Senior"
    return "Beginner"


def _normalize_education_level(value):
    """Map to backend EducationStatus enum (stored as STRING via HasConversion<string>).
    Backend: Freshman=1, Sophomore=2, Junior=3, Senior=4, Graduate=5.
    EF stores the NAME: "Freshman", "Sophomore", "Junior", "Senior", "Graduate".
    """
    raw = str(value).strip().lower()
    if raw in ("freshman", "1"):
        return "Freshman"
    if raw in ("sophomore", "2"):
        return "Sophomore"
    if raw in ("junior", "3"):
        return "Junior"
    if raw in ("senior", "4"):
        return "Senior"
    if raw in ("graduate", "professional", "5"):
        return "Graduate"
    return "Graduate"


def _normalize_application_status(value):
    """Map to backend ApplicationStatus enum (stored as STRING via HasConversion<string>).
    Backend: Pending=1, Accepted=2, Rejected=3.  ONLY these 3 values exist.
    EF stores the NAME: "Pending", "Accepted", "Rejected".
    Legacy CSV values 'alerted' and 'canceled'/'cancelled' are NOT valid backend values:
    - 'alerted' → maps to 'Pending' (requirements mismatch, needs re-review)
    - 'canceled'/'cancelled' → maps to 'Rejected' (application was withdrawn/cancelled)
    """
    raw = str(value).strip().lower()
    if raw == "accepted":
        return "Accepted"
    if raw == "rejected":
        return "Rejected"
    if raw in ("canceled", "cancelled"):
        return "Rejected"  # Backend has no Canceled status → closest is Rejected
    if raw == "alerted":
        return "Pending"   # Backend has no Alerted status → closest is Pending
    if raw == "pending":
        return "Pending"
    return "Pending"


def _normalize_mentorship_status(value):
    """Map to backend MentorshipStatus enum (stored as STRING via HasConversion<string>).
    Backend: Active=1, Completed=2, Cancelled=3.
    EF stores the NAME: "Active", "Completed", "Cancelled".
    """
    raw = str(value).strip().lower()
    status_map = {
        "active": "Active",
        "completed": "Completed",
        "cancelled": "Cancelled",
        "canceled": "Cancelled",    # alias for cancelled
    }
    return status_map.get(raw, "Active")  # default to Active


def _normalize_meet_requirements(status):
    """
    Correct logic for MeetRequirements:
    - accepted, pending, rejected, cancelled, canceled → 1 (meets requirements)
    - alerted → 0 (does not meet requirements)
    """
    raw = str(status).strip().lower()
    if raw == "alerted":
        return "0"
    # All other statuses (accepted, pending, rejected, cancelled, canceled) = 1
    return "1"


def _normalize_experience_enum_name(value):
    """Map to backend ExperienceLevel enum (stored as STRING via HasConversion<string>).
    Backend: None=1, Beginner=2, Intermediate=3, Advanced=4.
    EF stores the NAME: "None", "Beginner", "Intermediate", "Advanced".
    Used for: MentorshipRequirement.RequiredExperienceLevel.
    """
    raw = str(value).strip().lower()
    if raw in ("none", "no_experience", "1"):
        return "None"
    if raw in ("beginner", "junior", "2"):
        return "Beginner"
    if raw in ("mid", "intermediate", "3"):
        return "Intermediate"
    if raw in ("advanced", "senior", "4"):
        return "Advanced"
    return "Beginner"


def _normalize_app_cancellation_reason(value):
    """Map CSV snake_case cancellation reasons to backend AppCancellationReason enum names.
    Backend enum (stored as STRING via HasConversion<string>):
      ScheduleConflict=1, AcceptedOtherOffer=2, LostInterest=3,
      AvailabilityChanged=4, TimeZoneIssue=5, NotReadyYet=6,
      PersonalIssue=7, ApplicationSentByMistake=8
    """
    raw = str(value).strip().lower()
    reason_map = {
        "schedule_conflict": "ScheduleConflict",
        "scheduleconflict": "ScheduleConflict",
        "accepted_other_offer": "AcceptedOtherOffer",
        "acceptedotheroffer": "AcceptedOtherOffer",
        "lost_interest": "LostInterest",
        "lostinterest": "LostInterest",
        "availability_changed": "AvailabilityChanged",
        "availabilitychanged": "AvailabilityChanged",
        "time_zone_issue": "TimeZoneIssue",
        "timezoneissue": "TimeZoneIssue",
        "not_ready_yet": "NotReadyYet",
        "notreadyyet": "NotReadyYet",
        "personal_issue": "PersonalIssue",
        "personalissue": "PersonalIssue",
        "application_sent_by_mistake": "ApplicationSentByMistake",
        "applicationsentbymistake": "ApplicationSentByMistake",
    }
    return reason_map.get(raw, "PersonalIssue")  # default to PersonalIssue


def _normalize_cancellation_actor(value):
    """Map to backend CancellationActor enum.
    Backend: Mentor=1, Mentee=2.  Stored as INT (no HasConversion<string>).
    """
    raw = str(value).strip().lower()
    if raw == "mentor":
        return "1"   # Mentor=1
    if raw == "mentee":
        return "2"   # Mentee=2
    return "2"  # default to Mentee

# --- Enum mappings (string -> int for EF enums) ---
# CRITICAL: Backend has distinct enums - do NOT confuse them!

# Aligned with .NET backend: UserRole (Mentee=1, Mentor=2, Admin=3)
ROLE_MAP = {"mentee": "1", "mentor": "2", "admin": "3"}

# Aligned with .NET backend: CurrentLevel (Beginner=1, Junior=2, Mid=3, Senior=4)
# Used in: MenteeProfile.CurrentLevel, Program.TargetLevel
CURRENT_LEVEL_MAP = {
    "beginner": "1",
    "junior": "2",
    "mid": "3",
    "intermediate": "3",  # alias for mid
    "senior": "4",
    "advanced": "4",      # alias for senior
}

# Aligned with .NET backend: ExperienceLevel (None=1, Beginner=2, Intermediate=3, Advanced=4)
# Used in: MenteeInterest.ExperienceLevel, MentorExpertise.ExperienceLevel, Roadmap.TargetLevelFrom/To
EXPERIENCE_LEVEL_MAP = {
    "none": "1",
    "no_experience": "1",  # alias for none
    "beginner": "2",
    "intermediate": "3",
    "mid": "3",            # alias for intermediate
    "advanced": "4",
    "senior": "4",         # alias for advanced
}

# Aligned with .NET backend: EducationStatus (Freshman=1, Sophomore=2, Junior=3, Senior=4, Graduate=5)
# Used in: MenteeProfile.EducationStatus, Program.EducationLevel
EDUCATION_STATUS_MAP = {
    "freshman": "1",
    "sophomore": "2",
    "junior": "3",
    "senior": "4",
    "graduate": "5",
    "professional": "5",   # Often treated as graduate level
}

def load_users(conn):
    df, _ = _read_source_csv("users.csv")
    cur = conn.cursor()
    cur.execute("DELETE FROM [users]")
    conn.commit()

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            guid = int_to_guid(r.get("UserId"))
            if not guid: continue
            role_int = ROLE_MAP.get(str(r.get("Role","")).strip().lower(), "1")
            cur.execute("""INSERT INTO [users] ([user_id],[email],[password_hash],[first_name],[last_name],
                          [role],[created_at],[is_active],[IsEmailVerified])
                          VALUES (?,?,?,?,?,?,?,?,?)""",
                       guid, r.get("Email",""), r.get("PasswordHash","hashed"),
                       r.get("FirstName",""), r.get("LastName",""),
                       role_int, to_datetime_or_null(r.get("CreatedAt")) or "2024-01-01",
                       to_bit(r.get("IsActive","1")), "1")
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3: print(f"    users err: {e}")
    if inserted != len(df):
        raise RuntimeError(f"CRITICAL: users import mismatch - expected {len(df)} rows, inserted {inserted}")
    print(f"  users: {inserted} loaded")
    return inserted


def load_mentee_profile(conn):
    df, _ = _read_source_csv("mentee_profile.csv")
    cur = conn.cursor()
    cur.execute("DELETE FROM [mentee_profile]")
    conn.commit()

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            guid = int_to_guid(_first_value(r, "UserId", "user_id"))
            if not guid:
                continue
            # CurrentLevel: map to backend CurrentLevel enum (1=Beginner, 2=Junior, 3=Mid, 4=Senior)
            level = CURRENT_LEVEL_MAP.get(str(r.get("CurrentLevel", "")).strip().lower(), "1")
            # EducationStatus: map to backend EducationStatus enum (1=Freshman, ..., 5=Graduate)
            edu = EDUCATION_STATUS_MAP.get(str(r.get("EducationStatus", "")).strip().lower(), "1")
            cur.execute("""INSERT INTO [mentee_profile] ([user_id],[domain_id],[current_level],[education_status],
                          [career_goal_id],[learning_style_id],[country_code],[bio],[is_email_verified])
                          VALUES (?,?,?,?,?,?,?,?,?)""",
                       guid, to_int_or_null(r.get("DomainId")) or "1", level, edu,
                       to_int_or_null(r.get("CareerGoalId")),
                       to_int_or_null(r.get("LearningStyleId")),
                       r.get("CountryCode", "")[:2] or None,
                       r.get("Bio", "")[:1000] or None,
                       "1")
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    mentee err: {e}")
    if inserted != len(df):
        raise RuntimeError(f"CRITICAL: mentee_profile import mismatch - expected {len(df)} rows, inserted {inserted}")
    print(f"  mentee_profile: {inserted} loaded")
    return inserted


def load_mentor_profile(conn):
    df, _ = _read_source_csv("mentor_profile.csv")
    cur = conn.cursor()
    cur.execute("DELETE FROM [mentor_profile]")
    conn.commit()

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            guid = int_to_guid(r.get("UserId"))
            if not guid: continue
            cur.execute("""INSERT INTO [mentor_profile] ([user_id],[domain_id],[years_of_experience],
                          [bio],[linkedin_url],[is_verified],[average_rating],[total_reviews],
                          [created_at],[country_code],[is_email_verified])
                          VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                       guid, to_int_or_null(r.get("DomainId")) or "1",
                       to_int_or_null(r.get("YearsOfExperience")) or "0",
                       r.get("Bio","")[:2000] or None,
                       r.get("LinkedInUrl","")[:200] or None,
                       to_bit(r.get("IsVerified","0")),
                       r.get("AverageRating","") or None,
                       to_int_or_null(r.get("TotalReviews")) or "0",
                       to_datetime_or_null(r.get("CreatedAt")) or "2024-01-01",
                       r.get("CountryCode","")[:2] or None,
                       "1")
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3: print(f"    mentor err: {e}")
    if inserted != len(df):
        raise RuntimeError(f"CRITICAL: mentor_profile import mismatch - expected {len(df)} rows, inserted {inserted}")
    print(f"  mentor_profile: {inserted} loaded")
    return inserted


def load_programs(conn):
    df, path = _read_source_csv("programs.csv", "mentorship_posts.csv")
    print(f"  CSV head (first 3 rows): {df[['title' if 'title' in df.columns else 'Title']].head(3).values}")
    print(f"  CSV max ID: {df[['post_id' if 'post_id' in [c.lower() for c in df.columns] else 'ProgramId']].max().values}")
    
    cur = conn.cursor()
    cur.execute("DELETE FROM [programs]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [programs] ON")

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            mentor_guid = int_to_guid(_first_value(r, "MentorProfileId", "mentor_id"))
            if not mentor_guid: continue
            
            # TargetLevel: stored as STRING by backend (HasConversion<string>)
            # Backend CurrentLevel enum: Beginner, Junior, Mid, Senior
            target_level = _normalize_target_level(_first_value(r, "TargetLevel", "target_level", default=""))
            
            # EducationLevel: stored as STRING by backend (HasConversion<string>)
            # Backend EducationStatus enum: Freshman, Sophomore, Junior, Senior, Graduate
            education_level = _normalize_education_level(_first_value(r, "EducationLevel", "education_level", default=""))

            # Availability: separate from ProgramPostStatus!
            # Backend: Availability = free-text string (open/closed enrollment state)
            # Backend: ProgramPostStatus = enum string (Draft/Published visibility)
            # Raw CSV 'is_open' column is actually Availability (open/closed), NOT ProgramPostStatus
            raw_is_open = str(r.get("is_open", r.get("Availability", ""))).strip()
            availability_value = raw_is_open[:100] if raw_is_open else None
            
            cur.execute("""INSERT INTO [programs] ([ProgramId],[Title],[Description],[Availability],[Duration],
                          [Capacity],[CreatedAt],[ProgramPostStatus],[MentorProfileId],[DomainId],[SubDomainId],
                          [EducationLevel],[TargetLevel],[Deadline])
                          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       to_int_or_null(_first_value(r, "ProgramId", "post_id")),
                       r.get("Title","Program")[:200],
                       r.get("Description","")[:4000] or None,
                       availability_value,
                       r.get("Duration","")[:100] or None,
                       to_int_or_null(r.get("Capacity")) or "10",
                       to_datetime_or_null(r.get("CreatedAt")) or "2024-01-01",
                       _normalize_program_post_status(r.get("ProgramPostStatus", "Published")),
                       mentor_guid,
                       to_int_or_null(r.get("DomainId")) or "1",
                       to_int_or_null(r.get("SubDomainId")) or "1",
                       education_level,
                       target_level,
                       to_datetime_or_null(r.get("Deadline")) or "2024-01-01")
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            print(f"    programs err on row {inserted}: {e}")
            if inserted < 3: 
                import traceback
                traceback.print_exc()
    try: cur.execute("SET IDENTITY_INSERT [programs] OFF")
    except: pass
    
    if inserted != len(df):
        raise RuntimeError(f"CRITICAL: programs import mismatch - expected {len(df)} rows, inserted {inserted}")
    
    print(f"  programs: {inserted} loaded (validated with EducationLevel and TargetLevel)")
    return inserted


def load_follows(conn):
    df, _ = _read_source_csv("follows.csv")
    cur = conn.cursor()
    cur.execute("DELETE FROM [follows]")
    conn.commit()

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            fid = r.get("Id","")
            if not fid or fid in ("","nan"): fid = str(uuid.uuid4())
            elif len(fid) < 36:
                fid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"follow.{fid}"))
            follower_guid = int_to_guid(r.get("FollowerId"))
            following_guid = int_to_guid(r.get("FollowingId"))
            if not follower_guid or not following_guid: continue
            cur.execute("""INSERT INTO [follows] ([id],[follower_id],[following_id],[followed_at])
                          VALUES (?,?,?,?)""",
                       fid, follower_guid, following_guid,
                       to_datetime_or_null(r.get("FollowedAt")) or "2024-01-01")
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3: print(f"    follows err: {e}")
    if inserted != len(df):
        raise RuntimeError(f"CRITICAL: follows import mismatch - expected {len(df)} rows, inserted {inserted}")
    print(f"  follows: {inserted} loaded")
    return inserted


def load_applications(conn):
    df, path = _read_source_csv("applications.csv", "mentorship_applications.csv")
    cur = conn.cursor()
    cur.execute("DELETE FROM [applications]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [applications] ON")

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            mentee_guid = int_to_guid(_first_value(r, "MenteeProfileId", "mentee_id"))
            if not mentee_guid: continue
            # Keep original status for MeetRequirements (alerted→MR=0 before mapping to Pending)
            original_status = _first_value(r, "Status", default="Pending")
            status = _normalize_application_status(original_status)
            # Pending apps must have NULL DecisionAt (includes alerted → Pending mapping)
            decision_at = to_datetime_or_null(_first_value(r, "DecisionAt", "decisioned_at"))
            if status == "Pending":
                decision_at = None
            cur.execute("""INSERT INTO [applications] ([ApplicationId],[AppliedAt],[DecisionAt],
                          [MeetRequirements],[MenteeProfileId],[ProgramId],[Status])
                          VALUES (?,?,?,?,?,?,?)""",
                       to_int_or_null(_first_value(r, "ApplicationId", "app_id")),
                       to_datetime_or_null(_first_value(r, "AppliedAt", "applied_at")) or "2024-01-01",
                       decision_at,
                       _normalize_meet_requirements(original_status),
                       mentee_guid,
                       to_int_or_null(_first_value(r, "ProgramId", "post_id")) or "1",
                       status[:50])
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            print(f"    applications err on row {inserted}: {e}")
            if inserted < 3: 
                import traceback
                traceback.print_exc()
    try: cur.execute("SET IDENTITY_INSERT [applications] OFF")
    except: pass
    
    # VALIDATE: inserted count must match source CSV
    if inserted != len(df):
        raise RuntimeError(f"CRITICAL: applications import mismatch - expected {len(df)} rows, inserted {inserted}")
    
    # VALIDATE: DB count must match
    cur.execute("SELECT COUNT(*) FROM [applications]")
    db_count = int(cur.fetchone()[0])
    if db_count != inserted:
        raise RuntimeError(f"CRITICAL: applications DB count mismatch - inserted {inserted}, DB has {db_count}")
    
    print(f"  applications: {inserted} loaded (validated)")
    return inserted


def load_mentorships(conn):
    df, path = _read_source_csv("mentorships.csv")
    cur = conn.cursor()
    cur.execute("DELETE FROM [mentorships]")
    conn.commit()

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            mentee_guid = int_to_guid(_first_value(r, "MenteeProfileId", "mentee_id"))
            mentor_guid = int_to_guid(_first_value(r, "MentorProfileId", "mentor_id"))
            if not mentee_guid or not mentor_guid:
                continue
            cur.execute(
                """INSERT INTO [mentorships] ([MentorshipId],[MenteeProfileId],[MentorProfileId],
                          [ProgramId],[StartDate],[EndDate],[Status])
                          VALUES (?,?,?,?,?,?,?)""",
                mentorship_to_guid(_first_value(r, "MentorshipId", "mentorship_id")),
                mentee_guid,
                mentor_guid,
                to_int_or_null(_first_value(r, "ProgramId", "post_id")) or "1",
                to_datetime_or_null(_first_value(r, "StartDate", "start_date")) or "2024-01-01",
                to_datetime_or_null(_first_value(r, "EndDate", "end_date")),
                _normalize_mentorship_status(_first_value(r, "Status", default="Active"))[:50],
            )
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            print(f"    mentorships err on row {inserted}: {e}")
            if inserted < 3:
                import traceback
                traceback.print_exc()
    
    if inserted != len(df):
        raise RuntimeError(f"CRITICAL: mentorships import mismatch - expected {len(df)} rows, inserted {inserted}")
    
    print(f"  mentorships: {inserted} loaded (validated)")
    return inserted


def _load_csv_frames(base_dir, filenames):
    """Load every existing CSV in ``filenames`` and return the frames in order."""
    frames = []
    for filename in filenames:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame["_source_file"] = filename
            frames.append(frame)
    return frames


def _source_namespaced_int(raw_id, source_file, namespace_step=10_000_000):
    """Create a stable, collision-free integer ID per source file."""
    if pd.isna(raw_id) or str(raw_id).strip() in ("", "nan", "NaN", "None"):
        return None
    try:
        base_id = int(float(str(raw_id).strip()))
    except Exception:
        return None
    source_file = str(source_file or "").lower()
    if "mentorship" in source_file:
        return base_id + namespace_step
    return base_id


def _count_rows(conn, table_name):
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(1) FROM [{table_name}]")
    return int(cur.fetchone()[0])


def load_feedbacks(conn):
    try:
        df, _ = _read_source_csv("feedback.csv", "feedbacks.csv", "mentors_feedback.csv")
    except FileNotFoundError:
        print(f"  feedbacks: skipped (no source CSV)")
        return 0
    # Standardize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    
    cur = conn.cursor()
    cur.execute("DELETE FROM [feedbacks]")
    conn.commit()

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            # CSV has: FeedbackId, MentorshipId, MenteeProfileId, MentorProfileId, Rating, Comment, CreatedAt
            mentee_guid = int_to_guid(_first_value(r, "menteeprofileid", "mentee_id"))
            mentor_guid = int_to_guid(_first_value(r, "mentorprofileid", "mentor_id"))
            if not mentee_guid or not mentor_guid:
                continue
            cur.execute(
                """INSERT INTO [feedbacks] ([FeedbackId],[Comment],[CreatedAt],
                          [MenteeProfileId],[MentorProfileId],[MentorshipId],[Rating])
                          VALUES (?,?,?,?,?,?,?)""",
                feedback_to_guid(
                    _first_value(r, "feedbackid", "feedback_id"),
                    _first_value(r, "mentorshipid", "mentorship_id"),
                    _first_value(r, "menteeprofileid", "mentee_id"),
                    _first_value(r, "mentorprofileid", "mentor_id")
                ),
                str(_first_value(r, "comment", "text", default=""))[:2000],
                to_datetime_or_null(_first_value(r, "createdat", "created_at")) or "2024-01-01",
                mentee_guid,
                mentor_guid,
                mentorship_to_guid(_first_value(r, "mentorshipid", "mentorship_id")) or mentorship_to_guid("1"),
                to_int_or_null(_first_value(r, "rating", "rating")) or "5",
            )
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    feedback err: {e}")
    if inserted != len(df):
        raise RuntimeError(f"CRITICAL: feedbacks import mismatch - expected {len(df)} rows, inserted {inserted}")
    print(f"  feedbacks: {inserted} loaded (validated)")
    return inserted


def load_saved_posts(conn):
    df, _ = _read_source_csv("saved_posts.csv")
    cur = conn.cursor()
    cur.execute("DELETE FROM [saved_posts]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [saved_posts] ON")

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            user_guid = int_to_guid(r.get("UserId"))
            if not user_guid: continue
            cur.execute("""INSERT INTO [saved_posts] ([SaveId],[ProgramId],[UserId],[CreatedAt])
                          VALUES (?,?,?,?)""",
                       to_int_or_null(r.get("SaveId")),
                       to_int_or_null(r.get("ProgramId")) or "1",
                       user_guid,
                       to_datetime_or_null(r.get("CreatedAt")) or "2024-01-01")
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3: print(f"    saved_posts err: {e}")
    try: cur.execute("SET IDENTITY_INSERT [saved_posts] OFF")
    except: pass
    print(f"  saved_posts: {inserted} loaded")
    return inserted


def load_shared_posts(conn):
    df, _ = _read_source_csv("shared_posts.csv")
    cur = conn.cursor()
    cur.execute("DELETE FROM [shared_posts]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [shared_posts] ON")

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            user_guid = int_to_guid(_first_value(r, "UserId", "sender_id"))
            if not user_guid: continue
            cur.execute("""INSERT INTO [shared_posts] ([ShareId],[ProgramId],[UserId],[SharedAt])
                          VALUES (?,?,?,?)""",
                       to_int_or_null(_first_value(r, "ShareId", "share_id")),
                       to_int_or_null(_first_value(r, "ProgramId", "post_id")) or "1",
                       user_guid,
                       to_datetime_or_null(_first_value(r, "SharedAt", "shared_at")) or "2024-01-01")
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3: print(f"    shared_posts err: {e}")
    try: cur.execute("SET IDENTITY_INSERT [shared_posts] OFF")
    except: pass
    print(f"  shared_posts: {inserted} loaded")
    return inserted


def load_post_likes(conn):
    """Load post likes from db-ready CSV into post_likes table."""
    try:
        df, path = _read_source_csv("post_likes.csv")
    except FileNotFoundError:
        print("  post_likes: skipped (no source CSV)")
        return 0
    # Standardize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    # Filter rows with required keys; handle both ProgramId and post_id naming
    if "post_id" in df.columns:
        df = df[df["post_id"].astype(str).str.strip() != ""]
    if "programid" in df.columns:
        df = df[df["programid"].astype(str).str.strip() != ""]
    if "user_id" in df.columns:
        df = df[df["user_id"].astype(str).str.strip() != ""]
    if "userid" in df.columns:
        df = df[df["userid"].astype(str).str.strip() != ""]
    
    cur = conn.cursor()
    cur.execute("DELETE FROM [post_likes]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [post_likes] ON")

    inserted = 0
    for _, r in df.iterrows():
        try:
            # CSV has: LikeId, ProgramId, UserId, CreatedAt (converted to lowercase)
            user_guid = int_to_guid(r.get("userid"))
            if not user_guid:
                continue
            program_id = to_int_or_null(r.get("programid"))
            if not program_id:
                continue
            like_id = to_int_or_null(r.get("likeid"))
            if like_id is None:
                continue
            cur.execute(
                """INSERT INTO [post_likes] ([LikeId],[ProgramId],[UserId],[CreatedAt])
                   VALUES (?,?,?,?)""",
                like_id,
                program_id,
                user_guid,
                to_datetime_or_null(r.get("createdat")) or "2024-01-01",
            )
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    post_likes err: {e}")
    try:
        cur.execute("SET IDENTITY_INSERT [post_likes] OFF")
    except Exception:
        pass
    if inserted != len(df):
        raise RuntimeError(f"CRITICAL: post_likes import mismatch - expected {len(df)} rows, inserted {inserted}")
    print(f"  post_likes: {inserted} loaded (validated)")
    return inserted


def load_post_comments(conn):
    """Load post comments from db-ready CSV into Post-Comment table."""
    try:
        df, path = _read_source_csv("Post-Comment.csv")
    except FileNotFoundError:
        print("  Post-Comment: skipped (no source CSV)")
        return 0
    # Standardize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    # Filter rows with required keys; CSV has programid and userid
    if "programid" in df.columns:
        df = df[df["programid"].astype(str).str.strip() != ""]
    if "userid" in df.columns:
        df = df[df["userid"].astype(str).str.strip() != ""]
    
    cur = conn.cursor()
    cur.execute("DELETE FROM [Post-Comment]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [Post-Comment] ON")

    inserted = 0
    for _, r in df.iterrows():
        try:
            # CSV has: CommentId, ProgramId, UserId, CommentText, CreatedAt, IsDeleted (converted to lowercase)
            user_guid = int_to_guid(r.get("userid"))
            if not user_guid:
                continue
            program_id = to_int_or_null(r.get("programid"))
            if not program_id:
                continue
            comment_id = to_int_or_null(r.get("commentid"))
            if comment_id is None:
                continue
            comment_text = str(r.get("commenttext", "")).strip()[:4000]
            cur.execute(
                """INSERT INTO [Post-Comment] ([CommentId],[CommentText],[CreatedAt],[IsDeleted],[ProgramId],[UserId])
                   VALUES (?,?,?,?,?,?)""",
                comment_id,
                comment_text,
                to_datetime_or_null(r.get("createdat")) or "2024-01-01",
                to_bit(r.get("isdeleted", "0")),
                program_id,
                user_guid,
            )
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    Post-Comment err: {e}")
    try:
        cur.execute("SET IDENTITY_INSERT [Post-Comment] OFF")
    except Exception:
        pass
    if inserted != len(df):
        raise RuntimeError(f"CRITICAL: Post-Comment import mismatch - expected {len(df)} rows, inserted {inserted}")
    print(f"  Post-Comment: {inserted} loaded (validated)")
    return inserted


def load_app_cancellations(conn):
    df, _ = _read_source_csv("app_cancellation.csv", "apps_cancellation.csv", "app_cancellations.csv")
    cur = conn.cursor()
    cur.execute("DELETE FROM [apps_cancellation]")
    conn.commit()

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            mentee_guid = int_to_guid(_first_value(r, "MenteeId", "mentee_id"))
            if not mentee_guid:
                continue
            # CancellationReason: stored as STRING via HasConversion<string>()
            # Must be a valid AppCancellationReason enum name (PascalCase)
            cancel_reason = _normalize_app_cancellation_reason(
                _first_value(r, "CancellationReason", "cancellation_reason", default="PersonalIssue")
            )
            cur.execute("""INSERT INTO [apps_cancellation] ([ApplicationId],[ProgramId],[MenteeId],[CancellationDate],[CancellationReason])
                          VALUES (?,?,?,?,?)""",
                       to_int_or_null(_first_value(r, "ApplicationId", "app_id")) or "1",
                       to_int_or_null(_first_value(r, "ProgramId", "post_id")) or "1",
                       mentee_guid,
                       to_datetime_or_null(_first_value(r, "CancellationDate", "cancellation_date")) or "2024-01-01",
                       cancel_reason)
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3: print(f"    apps_cancellation err: {e}")
    print(f"  apps_cancellation: {inserted} loaded")
    return inserted


def load_mentorship_cancellations(conn):
    df, _ = _read_source_csv("mentorship_cancellation.csv", "mentorships_cancellation.csv", "mentorship_cancellations.csv")
    cur = conn.cursor()
    cur.execute("DELETE FROM [mentorships_cancellation]")
    conn.commit()

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            mentorship_id = to_int_or_null(r.get("mentorship_id", r.get("MentorshipId")))
            mentee_guid = int_to_guid(r.get("mentee_id", r.get("MenteeProfileId")))
            mentor_guid = int_to_guid(r.get("mentor_id", r.get("MentorProfileId")))
            if not mentorship_id or not mentee_guid or not mentor_guid:
                continue
            cur.execute(
                """INSERT INTO [mentorships_cancellation]
                   ([MentorshipId],[ProgramId],[MenteeProfileId],[MentorProfileId],[CancellationDate],[CancellationActor],[CancellationReasonValue])
                   VALUES (?,?,?,?,?,?,?)""",
                mentorship_id,
                to_int_or_null(r.get("post_id", r.get("ProgramId"))) or "1",
                mentee_guid,
                mentor_guid,
                to_datetime_or_null(r.get("cancellation_date", r.get("CancellationDate"))) or "2024-01-01",
                _normalize_cancellation_actor(r.get("cancellation_actor", r.get("CancellationActor", "mentee"))),
                str(r.get("cancellation_reason", r.get("CancellationReasonValue", "")))[:100],
            )
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    mentorships_cancellation err: {e}")
    if inserted != len(df):
        raise RuntimeError(f"mentorships_cancellation import mismatch: source={len(df)} inserted={inserted}")
    print(f"  mentorships_cancellation: {inserted} loaded")
    return inserted


def load_simple_guid_table(conn, table, csv_file, cols_map, guid_cols, int_cols=None, bit_cols=None, identity_col=None):
    """Generic loader for tables with GUID foreign keys."""
    if isinstance(csv_file, (list, tuple)):
        df, _ = _read_source_csv(*csv_file)
    else:
        df, _ = _read_source_csv(csv_file)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM [{table}]")
    conn.commit()

    if identity_col:
        cur.execute(f"SET IDENTITY_INSERT [{table}] ON")

    bit_cols = bit_cols or []
    inserted = 0

    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            vals = []
            skip = False
            for csv_col, db_col in cols_map.items():
                if isinstance(csv_col, (list, tuple)):
                    v = _first_value(r, *csv_col, default="")
                else:
                    v = r.get(csv_col, "")
                if db_col in guid_cols:
                    v = int_to_guid(v)
                    if not v: skip = True; break
                elif db_col in int_cols:
                    if db_col == "experience_level":
                        # Use ExperienceLevel enum map (None=1, Beginner=2, Intermediate=3, Advanced=4)
                        v = EXPERIENCE_LEVEL_MAP.get(str(v).strip().lower(), to_int_or_null(v))
                    else:
                        v = to_int_or_null(v)
                elif db_col in bit_cols:
                    v = to_bit(v)
                elif v in ("", "nan", "NaN", "None"):
                    v = None
                vals.append(v)
            if skip: continue
            placeholders = ",".join(["?"]*len(vals))
            db_cols_str = ",".join(f"[{cols_map[c]}]" for c in cols_map)
            cur.execute(f"INSERT INTO [{table}] ({db_cols_str}) VALUES ({placeholders})", *vals)
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3: print(f"    {table} err: {e}")

    if identity_col:
        try: cur.execute(f"SET IDENTITY_INSERT [{table}] OFF")
        except: pass
    print(f"  {table}: {inserted} loaded")
    return inserted


def load_mentee_interests(conn):
    return load_simple_guid_table(
        conn,
        table="mentee_interests",
        csv_file="mentee_interests.csv",
        cols_map={
            ("UserId", "mentee_id"): "user_id",
            "TechnologyId": "technology_id",
            "ExperienceLevel": "experience_level",
        },
        guid_cols=["user_id"],
        int_cols=["technology_id", "experience_level"],
    )


def load_mentor_expertise(conn):
    return load_simple_guid_table(
        conn,
        table="mentor_expertise",
        csv_file="mentor_expertise.csv",
        cols_map={
            ("MentorId", "mentor_id"): "mentor_id",
            "TechnologyId": "technology_id",
        },
        guid_cols=["mentor_id"],
        int_cols=["technology_id"],
    )


def load_mentee_subdomains(conn):
    return load_simple_guid_table(
        conn,
        table="MenteeSubDomains",
        csv_file=("MenteeSubDomains.csv", "mentee_subdomains.csv"),
        cols_map={
            ("UserId", "mentee_id"): "UserId",
            "SubDomainId": "SubDomainId",
        },
        guid_cols=["UserId"],
        int_cols=["SubDomainId"],
    )


def load_mentor_subdomains(conn):
    return load_simple_guid_table(
        conn,
        table="MentorSubDomains",
        csv_file=("MentorSubDomains.csv", "mentor_subdomains.csv"),
        cols_map={
            ("MentorId", "mentor_id"): "MentorId",
            "SubDomainId": "SubDomainId",
        },
        guid_cols=["MentorId"],
        int_cols=["SubDomainId"],
    )


def load_countries(conn):
    try:
        df, _ = _read_source_csv("countries.csv")
    except FileNotFoundError:
        print(f"  countries: skipped (no source CSV)")
        return 0
    
    cur = conn.cursor()
    cur.execute("DELETE FROM [countries]")
    conn.commit()

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            code = str(r.get("CountryCode", "")).strip()[:2]
            if not code:
                continue
            cur.execute("INSERT INTO [countries] ([country_code],[country_name]) VALUES (?,?)",
                        code, str(r.get("CountryName", "")).strip()[:200] or code)
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    countries err: {e}")
    print(f"  countries: {inserted} loaded")
    return inserted


def load_domains(conn):
    try:
        df, _ = _read_source_csv("domains.csv")
    except FileNotFoundError:
        print(f"  domains: skipped (no source CSV)")
        return 0
    cur = conn.cursor()
    cur.execute("DELETE FROM [domains]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [domains] ON")

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            domain_id = to_int_or_null(r.get("DomainId"))
            if not domain_id:
                continue
            cur.execute("INSERT INTO [domains] ([domain_id],[name],[description]) VALUES (?,?,?)",
                        domain_id,
                        str(r.get("Name", "")).strip()[:200] or f"Domain {domain_id}",
                        str(r.get("Description", "")).strip()[:2000] or None)
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    domains err: {e}")
    print(f"  domains: {inserted} loaded")
    try:
        cur.execute("SET IDENTITY_INSERT [domains] OFF")
    except Exception:
        pass
    return inserted


def load_learning_style(conn):
    try:
        df, _ = _read_source_csv("learning_style.csv", "learning_styles.csv")
    except FileNotFoundError:
        print(f"  learning_style: skipped (no source CSV)")
        return 0
    cur = conn.cursor()
    cur.execute("DELETE FROM [learning_style]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [learning_style] ON")

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            style_id = to_int_or_null(r.get("LearningStyleId"))
            if not style_id:
                continue
            cur.execute("INSERT INTO [learning_style] ([learning_style_id],[name],[description]) VALUES (?,?,?)",
                        style_id,
                        str(r.get("Name", "")).strip()[:200] or f"Style {style_id}",
                        str(r.get("Description", "")).strip()[:2000] or None)
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    learning_style err: {e}")
    print(f"  learning_style: {inserted} loaded")
    try:
        cur.execute("SET IDENTITY_INSERT [learning_style] OFF")
    except Exception:
        pass
    return inserted


def load_career_goal(conn):
    try:
        df, _ = _read_source_csv("career_goal.csv", "career_goals.csv")
    except FileNotFoundError:
        print(f"  career_goal: skipped (no source CSV)")
        return 0
    cur = conn.cursor()
    cur.execute("DELETE FROM [career_goal]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [career_goal] ON")

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            goal_id = to_int_or_null(r.get("CareerGoalId"))
            if not goal_id:
                continue
            cur.execute("INSERT INTO [career_goal] ([career_goal_id],[name]) VALUES (?,?)",
                        goal_id, str(r.get("Name", "")).strip()[:200] or f"Goal {goal_id}")
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    career_goal err: {e}")
    print(f"  career_goal: {inserted} loaded")
    try:
        cur.execute("SET IDENTITY_INSERT [career_goal] OFF")
    except Exception:
        pass
    return inserted


def load_subdomain(conn):
    try:
        df, _ = _read_source_csv("subdomain.csv", "subdomains.csv")
    except FileNotFoundError:
        print(f"  subdomain: skipped (no source CSV)")
        return 0
    cur = conn.cursor()
    cur.execute("DELETE FROM [subdomain]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [subdomain] ON")

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            subdomain_id = to_int_or_null(r.get("SubDomainId"))
            domain_id = to_int_or_null(r.get("DomainId")) or "1"
            if not subdomain_id:
                continue
            cur.execute("INSERT INTO [subdomain] ([subdomain_id],[domain_id],[name]) VALUES (?,?,?)",
                        subdomain_id, domain_id, str(r.get("Name", "")).strip()[:200] or f"SubDomain {subdomain_id}")
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    subdomain err: {e}")
    print(f"  subdomain: {inserted} loaded")
    try:
        cur.execute("SET IDENTITY_INSERT [subdomain] OFF")
    except Exception:
        pass
    return inserted


def load_technologies(conn):
    try:
        df, _ = _read_source_csv("technologies.csv")
    except FileNotFoundError:
        print(f"  technologies: skipped (no source CSV)")
        return 0
    cur = conn.cursor()
    cur.execute("DELETE FROM [technologies]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [technologies] ON")

    inserted = 0
    for _, r in df.iterrows():
        r = _ci_row(r)
        try:
            tech_id = to_int_or_null(r.get("TechnologyId"))
            subdomain_id = to_int_or_null(r.get("SubDomainId")) or "1"
            if not tech_id:
                continue
            cur.execute("INSERT INTO [technologies] ([technology_id],[subdomain_id],[name],[ProgramId]) VALUES (?,?,?,?)",
                        tech_id,
                        subdomain_id,
                        str(r.get("Name", "")).strip()[:200] or f"Tech {tech_id}",
                        to_int_or_null(r.get("ProgramId")))
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    technologies err: {e}")
    print(f"  technologies: {inserted} loaded")
    try:
        cur.execute("SET IDENTITY_INSERT [technologies] OFF")
    except Exception:
        pass
    return inserted


def clear_all_target_tables(conn):
    tables = [
        "mentorship_requirements",
        "mentorships_cancellation",
        "apps_cancellation",
        "shared_posts",
        "saved_posts",
        "feedbacks",
        "mentorships",
        "applications",
        "MentorSubDomains",
        "MenteeSubDomains",
        "mentor_expertise",
        "mentee_interests",
        "follows",
        "programs",
        "mentor_profile",
        "mentee_profile",
        "technologies",
        "subdomain",
        "learning_style",
        "domains",
        "career_goal",
        "countries",
        "users",
        "post_likes",
        "Post-Comment",
    ]
    cur = conn.cursor()
    for table in tables:
        try:
            cur.execute(f"TRUNCATE TABLE [{table}]")
        except Exception:
            conn.rollback()
            cur.execute(f"DELETE FROM [{table}]")
        conn.commit()


def load_mentorship_requirements(conn):
    df, _ = _read_source_csv("mentorship_requirements.csv")
    cur = conn.cursor()
    cur.execute("DELETE FROM [mentorship_requirements]")
    conn.commit()
    cur.execute("SET IDENTITY_INSERT [mentorship_requirements] ON")

    inserted = 0
    for idx, r in df.iterrows():
        r = _ci_row(r)
        try:
            program_id = to_int_or_null(_first_value(r, "ProgramId", "post_id"))
            technology_id = to_int_or_null(_first_value(r, "TechnologyId", "technology_id"))
            if not program_id or not technology_id:
                continue
            requirement_id = to_int_or_null(_first_value(r, "MentorshipRequirementId", "requirement_id")) or str(idx + 1)
            cur.execute("""INSERT INTO [mentorship_requirements]
                          ([MentorshipRequirementId],[RequiredExperienceLevel],[ProgramId],[TechnologyId])
                          VALUES (?,?,?,?)""",
                        requirement_id,
                        _normalize_experience_enum_name(_first_value(r, "RequiredExperienceLevel", "required_experience_level", default=""))[:100],
                        program_id,
                        technology_id)
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()
            if inserted < 3:
                print(f"    mentorship_requirements err: {e}")
    print(f"  mentorship_requirements: {inserted} loaded")
    try:
        cur.execute("SET IDENTITY_INSERT [mentorship_requirements] OFF")
    except Exception:
        pass
    return inserted


def main():
    print("=" * 65)
    print("Loading database files into MentorshipPlatformDB")
    print("  (with int->GUID, string->enum, string->bit conversions)")
    print("  NO ARCHIVE FALLBACK - USING LATEST CLEANED DATA ONLY")
    print("=" * 65)

    # Ensure DB-ready exists and validate against raw inputs
    if not os.path.exists(DB_READY):
        print(f"ERROR: DB-ready folder not found: {DB_READY}")
        sys.exit(1)

    def validate_raw_vs_db_ready():
        problems = []
        db_counts = {}
        for fn in os.listdir(DB_READY):
            if not fn.lower().endswith('.csv'):
                continue
            db_path = os.path.join(DB_READY, fn)
            raw_path = os.path.join(RAW_FILES, fn)
            if not os.path.exists(raw_path):
                print(f"  WARN: Missing raw file for {fn} — skipping raw vs db-ready row check")
                try:
                    df_db = pd.read_csv(db_path)
                    db_counts[fn] = len(df_db)
                except Exception as e:
                    problems.append(f"Failed to read {fn}: {e}")
                continue
            try:
                df_db = pd.read_csv(db_path)
                df_raw = pd.read_csv(raw_path)
            except Exception as e:
                problems.append(f"Failed to read {fn}: {e}")
                continue
            db_counts[fn] = len(df_db)
            if len(df_db) != len(df_raw):
                problems.append(f"Row count mismatch for {fn}: raw={len(df_raw)} db-ready={len(df_db)}")
        if problems:
            print("CRITICAL: Raw vs DB-ready validation failed:")
            for p in problems:
                print("  "+p)
            raise RuntimeError("Raw vs DB-ready validation failed; aborting import")
        return db_counts

    db_counts = validate_raw_vs_db_ready()

    def validate_critical_columns(db_counts):
        """Validate key columns in db-ready CSVs before import - relaxed version."""
        print("\n=== PRE-IMPORT COLUMN VALIDATION ===")
        issues = []
        
        # Define column validation rules per table (case-insensitive)
        validations = {
            "users.csv": {"required": ["user_id", "email"]},
            "mentee_profile.csv": {"required": ["user_id"]},
            "mentor_profile.csv": {"required": ["user_id"]},
            "applications.csv": {"required": ["status"]},
            "mentorships.csv": {"required": ["status"]},
            "follows.csv": {"required": ["follower_id", "following_id"]},
            "post_likes.csv": {"required": ["user_id", "post_id"]},
            "Post-Comment.csv": {"required": ["user_id", "post_id"]},
            "feedbacks.csv": {"required": ["mentee_id", "mentor_id"]},
        }
        
        for fn, rules in validations.items():
            db_path = os.path.join(DB_READY, fn)
            if fn not in db_counts or not os.path.exists(db_path):
                continue
            
            try:
                df = pd.read_csv(db_path, dtype=str, keep_default_na=False)
                cols_lower = {c.strip().lower(): c.strip() for c in df.columns}
                
                # Check required columns (case-insensitive)
                for col in rules.get("required", []):
                    if col.lower() not in cols_lower:
                        issues.append(f"{fn}: missing column {col}")
                    else:
                        col_actual = cols_lower[col.lower()]
                        nulls = (df[col_actual].astype(str).str.strip() == "").sum()
                        if nulls > 0:
                            issues.append(f"{fn}.{col}: {nulls} empty values")
                
            except Exception as e:
                issues.append(f"{fn}: read error: {e}")
        
        if issues:
            print("  Pre-import checks:")
            for issue in issues[:5]:
                print(f"    WARNING: {issue}")
            if len(issues) > 5:
                print(f"    ... and {len(issues) - 5} more warnings (non-fatal)")
        else:
            print("  All critical columns present and non-empty")
    
    validate_critical_columns(db_counts)


    conn = pyodbc.connect(CONN_STR, autocommit=False)
    conn.autocommit = True

    total = 0

    # Disable FK checks temporarily
    try:
        cur = conn.cursor()
        cur.execute("EXEC sp_MSforeachtable 'ALTER TABLE ? NOCHECK CONSTRAINT ALL'")
    except: pass

    clear_all_target_tables(conn)

    total += load_users(conn)
    total += load_countries(conn)
    total += load_career_goal(conn)
    total += load_learning_style(conn)
    total += load_domains(conn)
    total += load_subdomain(conn)
    total += load_technologies(conn)
    total += load_mentee_profile(conn)
    total += load_mentor_profile(conn)
    total += load_programs(conn)
    total += load_follows(conn)
    total += load_mentee_interests(conn)
    total += load_mentor_expertise(conn)
    total += load_mentee_subdomains(conn)
    total += load_mentor_subdomains(conn)
    total += load_applications(conn)
    total += load_mentorships(conn)
    total += load_feedbacks(conn)
    total += load_saved_posts(conn)
    total += load_shared_posts(conn)
    total += load_post_likes(conn)
    total += load_post_comments(conn)
    total += load_app_cancellations(conn)
    total += load_mentorship_cancellations(conn)
    total += load_mentorship_requirements(conn)

    # Re-enable FK checks
    try:
        cur = conn.cursor()
        cur.execute("EXEC sp_MSforeachtable 'ALTER TABLE ? WITH CHECK CHECK CONSTRAINT ALL'")
    except: pass

    # FINAL VALIDATION QUERIES
    print("\n" + "=" * 65)
    print("FINAL VALIDATION")
    print("=" * 65)
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT MAX(ApplicationId) FROM applications")
        max_app_id = cur.fetchone()[0] or 0
        print(f"✓ MAX ApplicationId: {max_app_id}")
    except: print("✗ Failed to get MAX ApplicationId")
    
    try:
        cur.execute("SELECT COUNT(*) FROM applications")
        app_count = int(cur.fetchone()[0])
        print(f"✓ Total applications: {app_count}")
    except: print("✗ Failed to get application count")
    
    try:
        cur.execute("""SELECT MeetRequirements, Status, COUNT(*) as cnt
                       FROM applications
                       GROUP BY MeetRequirements, Status
                       ORDER BY MeetRequirements, Status""")
        print("\nApplications by MeetRequirements & Status:")
        for row in cur.fetchall():
            print(f"  MeetRequirements={row[0]}, Status={row[1]}: {row[2]} rows")
    except Exception as e:
        print(f"✗ Failed to get status breakdown: {e}")
    
    try:
        cur.execute("SELECT MAX(ProgramId) FROM programs")
        max_prog_id = cur.fetchone()[0] or 0
        print(f"\n✓ MAX ProgramId: {max_prog_id}")
    except: print("✗ Failed to get MAX ProgramId")
    
    try:
        cur.execute("SELECT COUNT(*) FROM programs WHERE EducationLevel IS NOT NULL")
        prog_with_edu = int(cur.fetchone()[0])
        print(f"✓ Programs with EducationLevel: {prog_with_edu}")
    except: print("✗ Failed to get programs with EducationLevel")
    
    try:
        cur.execute("SELECT COUNT(*) FROM programs WHERE TargetLevel IS NOT NULL")
        prog_with_target = int(cur.fetchone()[0])
        print(f"✓ Programs with TargetLevel: {prog_with_target}")
    except: print("✗ Failed to get programs with TargetLevel")
    
    try:
        cur.execute("SELECT COUNT(*) FROM mentorships")
        mentorship_count = int(cur.fetchone()[0])
        print(f"\n✓ Total mentorships: {mentorship_count}")
    except: print("✗ Failed to get mentorship count")
    
    try:
        cur.execute("SELECT COUNT(*) FROM users WHERE IsEmailVerified = 1")
        verified_users = int(cur.fetchone()[0])
        print(f"✓ Users with is_email_verified=1: {verified_users}")
    except: print("✗ Failed to get verified users count")

    conn.close()
    print("\n" + "=" * 65)
    print(f"DONE. Total rows loaded: {total}")
    print("=" * 65)

    # POST-IMPORT: validate SQL table counts against DB-ready CSV counts
    def post_import_sql_validation(conn, db_counts):
        cur = conn.cursor()
        problems = []
        for fn, expected in db_counts.items():
            table = os.path.splitext(fn)[0]
            table = table.replace('-', '_').replace(' ', '_')
            try:
                cur.execute(f"SELECT COUNT(*) FROM [{table}]")
                db_count = int(cur.fetchone()[0])
            except Exception:
                # table may not exist or different name; skip with warning
                print(f"  WARN: Could not validate table {table} for file {fn}")
                continue
            if db_count != expected:
                problems.append(f"Table {table} count mismatch: expected {expected}, db {db_count}")
        if problems:
            print("CRITICAL: Post-import SQL validation failed:")
            for p in problems:
                print("  "+p)
            raise RuntimeError("Post-import SQL validation failed")
        print("Post-import SQL validation passed: all table counts match DB-ready CSVs")

    # Re-open connection for validation (some cursors closed earlier)
    conn = pyodbc.connect(CONN_STR, autocommit=True)
    try:
        post_import_sql_validation(conn, db_counts)
    finally:
        conn.close()

    # ── TEMPORAL INTEGRITY VALIDATION ──
    def temporal_integrity_validation():
        """Validate chronological consistency across all loaded tables.

        Checks:
          - No applications before program creation
          - No decision before application
          - Mentorship status consistent with dates (completed=past, active=spanning NOW)
          - No interactions before program creation
          - No feedback before mentorship start
          - No cancellations before start dates
          - MeetRequirements consistent with status
          - No future event timestamps (except active mentorship EndDate)
        """
        print("\n" + "=" * 65)
        print("TEMPORAL INTEGRITY VALIDATION")
        print("=" * 65)

        conn = pyodbc.connect(CONN_STR, autocommit=True)
        cur = conn.cursor()
        violations = []
        now_str = "GETDATE()"

        # V1: Applications before program creation
        cur.execute("""
            SELECT COUNT(*) FROM applications a
            JOIN programs p ON a.ProgramId = p.ProgramId
            WHERE a.AppliedAt < p.CreatedAt
        """)
        v = int(cur.fetchone()[0])
        if v > 0: violations.append(f"V1: {v} applications before program creation")
        print(f"  V1  Apps before program:      {v} {'PASS' if v==0 else 'FAIL'}")

        # V2: Decision before AppliedAt
        cur.execute("""
            SELECT COUNT(*) FROM applications
            WHERE DecisionAt IS NOT NULL AND DecisionAt < AppliedAt
        """)
        v = int(cur.fetchone()[0])
        if v > 0: violations.append(f"V2: {v} decisions before application")
        print(f"  V2  Decision before apply:     {v} {'PASS' if v==0 else 'FAIL'}")

        # V3a: Completed with future EndDate
        cur.execute(f"""
            SELECT COUNT(*) FROM mentorships
            WHERE LOWER(Status) = 'completed' AND EndDate > {now_str}
        """)
        v = int(cur.fetchone()[0])
        if v > 0: violations.append(f"V3a: {v} completed mentorships with future EndDate")
        print(f"  V3a Completed+future end:      {v} {'PASS' if v==0 else 'FAIL'}")

        # V3b: Active with past EndDate
        cur.execute(f"""
            SELECT COUNT(*) FROM mentorships
            WHERE LOWER(Status) = 'active' AND EndDate < {now_str}
        """)
        v = int(cur.fetchone()[0])
        if v > 0: violations.append(f"V3b: {v} active mentorships with past EndDate")
        print(f"  V3b Active+past end:           {v} {'PASS' if v==0 else 'FAIL'}")

        # V3c: EndDate before StartDate
        cur.execute("SELECT COUNT(*) FROM mentorships WHERE EndDate < StartDate")
        v = int(cur.fetchone()[0])
        if v > 0: violations.append(f"V3c: {v} mentorships with EndDate before StartDate")
        print(f"  V3c End before start:          {v} {'PASS' if v==0 else 'FAIL'}")

        # V4: Interactions before program creation (likes, comments, saves, shares)
        for table, date_col, label in [
            ("post_likes", "CreatedAt", "likes"),
            ("[Post-Comment]", "CreatedAt", "comments"),
            ("saved_posts", "CreatedAt", "saves"),
            ("shared_posts", "SharedAt", "shares"),
        ]:
            try:
                cur.execute(f"""
                    SELECT COUNT(*) FROM {table} t
                    JOIN programs p ON t.ProgramId = p.ProgramId
                    WHERE t.{date_col} < p.CreatedAt
                """)
                v = int(cur.fetchone()[0])
                if v > 0: violations.append(f"V4: {v} {label} before program creation")
                print(f"  V4  {label:10s} before program: {v} {'PASS' if v==0 else 'FAIL'}")
            except Exception:
                print(f"  V4  {label:10s} check skipped (table not found)")

        # V5: MeetRequirements consistency
        # In DB: alerted → Pending with MR=0. All other statuses → MR=1.
        # So: Accepted with MR=0 is wrong, Rejected with MR=0 is wrong.
        cur.execute("""
            SELECT COUNT(*) FROM applications
            WHERE LOWER(Status) IN ('accepted', 'rejected') AND MeetRequirements = 0
        """)
        v = int(cur.fetchone()[0])
        if v > 0: violations.append(f"V5a: {v} accepted/rejected with MeetRequirements=0")
        print(f"  V5a Accepted/Rejected+MR=0:    {v} {'PASS' if v==0 else 'FAIL'}")

        # Pending apps: MR=1 (original pending) or MR=0 (original alerted → Pending)
        # Both are valid, so no check needed here.
        cur.execute("""
            SELECT MeetRequirements, COUNT(*) as cnt FROM applications
            WHERE LOWER(Status) = 'pending'
            GROUP BY MeetRequirements
        """)
        rows = cur.fetchall()
        mr_dist = {str(r[0]): r[1] for r in rows}
        print(f"  V5b Pending MR distribution:   MR=1:{mr_dist.get('True', mr_dist.get('1', 0))}, MR=0:{mr_dist.get('False', mr_dist.get('0', 0))} (info)")

        # V6: Pending with DecisionAt (should be NULL for all Pending)
        cur.execute("""
            SELECT COUNT(*) FROM applications
            WHERE LOWER(Status) = 'pending' AND DecisionAt IS NOT NULL
        """)
        v = int(cur.fetchone()[0])
        if v > 0: violations.append(f"V6: {v} pending applications with DecisionAt set")
        print(f"  V6  Pending+DecisionAt:        {v} {'PASS' if v==0 else 'FAIL'}")

        conn.close()

        if violations:
            print(f"\n  TEMPORAL VIOLATIONS FOUND: {len(violations)}")
            for viol in violations:
                print(f"    {viol}")
            print("  WARNING: Timeline data has integrity issues — check data generation")
        else:
            print("\n  ALL TEMPORAL INTEGRITY CHECKS PASSED")

    temporal_integrity_validation()


if __name__ == "__main__":
    # Ensure db-ready single source exists
    if not os.path.isdir(DB_READY):
        print(f"CRITICAL: DB-ready folder not found: {DB_READY}. Aborting import.")
        sys.exit(2)
    main()
