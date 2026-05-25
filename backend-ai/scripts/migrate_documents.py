"""Database Migration — Create tables for semantic document storage.

Run this migration to set up document storage:
    python scripts/migrate_documents.py
"""

import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import database
from sqlalchemy import text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Migration Queries
# ─────────────────────────────────────────────────────────────────────

CREATE_DOCUMENTS_TABLE = """
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = N'documents')
BEGIN
    CREATE TABLE documents (
        document_id INT PRIMARY KEY IDENTITY(1,1),
        user_id UNIQUEIDENTIFIER NOT NULL,
        role NVARCHAR(20) NOT NULL,
        file_name NVARCHAR(255) NOT NULL,
        file_type NVARCHAR(20) NOT NULL,
        file_path NVARCHAR(500) NOT NULL,
        file_size INT NOT NULL,
        upload_status NVARCHAR(20) DEFAULT 'pending',
        extracted_successfully BIT DEFAULT 0,
        extraction_error NVARCHAR(1000),
        chunk_count INT DEFAULT 0,
        created_at DATETIME DEFAULT GETDATE(),
        updated_at DATETIME DEFAULT GETDATE()
    );
    
    CREATE INDEX idx_user_role ON documents(user_id, role);
    CREATE INDEX idx_upload_status ON documents(upload_status);
END
"""

CREATE_DOCUMENT_CHUNKS_TABLE = """
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = N'document_chunks')
BEGIN
    CREATE TABLE document_chunks (
        chunk_id INT PRIMARY KEY IDENTITY(1,1),
        document_id INT NOT NULL,
        chunk_text NVARCHAR(MAX) NOT NULL,
        chunk_index INT NOT NULL,
        token_count INT,
        embedding_json NVARCHAR(MAX),
        created_at DATETIME DEFAULT GETDATE(),
        
        FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
    );
    
    CREATE INDEX idx_document ON document_chunks(document_id);
    CREATE INDEX idx_chunk_index ON document_chunks(document_id, chunk_index);
END
"""

# ─────────────────────────────────────────────────────────────────────
# Migration Runner
# ─────────────────────────────────────────────────────────────────────


def run_migration():
    """Run database migration."""
    try:
        logger.info("Starting document storage migration...")
        engine = database.get_engine()

        # Create documents table
        logger.info("Creating documents table...")
        with engine.connect() as connection:
            connection.execute(text(CREATE_DOCUMENTS_TABLE))
            connection.commit()
        logger.info("✓ Documents table created")

        # Create document_chunks table
        logger.info("Creating document_chunks table...")
        with engine.connect() as connection:
            connection.execute(text(CREATE_DOCUMENT_CHUNKS_TABLE))
            connection.commit()
        logger.info("✓ Document chunks table created")

        logger.info("✅ Migration completed successfully")
        return True

    except Exception as e:
        logger.error("❌ Migration failed: %s", e)
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    success = run_migration()
    sys.exit(0 if success else 1)
