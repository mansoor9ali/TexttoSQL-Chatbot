"""
Setup Schema Embeddings Table
This script creates the schema_embeddings table and ingests schemas from YAML
"""

import os
import yaml
import json
import re
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"
SCHEMA_FILE = "airplanes-flights-pg.yml"

def get_connection():
    """Create PostgreSQL connection"""
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        database=os.getenv("PGDATABASE", "airline_db"),
        user=os.getenv("PGUSER", "user-name"),
        password=os.getenv("PGPASSWORD", "strong-password"),
        port=int(os.getenv("PGPORT", "5432"))
    )

def create_schema_embeddings_table():
    """Create the schema_embeddings table if it doesn't exist"""
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            print("Creating pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            print("Creating schema_embeddings table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_embeddings (
                    id SERIAL PRIMARY KEY,
                    database_name VARCHAR(100),
                    table_name VARCHAR(100),
                    schema_text TEXT,
                    metadata JSONB,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            print("Creating index for faster similarity search...")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS schema_embeddings_vector_idx 
                ON schema_embeddings 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

        conn.commit()
        print("✓ Schema embeddings table created successfully!")

    except Exception as e:
        print(f"✗ Error creating table: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def load_schemas_from_yaml(yaml_file):
    """Load database schemas from YAML file"""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    schemas = []
    database_name = data.get('database', 'unknown')

    for idx, schema in enumerate(data.get('table_schemas', [])):
        # Extract table name from CREATE TABLE statement
        match = re.search(r'CREATE TABLE\s+(?:\w+\.)?(\w+)', schema, re.IGNORECASE)
        table_name = match.group(1) if match else f"table_{idx}"

        # Wrap schema in XML tags for better structure
        schema_text = f"<table_schema>\n{schema.strip()}\n</table_schema>"

        schemas.append({
            'database': database_name,
            'table_name': table_name,
            'schema_text': schema_text,
            'metadata': {
                'source': 'yaml',
                'database': database_name,
                'table_name': table_name
            }
        })

    print(f"✓ Loaded {len(schemas)} schemas from {yaml_file}")
    return schemas

def generate_embedding(client, text):
    """Generate embedding for given text using OpenAI"""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def ingest_schemas(schemas):
    """Ingest schemas with embeddings into PostgreSQL"""
    if not OPENAI_API_KEY:
        print("✗ OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)
    conn = get_connection()

    try:
        print(f"Ingesting {len(schemas)} schemas...")

        # Clear existing schemas for this database
        with conn.cursor() as cur:
            cur.execute("DELETE FROM schema_embeddings WHERE database_name = %s",
                       (schemas[0]['database'],))
        conn.commit()
        print(f"✓ Cleared existing schemas for database '{schemas[0]['database']}'")

        # Generate embeddings and insert
        for idx, schema in enumerate(schemas, 1):
            print(f"  Processing schema {idx}/{len(schemas)}: {schema['table_name']}...")
            embedding = generate_embedding(client, schema['schema_text'])

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO schema_embeddings 
                    (database_name, table_name, schema_text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                """, (
                    schema['database'],
                    schema['table_name'],
                    schema['schema_text'],
                    json.dumps(schema['metadata']),
                    '[' + ','.join(map(str, embedding)) + ']'
                ))

        conn.commit()
        print(f"✓ Successfully ingested {len(schemas)} schemas with embeddings!")

    except Exception as e:
        print(f"✗ Error ingesting schemas: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def main():
    """Main setup function"""
    print("="*70)
    print("Schema Embeddings Setup")
    print("="*70)

    try:
        # Step 1: Create table
        print("\n[1/3] Creating schema_embeddings table...")
        create_schema_embeddings_table()

        # Step 2: Load schemas
        print(f"\n[2/3] Loading schemas from {SCHEMA_FILE}...")
        schemas = load_schemas_from_yaml(SCHEMA_FILE)

        # Step 3: Ingest schemas with embeddings
        print("\n[3/3] Generating embeddings and ingesting schemas...")
        ingest_schemas(schemas)

        print("\n" + "="*70)
        print("✓ Setup completed successfully!")
        print("="*70)
        print("\nYou can now run the text-to-SQL system:")
        print("  python text2sql_pgvector.py interactive")

    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        print("\nPlease check:")
        print("  1. PostgreSQL server is running")
        print("  2. Database 'airline_db' exists")
        print("  3. pgvector extension is installed")
        print("  4. OPENAI_API_KEY is set in .env")
        print("  5. Database credentials in .env are correct")

if __name__ == "__main__":
    main()

