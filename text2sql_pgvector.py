"""
Text-to-SQL using OpenAI GPT and PostgreSQL with pgvector
========================================================
This script implements a text-to-SQL system that:
1. Stores database schemas in PostgreSQL using pgvector for embeddings
2. Uses semantic search to retrieve relevant table schemas
3. Generates SQL queries using OpenAI GPT models
4. Executes queries and provides natural language analysis of results

Based on: llama3-2-chromadb-text2sql.ipynb
Database: PostgreSQL (airline_db)
Tables: airplanes, flights
"""

import os
import re
import psycopg2
from openai import OpenAI
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"  # 1536 dimensions
LLM_MODEL = "gpt-4o"  # or "gpt-4", "gpt-3.5-turbo"
SCHEMA_FILE = "airplanes-flights-pg.yml"
TOP_K_SCHEMAS = 3  # Number of relevant schemas to retrieve


class PostgreSQLConnection:
    """Manages PostgreSQL database connections"""

    @staticmethod
    def get_connection():
        """Create and return a PostgreSQL connection"""
        return psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "airline_db"),
            user=os.getenv("PGUSER", "user-name"),
            password=os.getenv("PGPASSWORD", "strong-password"),
            port=int(os.getenv("PGPORT", "5432"))
        )


class SchemaEmbeddingStore:
    """Handles storage and retrieval of database schemas with embeddings in PostgreSQL"""

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.conn = PostgreSQLConnection.get_connection()
        logger.info("✓ Connected to schema embeddings store")


    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text using OpenAI"""
        response = self.client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )
        return response.data[0].embedding


    def retrieve_relevant_schemas(self, question: str, top_k: int = TOP_K_SCHEMAS) -> str:
        """Retrieve relevant schemas based on semantic similarity to the question"""
        # Generate embedding for the question
        question_embedding = self.generate_embedding(question)

        # Perform similarity search
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT schema_text, table_name, 
                       1 - (embedding <=> %s::vector) as similarity
                FROM schema_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (
                '[' + ','.join(map(str, question_embedding)) + ']',
                '[' + ','.join(map(str, question_embedding)) + ']',
                top_k
            ))

            results = cur.fetchall()

        # Combine schemas into a single string
        schemas = []
        for schema_text, table_name, similarity in results:
            logger.info(f"Retrieved schema for table '{table_name}' (similarity: {similarity:.4f})")
            schemas.append(schema_text)

        combined_schemas = "\n\n".join(schemas)
        return f"<table_schemas>\n{combined_schemas}\n</table_schemas>"

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class Text2SQLEngine:
    """Main engine for text-to-SQL conversion and execution"""

    def __init__(self, openai_client: OpenAI, schema_store: SchemaEmbeddingStore):
        self.client = openai_client
        self.schema_store = schema_store
        self.db_conn = PostgreSQLConnection.get_connection()

    def format_sql_prompt(self, question: str, table_schemas: str, db_type: str = "PostgreSQL") -> tuple:
        """Format the prompt for SQL generation"""
        system_prompt = f"""You are an expert {db_type} SQL query generator. Your task is to generate accurate SQL queries based on natural language questions.

**Instructions:**
1. Analyze the provided table schemas carefully
2. Generate a syntactically correct {db_type} SQL query that answers the user's question
3. Use proper joins when multiple tables are involved
4. Include appropriate WHERE clauses, GROUP BY, ORDER BY as needed
5. Enclose the SQL query within <sql></sql> XML tags
6. Do NOT include any explanations outside the SQL tags
7. Use table names with database prefix if specified in schema

**Table Schemas:**
{table_schemas}

**Important Notes:**
- Return ONLY the SQL query within <sql></sql> tags
- Ensure the query is executable and will return the requested information
- Use appropriate aggregate functions (COUNT, SUM, AVG, etc.) when needed
- For date/time comparisons, use proper {db_type} date functions"""

        user_prompt = f"**Question:** {question}\n\nGenerate the SQL query:"

        return system_prompt, user_prompt

    def generate_sql_query(self, question: str) -> str:
        """Generate SQL query from natural language question"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Question: {question}")
        logger.info(f"{'='*70}")

        # Step 1: Retrieve relevant schemas
        logger.info("\n[1/4] Retrieving relevant table schemas...")
        table_schemas = self.schema_store.retrieve_relevant_schemas(question)

        # Step 2: Generate SQL using LLM
        logger.info("\n[2/4] Generating SQL query using OpenAI GPT...")
        system_prompt, user_prompt = self.format_sql_prompt(question, table_schemas)

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        llm_response = response.choices[0].message.content

        # Extract SQL from response - try multiple patterns
        sql_query = None

        # Pattern 1: XML tags <sql></sql>
        sql_match = re.search(r'<sql>(.*?)</sql>', llm_response, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()

        # Pattern 2: Markdown code blocks ```sql ... ```
        if not sql_query:
            sql_match = re.search(r'```sql\s*(.*?)\s*```', llm_response, re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1).strip()

        # Pattern 3: Generic code blocks ``` ... ```
        if not sql_query:
            sql_match = re.search(r'```\s*(.*?)\s*```', llm_response, re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1).strip()

        # Pattern 4: Direct SQL (fallback)
        if not sql_query:
            sql_query = llm_response.strip()

        # Clean up the SQL query - remove any remaining markdown or extra formatting
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()

        logger.info(f"\n✓ Generated SQL:\n{sql_query}")
        return sql_query

    def execute_query(self, sql_query: str) -> Tuple[List[tuple], List[str]]:
        """Execute SQL query and return results"""
        logger.info("\n[3/4] Executing SQL query...")

        try:
            with self.db_conn.cursor() as cur:
                cur.execute(sql_query)
                results = cur.fetchall()

                # Get column names
                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    logger.info(f"✓ Query executed successfully. Columns: {columns}")
                    logger.info(f"✓ Retrieved {len(results)} row(s)")
                    return results, columns
                else:
                    return [], []

        except Exception as e:
            logger.error(f"✗ Error executing query: {e}")
            raise

    def analyze_results(self, question: str, sql_query: str, results: List[tuple], columns: List[str]) -> str:
        """Generate natural language analysis of query results"""
        logger.info("\n[4/4] Analyzing results with OpenAI GPT...")

        # Format results for the LLM
        if not results:
            results_text = "No results found."
        else:
            # Create a formatted table
            results_text = f"Columns: {', '.join(columns)}\n\n"
            for row in results[:10]:  # Limit to first 10 rows
                results_text += f"{row}\n"
            if len(results) > 10:
                results_text += f"\n... and {len(results) - 10} more rows"

        analysis_prompt = f"""You are a data analyst. Provide a clear, concise analysis of the SQL query results.

**Original Question:** {question}

**SQL Query Executed:**
```sql
{sql_query}
```

**Query Results:**
{results_text}

**Your Task:**
Provide a natural language answer to the original question based on these results. Be specific, accurate, and concise.
If there are multiple rows, summarize the key findings.
If there are no results, explain what that means."""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        analysis = response.choices[0].message.content
        logger.info(f"\n✓ Analysis complete")
        return analysis

    def query(self, question: str) -> Dict[str, Any]:
        """Complete text-to-SQL pipeline: question -> SQL -> results -> analysis"""
        try:
            # Generate SQL
            sql_query = self.generate_sql_query(question)

            # Execute query
            results, columns = self.execute_query(sql_query)

            # Analyze results
            analysis = self.analyze_results(question, sql_query, results, columns)

            return {
                'success': True,
                'question': question,
                'sql_query': sql_query,
                'results': results,
                'columns': columns,
                'analysis': analysis
            }

        except Exception as e:
            logger.error(f"Error in query pipeline: {e}")
            return {
                'success': False,
                'question': question,
                'error': str(e)
            }

    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()



def interactive_mode():
    """Run interactive text-to-SQL session"""
    logger.info("\n" + "="*70)
    logger.info("Text-to-SQL Interactive Mode")
    logger.info("="*70)
    logger.info("Type your questions in natural language. Type 'exit' to quit.\n")

    # Initialize
    client = OpenAI(api_key=OPENAI_API_KEY)
    schema_store = SchemaEmbeddingStore(client)
    engine = Text2SQLEngine(client, schema_store)

    try:
        while True:
            question = input("\n💬 Your question: ").strip()

            if question.lower() in ['exit', 'quit', 'q']:
                logger.info("Goodbye!")
                break

            if not question:
                continue

            # Process question
            result = engine.query(question)

            if result['success']:
                print("\n" + "="*70)
                print("📊 RESULTS")
                print("="*70)
                print(f"\n📝 Generated SQL:\n{result['sql_query']}\n")
                print(f"📈 Analysis:\n{result['analysis']}\n")

                if result['results']:
                    print(f"📋 Raw Results ({len(result['results'])} rows):")
                    print(f"Columns: {', '.join(result['columns'])}")
                    for i, row in enumerate(result['results'][:5], 1):
                        print(f"  {i}. {row}")
                    if len(result['results']) > 5:
                        print(f"  ... and {len(result['results']) - 5} more rows")
            else:
                print(f"\n❌ Error: {result['error']}")

    finally:
        engine.close()
        schema_store.close()


def run_sample_queries():
    """Run predefined sample queries for testing"""
    logger.info("\n" + "="*70)
    logger.info("Running Sample Queries")
    logger.info("="*70)

    # Sample questions
    questions = [
        "What are the different airplane producers represented in the database?",
        "How many flights are scheduled in total?",
        "Which airplane types are used for flights to New York?",
        "Find the airplane IDs and producers for airplanes that have flown to Chicago",
        "What is the total count of airplanes?",
        "List all flights departing on 2023-06-20",
    ]

    # Initialize
    client = OpenAI(api_key=OPENAI_API_KEY)
    schema_store = SchemaEmbeddingStore(client)
    engine = Text2SQLEngine(client, schema_store)

    try:
        for question in questions:
            result = engine.query(question)

            if result['success']:
                print("\n" + "="*70)
                print(f"Question: {question}")
                print("="*70)
                print(f"\nSQL: {result['sql_query']}")
                print(f"\nAnalysis: {result['analysis']}")
                print("-"*70)
            else:
                print(f"\n❌ Error for '{question}': {result['error']}")

    finally:
        engine.close()
        schema_store.close()


def main():
    """Main entry point"""
    import sys

    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY not found in environment variables")
        logger.error("Please set it in your .env file")
        sys.exit(1)

    # Check command line arguments
    # if len(sys.argv) > 1:
    #     command = sys.argv[1].lower()
    #
    #     if command == 'interactive':
    #         # Run interactive mode
    #         interactive_mode()
    #
    #     elif command == 'sample':
    #         # Run sample queries
    #         run_sample_queries()
    #
    #     else:
    #         print(f"Unknown command: {command}")
    #         print("Usage: python text2sql_pgvector.py [interactive|sample]")
    #         print("\nNote: To setup schemas, run: python setup_schemas.py")
    #
    # else:
    #     # Default: run interactive mode
    #     print("="*70)
    #     print("Text-to-SQL System")
    #     print("="*70)
    #     print("\nMake sure you have run 'python setup_schemas.py' first to setup the database.")
    #     print("\nStarting interactive mode...\n")

    interactive_mode()


if __name__ == "__main__":
    main()

