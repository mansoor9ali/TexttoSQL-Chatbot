"""
PostgreSQL Database Query Script for Airline Database
Queries the airline_db database with airplanes and flights tables
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def get_connection_string():
    """Build PostgreSQL connection string from environment variables"""
    connection_string = (
        f"postgresql://{os.getenv('PGUSER', 'user-name')}:"
        f"{os.getenv('PGPASSWORD', 'strong-password')}@"
        f"{os.getenv('PGHOST', 'localhost')}:"
        f"{os.getenv('PGPORT', '5432')}/"
        f"{os.getenv('PGDATABASE', 'airline_db')}"
    )
    return connection_string


def query_airplanes(conn):
    """Query all records from the airplanes table"""
    print("\n" + "="*50)
    print("AIRPLANES TABLE")
    print("="*50)

    with conn.cursor() as cur:
        cur.execute("SELECT * FROM airplanes ORDER BY Airplane_id")
        rows = cur.fetchall()

        if rows:
            print(f"\nFound {len(rows)} airplane(s):\n")
            print(f"{'Airplane ID':<15} {'Producer':<25} {'Type':<15}")
            print("-" * 55)
            for row in rows:
                print(f"{row[0]:<15} {row[1]:<25} {row[2]:<15}")
        else:
            print("\nNo airplanes found in the database.")

    return rows


def query_flights(conn):
    """Query all records from the flights table"""
    print("\n" + "="*50)
    print("FLIGHTS TABLE")
    print("="*50)

    with conn.cursor() as cur:
        cur.execute("SELECT * FROM flights ORDER BY Flight_number")
        rows = cur.fetchall()

        if rows:
            print(f"\nFound {len(rows)} flight(s):\n")
            print(f"{'Flight #':<12} {'Arrival Time':<20} {'Arrival Date':<15} "
                  f"{'Dept Time':<20} {'Dept Date':<15} {'Destination':<15} {'Airplane ID':<12}")
            print("-" * 120)
            for row in rows:
                print(f"{row[0]:<12} {row[1]:<20} {row[2]:<15} "
                      f"{row[3]:<20} {row[4]:<15} {row[5]:<15} {row[6]:<12}")
        else:
            print("\nNo flights found in the database.")

    return rows


def query_flights_with_airplanes(conn):
    """Query flights joined with airplane information"""

    with conn.cursor() as cur:
        query = """
            SELECT 
                f.Flight_number,
                f.Departure_date,
                f.Departure_time,
                f.Arrival_date,
                f.Arrival_time,
                f.Destination,
                a.Airplane_id,
                a.Producer,
                a.Type
            FROM flights f
            JOIN airplanes a ON f.Airplane_id = a.Airplane_id
            ORDER BY f.Departure_date, f.Departure_time
        """
        cur.execute(query)
        rows = cur.fetchall()

        if rows:
            print(f"Found {len(rows)} flight(s) with airplane details:\n")
            print(f"{'Flight #':<12} {'Dept Date':<15} {'Dept Time':<20} "
                  f"{'Arr Date':<15} {'Arr Time':<20} {'Destination':<15} "
                  f"{'Airplane':<10} {'Producer':<20} {'Type':<10}")
            print("-" * 150)
            for row in rows:
                print(f"{row[0]:<12} {row[1]:<15} {row[2]:<20} "
                      f"{row[3]:<15} {row[4]:<20} {row[5]:<15} "
                      f"{row[6]:<10} {row[7]:<20} {row[8]:<10}")
        else:
            print("\nNo flights with airplane details found.")

    return rows


def get_table_statistics(conn):
    """Get basic statistics about the tables"""
    print("\n" + "="*50)
    print("DATABASE STATISTICS")
    print("="*50)

    with conn.cursor() as cur:
        # Count airplanes
        cur.execute("SELECT COUNT(*) FROM airplanes")
        airplane_count = cur.fetchone()[0]

        # Count flights
        cur.execute("SELECT COUNT(*) FROM flights")
        flight_count = cur.fetchone()[0]

        # Count distinct producers
        cur.execute("SELECT COUNT(DISTINCT Producer) FROM airplanes")
        producer_count = cur.fetchone()[0]

        # Count distinct destinations
        cur.execute("SELECT COUNT(DISTINCT Destination) FROM flights")
        destination_count = cur.fetchone()[0]

        print(f"\nTotal Airplanes: {airplane_count}")
        print(f"Total Flights: {flight_count}")
        print(f"Distinct Producers: {producer_count}")
        print(f"Distinct Destinations: {destination_count}")


def main():
    """Main function to connect to database and execute queries"""
    connection_string = get_connection_string()
    print("Connecting to PostgreSQL database...")
    try:
        # Connect to the database
        with psycopg2.connect(connection_string) as conn:
            print("✓ Successfully connected to the database!\n")

            # Execute queries
            # query_airplanes(conn)
            # query_flights(conn)
            query_flights_with_airplanes(conn)
            #get_table_statistics(conn)

            print("\n" + "="*50)
            print("All queries completed successfully!")
            print("="*50)

    except psycopg2.OperationalError as e:
        print(f"\n✗ Error connecting to the database:")
        print(f"  {e}")
        print("\nPlease check your connection settings and ensure:")
        print("  1. PostgreSQL server is running")
        print("  2. Database 'airline_db' exists")
        print("  3. Connection credentials are correct")
        print("  4. Network connection is available")
    except Exception as e:
        print(f"\n✗ An error occurred:")
        print(f"  {e}")


if __name__ == '__main__':
    main()
