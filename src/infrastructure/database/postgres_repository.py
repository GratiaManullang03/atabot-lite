import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime, date

from src.domain.interfaces import IDatabaseRepository
from src.domain.entities import Table, TableColumn

logger = logging.getLogger(__name__)

class PostgresRepository(IDatabaseRepository):
    """
    Repository for PostgreSQL database operations
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    async def get_schemas(self) -> List[str]:
        """Get all available schemas"""
        query = """
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            ORDER BY schema_name;
        """
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                return [row['schema_name'] for row in cur.fetchall()]
    
    async def get_tables(self, schema: str) -> List[Table]:
        """Get all tables in a schema"""
        tables = []
        
        # Get table names
        table_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s 
            ORDER BY table_name;
        """
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(table_query, (schema,))
                table_names = [row['table_name'] for row in cur.fetchall()]
                
                for table_name in table_names:
                    columns = await self._get_table_columns(conn, schema, table_name)
                    row_count = await self._get_table_row_count(conn, schema, table_name)
                    
                    tables.append(Table(
                        schema_name=schema,
                        table_name=table_name,
                        columns=columns,
                        row_count=row_count
                    ))
        
        return tables
    
    async def _get_table_columns(self, conn, schema: str, table: str) -> List[TableColumn]:
        """Get columns information for a table"""
        query = """
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable::boolean,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key,
                CASE WHEN fk.column_name IS NOT NULL THEN true ELSE false END as is_foreign_key,
                fk.foreign_table_name,
                fk.foreign_column_name
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT 
                    kcu.column_name,
                    kcu.table_name,
                    kcu.table_schema
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
            ) pk ON c.column_name = pk.column_name 
                AND c.table_name = pk.table_name 
                AND c.table_schema = pk.table_schema
            LEFT JOIN (
                SELECT 
                    kcu.column_name,
                    kcu.table_name,
                    kcu.table_schema,
                    ccu.table_name as foreign_table_name,
                    ccu.column_name as foreign_column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu 
                    ON tc.constraint_name = ccu.constraint_name
                    AND tc.table_schema = ccu.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
            ) fk ON c.column_name = fk.column_name 
                AND c.table_name = fk.table_name 
                AND c.table_schema = fk.table_schema
            WHERE c.table_schema = %s AND c.table_name = %s
            ORDER BY c.ordinal_position;
        """
        
        columns = []
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (schema, table))
            for row in cur.fetchall():
                columns.append(TableColumn(
                    name=row['column_name'],
                    data_type=row['data_type'],
                    is_nullable=row['is_nullable'],
                    is_primary_key=row['is_primary_key'],
                    is_foreign_key=row['is_foreign_key'],
                    foreign_table=row.get('foreign_table_name'),
                    foreign_column=row.get('foreign_column_name')
                ))
        
        return columns
    
    async def _get_table_row_count(self, conn, schema: str, table: str) -> int:
        """Get row count for a table"""
        query = sql.SQL("SELECT COUNT(*) as count FROM {}.{}").format(
            sql.Identifier(schema),
            sql.Identifier(table)
        )
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                result = cur.fetchone()
                return result['count'] if result else 0
        except Exception as e:
            logger.warning(f"Could not get row count for {schema}.{table}: {e}")
            return 0
    
    async def get_table_data(
        self, 
        schema: str, 
        table: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get data from a table"""
        if limit:
            query = sql.SQL("SELECT * FROM {}.{} LIMIT %s").format(
                sql.Identifier(schema),
                sql.Identifier(table)
            )
            params = (limit,)
        else:
            query = sql.SQL("SELECT * FROM {}.{}").format(
                sql.Identifier(schema),
                sql.Identifier(table)
            )
            params = None
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)
                
                results = cur.fetchall()
                
                # Convert special types to JSON-serializable format
                for row in results:
                    for key, value in row.items():
                        if isinstance(value, (datetime, date)):
                            row[key] = value.isoformat()
                        elif value is not None:
                            try:
                                json.dumps({key: value})
                            except TypeError:
                                row[key] = str(value)
                
                return results