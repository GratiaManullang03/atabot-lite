from typing import List, Dict, Any
import logging

from src.domain.interfaces import IDatabaseRepository

logger = logging.getLogger(__name__)

class SchemaService:
    """
    Application service for database schema operations
    """
    
    def __init__(self, db_repository: IDatabaseRepository):
        self.db_repository = db_repository
    
    async def get_all_schemas(self) -> List[str]:
        """
        Get all available database schemas
        """
        schemas = await self.db_repository.get_schemas()
        
        # Filter out system schemas if any slipped through
        user_schemas = [
            s for s in schemas 
            if not s.startswith('pg_') and s != 'information_schema'
        ]
        
        return user_schemas
    
    async def get_schema_structure(
        self, 
        schema_name: str
    ) -> Dict[str, Any]:
        """
        Get complete structure of a schema
        """
        tables = await self.db_repository.get_tables(schema_name)
        
        structure = {
            "schema_name": schema_name,
            "tables": [],
            "total_tables": len(tables),
            "total_rows": 0,
            "relationships": []
        }
        
        for table in tables:
            table_info = {
                "name": table.table_name,
                "columns": len(table.columns),
                "rows": table.row_count or 0,
                "primary_keys": [],
                "foreign_keys": []
            }
            
            # Identify keys
            for col in table.columns:
                if col.is_primary_key:
                    table_info["primary_keys"].append(col.name)
                if col.is_foreign_key:
                    table_info["foreign_keys"].append({
                        "column": col.name,
                        "references": f"{col.foreign_table}.{col.foreign_column}"
                    })
            
            structure["tables"].append(table_info)
            structure["total_rows"] += table_info["rows"]
            
            # Extract relationships
            for fk in table_info["foreign_keys"]:
                structure["relationships"].append({
                    "from": f"{table.table_name}.{fk['column']}",
                    "to": fk["references"]
                })
        
        return structure
    
    async def get_table_description(
        self,
        schema_name: str,
        table_name: str
    ) -> str:
        """
        Generate human-readable description of a table
        """
        tables = await self.db_repository.get_tables(schema_name)
        
        # Find the specific table
        table = next((t for t in tables if t.table_name == table_name), None)
        if not table:
            raise ValueError(f"Table {schema_name}.{table_name} not found")
        
        # Build description
        parts = [f"Tabel '{table_name}' dalam schema '{schema_name}'"]
        
        if table.row_count:
            parts.append(f"memiliki {table.row_count} baris data")
        
        parts.append(f"dengan {len(table.columns)} kolom:")
        
        # Describe columns
        for col in table.columns:
            col_desc = f"- {col.name} ({col.data_type})"
            
            if col.is_primary_key:
                col_desc += " [PRIMARY KEY]"
            if col.is_foreign_key:
                col_desc += f" [FOREIGN KEY -> {col.foreign_table}.{col.foreign_column}]"
            if not col.is_nullable:
                col_desc += " [NOT NULL]"
            
            parts.append(col_desc)
        
        return "\n".join(parts)
    
    async def analyze_schema_quality(
        self,
        schema_name: str
    ) -> Dict[str, Any]:
        """
        Analyze schema quality and provide recommendations
        """
        tables = await self.db_repository.get_tables(schema_name)
        
        analysis = {
            "schema": schema_name,
            "issues": [],
            "recommendations": [],
            "stats": {
                "total_tables": len(tables),
                "tables_without_pk": 0,
                "tables_without_fk": 0,
                "empty_tables": 0
            }
        }
        
        for table in tables:
            # Check for primary key
            has_pk = any(col.is_primary_key for col in table.columns)
            if not has_pk:
                analysis["issues"].append(f"Table '{table.table_name}' lacks primary key")
                analysis["stats"]["tables_without_pk"] += 1
            
            # Check for foreign keys
            has_fk = any(col.is_foreign_key for col in table.columns)
            if not has_fk and len(tables) > 1:
                analysis["stats"]["tables_without_fk"] += 1
            
            # Check for empty tables
            if table.row_count == 0:
                analysis["stats"]["empty_tables"] += 1
                analysis["issues"].append(f"Table '{table.table_name}' is empty")
        
        # Generate recommendations
        if analysis["stats"]["tables_without_pk"] > 0:
            analysis["recommendations"].append(
                "Add primary keys to all tables for better performance and data integrity"
            )
        
        if analysis["stats"]["tables_without_fk"] > len(tables) * 0.7:
            analysis["recommendations"].append(
                "Consider adding foreign keys to establish relationships between tables"
            )
        
        if analysis["stats"]["empty_tables"] > 0:
            analysis["recommendations"].append(
                "Populate empty tables with data or remove them if not needed"
            )
        
        return analysis