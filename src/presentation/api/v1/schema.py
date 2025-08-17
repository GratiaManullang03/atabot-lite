from fastapi import APIRouter, Depends, HTTPException
from typing import List

from src.presentation.models.response_models import SchemaResponse
from src.presentation.api.dependencies import get_postgres_repository
from src.infrastructure.database.postgres_repository import PostgresRepository

router = APIRouter()

@router.get("/", response_model=List[str])
async def get_schemas(
    db: PostgresRepository = Depends(get_postgres_repository)
):
    """
    Get all available database schemas
    """
    try:
        schemas = await db.get_schemas()
        return schemas
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{schema_name}", response_model=SchemaResponse)
async def get_schema_info(
    schema_name: str,
    db: PostgresRepository = Depends(get_postgres_repository)
):
    """
    Get detailed information about a schema
    """
    try:
        tables = await db.get_tables(schema_name)
        
        # Convert to dict format
        tables_dict = []
        for table in tables:
            table_dict = {
                "name": table.table_name,
                "row_count": table.row_count,
                "columns": [
                    {
                        "name": col.name,
                        "type": col.data_type,
                        "nullable": col.is_nullable,
                        "is_primary": col.is_primary_key,
                        "is_foreign": col.is_foreign_key
                    }
                    for col in table.columns
                ]
            }
            tables_dict.append(table_dict)
        
        return SchemaResponse(
            schema_name=schema_name,
            tables=tables_dict,
            total_tables=len(tables_dict)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))