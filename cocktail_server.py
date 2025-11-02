from fastmcp import FastMCP
from services.rag_service import run_cocktail_rag

mcp = FastMCP("Demo 🚀")

@mcp.tool
def rag_cocktails(query: str):
    """Retrieve cocktail recommendations based on a query."""
    return run_cocktail_rag(query=query)

if __name__ == "__main__":
    mcp.run(transport="stdio")