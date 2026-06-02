import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from fastmcp import FastMCP
from retriever import PHISRetriever

mcp = FastMCP("PHIS SDS Search")

retriever = PHISRetriever(verbose=True)


@mcp.tool()
def search_phis_docs(query: str, n_results: int = 3) -> str:
    """
    Search PHIS SDS SAM documentation using semantic search.
    Use this to find information about Special Approval Medicine system design.

    Args:
        query: Natural language question about PHIS SAM system
        n_results: Number of results to return (default 3)

    Returns:
        Relevant documentation chunks with page numbers
    """
    return retriever.search_formatted(query, n=n_results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
