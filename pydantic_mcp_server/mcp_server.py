from fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from typing import Any, Dict, Union

server = FastMCP(
    'Flexible Key-Value Extraction MCP Server',
    description="""
    Extracts key-value pairs from noisy or unstructured text using LLM+Pydantic-AI,
    and returns them in a type-safe way in the specified format (JSON/YAML/TOML).
    Robust to any input, always attempts to structure data as much as possible.
    """
)

agent = Agent('openai:o4-mini')

class KeyValuePairsModel(BaseModel):
    """
    Flexible model for type-safe key-value pairs extraction.
    """
    data: Dict[str, Any] = Field(
        ...,
        description="Extracted key-value pairs. Keys are strings, values can be any type."
    )

def safe_run_agent(prompt: str, format: str) -> Dict[str, Union[bool, str, dict]]:
    """
    Extract type-safe key-value pairs from noisy text using LLM + pydantic-ai.
    Returns error details in a type-safe way on failure.
    Args:
        prompt (str): Input text, possibly noisy or unstructured.
        format (str): Output format ("json", "yaml", or "toml").
    Returns:
        dict: {"success": True, "result": ...} or {"success": False, "error": ...}
    """
    try:
        result = agent.run(prompt, model=KeyValuePairsModel, format=format)
        if format == "json":
            return {"success": True, "result": result.output}
        else:
            return {"success": True, "result": result.raw_output}
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@server.tool(
    name="extract_json",
    description="""
    Extracts key-value pairs from arbitrary noisy text and returns them as type-safe JSON (dict).
    Example: {"foo": 1, "bar": "baz"}
    """,
    args={
        "input_text": {
            "type": "string",
            "description": "Input string containing noisy or unstructured data."
        }
    }
)
async def extract_json(input_text: str) -> dict:
    return safe_run_agent(input_text, "json")

@server.tool(
    name="extract_yaml",
    description="""
    Extracts key-value pairs from arbitrary noisy text and returns them as type-safe YAML (str).
    Example: foo: 1\nbar: baz
    """,
    args={
        "input_text": {
            "type": "string",
            "description": "Input string containing noisy or unstructured data."
        }
    }
)
async def extract_yaml(input_text: str) -> dict:
    return safe_run_agent(input_text, "yaml")

@server.tool(
    name="extract_toml",
    description="""
    Extracts key-value pairs from arbitrary noisy text and returns them as type-safe TOML (str).
    Example: foo = 1\nbar = "baz"
    """,
    args={
        "input_text": {
            "type": "string",
            "description": "Input string containing noisy or unstructured data."
        }
    }
)
async def extract_toml(input_text: str) -> dict:
    return safe_run_agent(input_text, "toml")

if __name__ == '__main__':
    server.run()