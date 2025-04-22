# Flexible Key-Value Extraction MCP Server

This MCP server extracts key-value pairs from arbitrary, noisy, or unstructured text using LLM (o4-mini) and pydantic-ai. It guarantees type safety and supports multiple output formats (JSON, YAML, TOML). The server is robust to any input and always attempts to structure data as much as possible.

**Note**: This functionality totally relies on pydantic-ai and LLM, so it does not guarantee perfect extraction.

## Features
- **Flexible extraction**: Handles any input, including noisy or broken data.
- **Type-safe output**: Uses Pydantic for output validation.
- **Multiple formats**: Returns results as JSON, YAML, or TOML.
- **Robust error handling**: Always returns a well-formed response, even on failure.

## Tools

### 1. `extract_json`
- **Description**: Extracts key-value pairs from arbitrary noisy text and returns them as type-safe JSON (Python dict).
- **Arguments**:
  - `input_text` (string): Input string containing noisy or unstructured data.
- **Returns**: `{ "success": True, "result": ... }` or `{ "success": False, "error": ... }`
- **Example**:
  ```json
  {
    "success": true,
    "result": { "foo": 1, "bar": "baz" }
  }
  ```

### 2. `extract_yaml`
- **Description**: Extracts key-value pairs from arbitrary noisy text and returns them as type-safe YAML (string).
- **Arguments**:
  - `input_text` (string): Input string containing noisy or unstructured data.
- **Returns**: `{ "success": True, "result": ... }` or `{ "success": False, "error": ... }`
- **Example**:
  ```json
  {
    "success": true,
    "result": "foo: 1\nbar: baz"
  }
  ```

### 3. `extract_toml`
- **Description**: Extracts key-value pairs from arbitrary noisy text and returns them as type-safe TOML (string).
- **Arguments**:
  - `input_text` (string): Input string containing noisy or unstructured data.
- **Returns**: `{ "success": True, "result": ... }` or `{ "success": False, "error": ... }`
- **Example**:
  ```json
  {
    "success": true,
    "result": "foo = 1\nbar = \"baz\""
  }
  ```

## Usage

### Requirements
- Python 3.9+
- `fastmcp`, `pydantic-ai`, and their dependencies (see `requirements.txt`)
- API key for o4-mini (set in `settings.json` under `env`)

### Running the Server

```bash
python mcp_server.py
```

### Example Request (via MCP Host)

```json
{
  "tool": "extract_json",
  "arguments": {
    "input_text": "foo: 1, bar: baz, some noise here, xyz=42"
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": { "foo": 1, "bar": "baz", "xyz": 42 }
}
```

## MCP Host Configuration Example

You can register this server in your MCP Host settings like below:

```json
"pydantic-ai-mcp-server": {
  "command": "pipx",
  "args": [
    "pydantic-ai-mcp-server"
  ],
  "env": {
    "OPENAI_API_KEY": "api-key"
  }
},
```

And register the tools as:

```json
{
  "tools": [
    {
      "name": "extract_json",
      "description": "Extracts key-value pairs from noisy text as JSON.",
      "arguments": {
        "input_text": "string"
      }
    },
    {
      "name": "extract_yaml",
      "description": "Extracts key-value pairs from noisy text as YAML.",
      "arguments": {
        "input_text": "string"
      }
    },
    {
      "name": "extract_toml",
      "description": "Extracts key-value pairs from noisy text as TOML.",
      "arguments": {
        "input_text": "string"
      }
    }
  ],
  "endpoint": "http://localhost:8000"
}
```

## Environment Variables
- Set your OpenAI-compatible API key for o4-mini in `settings.json`:

```json
{
  "env": {
    "OPENAI_API_KEY": "sk-..."
  }
}
```

## License
MIT

## Author
Kunihiro Saito (and contributors)
