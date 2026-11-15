from illustrated_agents.tools import Tools
from illustrated_agents.llm import Response

def multiply(a: str, b: str) -> float:
    return float(a) * float(b)

# Register tool
tools = Tools()
tools.add_tool(
    "multiply", multiply, "Multiplies two numbers: multiply(a: str, b: str)"
)

# Fake LLM response
response = Response(
    content='{"tool": "multiply", "kwargs": {"a": "3.1", "b": "6.5"} }'
)

# Parse into a tool call
response = tools.parse(response)

# Execute the tool
print(tools.execute(response))
# -> 20.150000000000002
