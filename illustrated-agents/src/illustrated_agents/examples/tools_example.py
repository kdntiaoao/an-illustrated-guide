from illustrated_agents.tools import Tools

def multiply(a: str, b: str) -> float:
    return float(a) * float(b)

# Register tool
tools = Tools()
tools.add_tool(
    name="multiply",
    func=multiply,
    description="Multiplies two numbers: multiply(a: str, b: str)"
)

# Show prompt
print(tools.prompt)
# ->
"""
# Tools

If needed, you can only use the following tools to assist you
in completing tasks:

`multiply`: Multiplies two numbers: multiply(a: str, b: str)

To use a tool, respond with JSON:
{"tool": "name", "kwargs": {"param": "value"}}
"""
