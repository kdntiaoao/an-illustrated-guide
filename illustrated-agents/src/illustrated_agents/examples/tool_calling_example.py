import json
import urllib.request

# Prompt
messages = [{"role": "user", "content": "What is 5.1 times 7.3?"}]

tool_schema = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    },
}

# Prepare the data
body = {
    "model": "gemma4:e4b",
    "messages": messages,
    "stream": False,
    "tools": [tool_schema],
}

# Prepare the request
request = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)

# Call and parse the response
with urllib.request.urlopen(request) as response:
    print(json.loads(response.read())["message"])
# -> {'role': 'assistant', 'content': '', 'thinking': '1. **Identify the user\'s request:** The user is asking to calculate "5.1 times 7.3".\n2. **Analyze the available tools:** I have one tool: `multiply(a: number, b: number)`.\n3. **Determine the required inputs:** The multiplication request requires two numbers: 5.1 and 7.3.\n4. **Check if the tool can fulfill the request:** Yes, the `multiply` tool is exactly designed for this.\n5. **Construct the tool call:**\n    * `a` should be 5.1.\n    * `b` should be 7.3.\n    * The call is `multiply(a=5.1, b=7.3)`.\n6. **Format the output:** Generate the required JSON structure for the tool call.', 'tool_calls': [{'id': 'call_0ohwx8d6', 'function': {'index': 0, 'name': 'multiply', 'arguments': {'a': 5.1, 'b': 7.3}}}]}
