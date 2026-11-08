import json
import urllib.request

# Prepare request
data = json.dumps(
    {
        "model": "gemma4:e4b",
        "messages": [{"role": "user", "content": "こんにちは！"}],
    }
).encode("utf-8")

# Post request
req = urllib.request.Request(
    url="http://localhost:11434/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"},
)

# Parse response
with urllib.request.urlopen(req) as response:
    result = json.loads(response.read())

# Print response
print(result["choices"][0])
# -> {'index': 0, 'message': {'role': 'assistant', 'content': 'こんにちは！😊 お話しできて嬉しいです。\n\n何かお困りのことはありますか？それとも、ただお話に花を咲かせたいだけでしょうか？\n\nお好きなトピックがあれば、何でも聞かせてくださいね！☕✨'}, 'finish_reason': 'stop'}
