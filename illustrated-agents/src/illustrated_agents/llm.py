import json
import urllib.request
from dataclasses import dataclass


@dataclass
class Response:
    """LLM 呼び出しからの構造化された応答"""

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None


class LLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "no_key",
        think: bool = False,
        temperature: float | None = None,
    ):
        """指定されたモデルで LLM を初期化します"""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think
        self.temperature = temperature

    def generate(self, messages: list[dict], tools: list | None = None) -> Response:
        """メッセージのリストを指定して LLM からの応答を生成します"""
        # Build the request body
        body = {
            "model": self.model,
            "messages": messages,
        }

        # Tools and Reasoning
        if tools:
            body["tools"] = tools
        if not self.think:
            body["reasoning_effort"] = "none"
        if self.temperature is not None:
            body["temperature"] = self.temperature

        # POST to the OpenAI-compatible /chat/completions endpoint
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())

        # Extract message, tool_call, and metadata
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        tool_call = tool_calls[0] if tool_calls else None
        metadata = {
            "model": data["model"],
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
        }

        # Format as Response dataclass
        return Response(
            content=message.get("content"),
            reasoning=message.get("reasoning"),
            tool_call=tool_call,
            metadata=metadata,
        )

class EmbeddingModel:
    """Generate embeddings."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
    ):
        """指定されたモデルで埋め込みモデルを初期化"""
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        """テキストを数値ベクトルに変換する"""
        # OpenAI 互換の /embeddings エンドポイントへの POST
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as resp:
            response = json.loads(resp.read())

        # embedding を抽出して返す
        return response["data"][0]["embedding"]