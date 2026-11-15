import json
import inspect
from typing import Any, Callable
from illustrated_agents.llm import Response

# Convert specific types to string descriptions
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool_to_schema(function: Callable) -> dict:
    """Convert a Python function to an OpenAI-style tool schema."""
    signature = inspect.signature(function)

    # Extract meatadata
    properties, required = {}, []
    for name, parameter in signature.parameters.items():
        properties[name] = {"type": TYPE_MAP.get(parameter.annotation, "string")}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    # Fill schema
    schema = {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": inspect.getdoc(function),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    return schema


class Tools:
    """Tool registry for the Agent."""

    def __init__(self, requires_approval: list[str] = []):
        """実行前に承認が必要なツールを初期化して選択する"""
        self.registry = {}
        self.requires_approval = requires_approval

    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """エージェントが使用できるツールを登録する

        Arguments:
            name: ツールの名前
            func: ツールを実装する機能
            description: ツールの説明
        """
        self.registry[name] = {"function": func, "description": description}

    @property
    def descriptions(self) -> str:
        """登録されているすべてのツールの説明を取得する"""
        return "\n".join(
            f"`{tool}`: {self.registry[tool]['description']}" for tool in self.registry
        )

    @property
    def prompt(self) -> str:
        return f"""
# Tools

If needed, you can only use the following tools to assist you
in completing tasks:

{self.descriptions}

To use a tool, respond with JSON:
{{"tool": "name", "kwargs": {{"param": "value"}}}}
"""

    def parse(self, response: Response) -> Response:
        """テキストから JSON ツール呼び出しを解析する"""
        text = response.content

        if '"tool":' in text or '"tool:"' in text:
            start, end = text.find("{"), text.rfind("}") + 1
            tool_call = json.loads(text[start:end])

            # 解析されたツール呼び出しを応答に追加する
            return Response(
                content=response.content,
                reasoning=response.reasoning,
                tool_call=tool_call,
            )

        return response

    def execute(self, response: Response) -> Any:
        """Run a registered tool.

        Arguments:
            tool_call: 「tool」と「kwargs」を含む解析されたツール呼び出し辞書
        """
        tool_call = response.tool_call
        name, kwargs = tool_call["tool"], tool_call.get("kwargs", {})

        # Human-in-the-loop: 危険なツールを実行する前に尋ねる
        if name in self.registry and name in self.requires_approval:
            response = input(f"Allow {name}? [y/N] ").strip().lower()
            if response not in ("y", "yes"):
                return f"Tool '{name}' was denied by the user."

        # 登録されたツールを操作する
        if name in self.registry:
            tool_func = self.registry[name]["function"]
            return tool_func(**kwargs)

        return f"Tool '{name}' not found."

    def observation(self, result: str) -> tuple[str, str]:
        """ユーザーとしての観察を返す"""
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """`TinyAgent` の停止メカニズム"""
        if not response.tool_call:
            return True
        if response.tool_call["tool"] == "final_answer":
            response.content = response.tool_call.get("kwargs", "")
            return True
        return False

    @property
    def schemas(self) -> None:
        """ネイティブツールの呼び出しのみに使用する"""
        return None


class NativeTools(Tools):
    """ネイティブ関数呼び出しを使用したツールレジストリ"""

    @property
    def schemas(self) -> list[dict]:
        """ネイティブ関数呼び出し用のツール関数を返す"""
        return [tool_to_schema(tool["function"]) for tool in self.registry.values()]

    @property
    def prompt(self) -> str:
        """ネイティブツール呼び出しのプロンプトが必要ないため空"""
        return ""

    def parse(self, response: Response) -> Response:
        """Parse a tool call."""
        # ツール呼び出しがない場合はそのままレスポンスを返す
        if not response.tool_call:
            return response

        # ツール呼び出しからツール名と引数を抽出する
        args = response.tool_call["function"]["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        tool_call = {
            "tool": response.tool_call["function"]["name"],
            "kwargs": args,
        }

        # 解析されたツール呼び出しを応答に追加する
        return Response(
            content=response.content,
            reasoning=response.reasoning,
            tool_call=tool_call,
        )

    def observation(self, result: str) -> tuple[str, str]:
        """ネイティブツールの結果では「ツール」ロールが使用される"""
        return "tool", str(result)

    def is_done(self, response: Response) -> bool:
        """ツール呼び出しがない場合は「TinyAgent」が完了したことを意味する"""
        return not response.tool_call
