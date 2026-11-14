from typing import Callable

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
            f"`{tool}`: {self.registry[tool]['description']}"
            for tool in self.registry
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

    @property
    def schemas(self) -> None:
        """ネイティブツールの呼び出しのみに使用する"""
        return None