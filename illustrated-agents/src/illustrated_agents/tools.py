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
    def schemas(self) -> None:
        """Used only for native tool-calling."""
        return None