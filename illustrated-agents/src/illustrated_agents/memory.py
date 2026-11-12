class Memory:
    """会話履歴を保存するシンプルなメモリモジュール"""

    def __init__(self):
        self.messages = []

    def add(
        self, role: str, content: str, tool_call: dict | None = None, **kwargs
    ) -> None:
        """Add a message to memory."""
        message = {"role": role, "content": content}

        # Tool call
        if tool_call:
            message["tool_calls"] = [tool_call]

        # Append message to memory
        self.messages.append(message)

    def get_messages(self) -> list[dict]:
        """Get all messages."""
        return self.messages


class TrimmingMemory(Memory):
    """ユーザー / アシスタント の最後の 2 つのターンのみを保持するメモリ"""

    def add(self, role: str, content: str, **kwargs) -> None:
        # 最初に親クラスを使用して新しいメッセージを追加する (Memory)
        super().add(role, content, **kwargs)

        # 次に、システムメッセージと最新の 2 ターンを保持する (4 messages)
        system = [
            message for message in self.messages if message["role"] == "system"
        ]
        turns = [
            message for message in self.messages if message["role"] != "system"
        ]
        self.messages = system + turns[-4:]