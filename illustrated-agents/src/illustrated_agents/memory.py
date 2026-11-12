from illustrated_agents.llm import LLM

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

class SummarizationMemory(Memory):
    """過去を圧縮したメモリは、LLM を介して実行中のサマリーに変わる"""

    def __init__(self, llm: LLM):
        super().__init__()
        self.llm = llm

    def add(self, role: str, content: str, **kwargs) -> None:
        # 最初に親クラスを使用して新しいメッセージを追加する (Memory)
        super().add(role, content, **kwargs)

        # ターンが完了するたびに、実行概要を更新する
        if role == "assistant":
            # 新しい会話ターンから既存の概要を分割する
            summary = ""
            conversation = ""
            for message in self.messages:
                if message["role"] == "system":
                    summary = message["content"]
                else:
                    conversation += f"{message['role']}: {message['content']}\n"

            # 新しい会話で概要を拡張するよう LLM に依頼する
            prompt = f"""Update the summary with the new conversation.

Summary: {summary}

Conversation:
{conversation}
Output the updated summary only."""
            response = self.llm.generate([{"role": "user", "content": prompt}])
            self.messages = [{"role": "system", "content": response.content}]