from illustrated_agents.llm import LLM
from illustrated_agents.llm import EmbeddingModel


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
        system = [message for message in self.messages if message["role"] == "system"]
        turns = [message for message in self.messages if message["role"] != "system"]
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


class RAGMemory(Memory):
    """RAG による長期記憶"""

    def __init__(self, embedding_model: EmbeddingModel, documents: list[str]):
        super().__init__()
        self.embedding_model = embedding_model
        self.documents = documents
        self.embeddings = [embedding_model.embed(doc) for doc in documents]

    def add(self, role: str, content: str, **kwargs) -> None:
        # 保存する前に、取得したコンテキストを使用してユーザーのクエリを拡張する
        if role == "user":
            context = "\n".join(self.search(content))
            content = f"""Context:
{context}

Question: {content}"""
        super().add(role, content, **kwargs)

    def search(self, query: str) -> list[str]:
        """クエリに最も類似した上位 k 個のドキュメントを返す"""
        query_embed = self.embedding_model.embed(query)
        scores = [self._cosine(query_embed, embed) for embed in self.embeddings]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[index] for index in ranked[:3]]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        """2 つの embeddings 間のコサイン類似度を計算する"""
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
        return dot / norm


class MultimodalMemory(Memory):
    """Memory クラスに「画像データ」も会話履歴に含められるよう拡張したクラス"""

    def add(
        self,
        role: str,
        content: str,
        tool_call: dict = None,
        image_data: str = None,
    ):
        """Add a message to memory."""
        # Image
        if image_data:
            is_url = image_data.startswith(("http://", "https://"))
            url = image_data if is_url else f"data:image/png;base64,{image_data}"
            content = [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": content},
            ]

        # Main message
        message = {"role": role, "content": content}

        # Tool call
        if tool_call:
            message["tool_calls"] = [tool_call]

        # Append message to memory
        self.messages.append(message)
