class TinyAgent:
    """最小限のモジュール式の教育エージェント フレームワーク"""

    def __init__(self) -> None:
        self.llm = None  # Chapter 2 & 3: Add LLM
        self.memory = None  # Chapter 4: Add Memory
        self.tools = None  # Chapter 5: Add Tools
        self.planner = None  # Chapter 6: Add Planning

    def run(self, task: str) -> str:
        """タスクに対してエージェントを実行します"""
        return self._step(task)

    def _step(self, task: str) -> str:
        """単一のステップを実行します"""
        # Placeholder - 後の章で実装されます
        return f"Received: {task}"

    def _execute_action(self, action: str) -> str:
        """ツールアクションを実行します"""
        # Placeholder - 後の章で実装されます
        return f"Executed action: {action}"


# 実行
agent = TinyAgent()
print(agent.run("What is 2 + 2?"))
