from illustrated_agents.llm import LLM
from illustrated_agents.trajectory import Trajectory
from illustrated_agents.memory import Memory

class TinyAgent:
    """最小限のモジュール式の教育エージェント フレームワーク"""

    def __init__(self, llm: LLM, memory: Memory) -> None:
        self.llm = llm
        self.memory = memory
        self.tools = None  # Chapter 5: Add Tools
        self.planner = None  # Chapter 6: Add Planning

        self.trajectory = Trajectory()

    def run(self, task: str) -> str:
        """タスクに対してエージェントを実行します"""
        self.memory.add("user", task)
        self.trajectory.initialize(task)
        return self._step(task)

    def _step(self, task: str) -> str:
        """単一のステップを実行します"""
        response = self.llm.generate(self.memory.get_messages())
        self.memory.add("assistant", response.content)
        self.trajectory.add(response)
        return response.content

    def _execute_action(self, action: str) -> str:
        """ツールアクションを実行します"""
        # Placeholder - 後の章で実装されます
        return f"Executed action: {action}"


llm = LLM(model="gemma3:12b")
memory = Memory()
agent_with_memory = TinyAgent(llm, memory)
# First query
first_response = agent_with_memory.run("Hi! We are Maarten and Jay, authors of 'An Illustrated Guide to AI Agents'.")
print(first_response)
# ->
"""
Hello Maarten and Jay! It's a pleasure to "meet" you, the authors of 'An Illustrated Guide to AI Agents.'

That sounds like a fascinating and very timely book! AI Agents are such a hot topic right now.

**How can I help you today?** Are you looking for:

* **Promotional ideas** for the book?
* **Technical explanations** that might help you structure future content?
* **Feedback** on concepts related to AI agents?
* Just a friendly chat about the future of autonomous systems?

Let me know what's on your mind! I'm ready to discuss AI agents! 🤖
"""

# Second query
second_response = agent_with_memory.run("Hi! What are our names?")
print(second_response)
# -> You mentioned your names are **Maarten and Jay**.