from illustrated_agents.llm import LLM
from illustrated_agents.trajectory import Trajectory
from illustrated_agents.memory import Memory

# Re-initialize the LLM with the updated class
llm = LLM(model="gemma4:e4b")

memory = Memory()


class TinyAgent:
    """最小限のモジュール式の教育エージェント フレームワーク"""

    def __init__(self) -> None:
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


# 実行
agent_with_memory = TinyAgent()
# First query
first_response = agent_with_memory.run("Hi! We are Maarten and Jay, authors of 'An Illustrated Guide to AI Agents'.")
print(first_response)
# ->
# Hi Maarten and Jay! It's a pleasure to "meet" the authors of *An Illustrated Guide to AI Agents*.

# That sounds like a fascinating and very timely book! AI Agents are such a hot topic right now.

# **I'd love to know more! To help me respond better, could you tell me a bit about what you'd like to discuss? For example, are you looking to:**

# 1. **Share insights** about a specific concept in the book?
# 2. **Get feedback** on a topic you're researching?
# 3. **Brainstorm marketing ideas** or talking points?
# 4. **Ask me questions** about AI Agents, LLMs, or related technologies?

# What's on your mind today? 😊

# Second query
second_response = agent_with_memory.run("Hi! What are our names?")
print(second_response)
# -> Your names are Maarten and Jay.