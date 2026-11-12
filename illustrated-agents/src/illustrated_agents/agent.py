from illustrated_agents.llm import LLM
from illustrated_agents.trajectory import Trajectory
from illustrated_agents.memory import Memory, TrimmingMemory

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

# Create the Agent
llm = LLM(model="gemma3:12b")
memory = TrimmingMemory()
agent = TinyAgent(llm, memory)

# Run some interactions
response_1 = agent.run("Hi! I'm reading 'An Illustrated Guide to AI Agents'.")
response_2 = agent.run("There are many flamingos in this book, why?")
response_3 = agent.run("What is 1+1?")
response_4 = agent.run("What is 2+2?")
print(agent.memory.get_messages())
# -> [{'role': 'user', 'content': 'What is 1+1?'}, {'role': 'assistant', 'content': "That's a classic! 1 + 1 = 2. 😊\n\nIt seems like a simple question, but it's a fundamental concept in mathematics. I'm glad you asked!\n\n\n\nAre you using this to test if I'm working correctly? 😉"}, {'role': 'user', 'content': 'What is 2+2?'}, {'role': 'assistant', 'content': "2 + 2 = 4! \n\nI'm doing just fine, thanks for checking! 😊 Do you want to try another math problem?"}]
