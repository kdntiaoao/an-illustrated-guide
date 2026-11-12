from illustrated_agents.llm import LLM
from illustrated_agents.trajectory import Trajectory
from illustrated_agents.memory import Memory, SummarizationMemory

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
memory = SummarizationMemory(llm=llm)
agent = TinyAgent(llm=llm, memory=memory)

# Run a query
response = agent.run("Hi, my name is Sarah. Why are flamingos pink?")
print(agent.memory.get_messages())
# -> [{'role': 'system', 'content': "Hi Sarah! That's a great question! Flamingos aren't born pink – they're actually born with grey or white feathers. They get their beautiful pink color from their diet!\n\nHere's the breakdown:\n\n*   **They eat algae and small crustaceans:** Flamingos primarily eat algae (a type of plant) and small crustaceans like brine shrimp.\n*   **Carotenoids in their food:** These algae and crustaceans contain pigments called carotenoids. Carotenoids are the same pigments that make carrots orange and tomatoes red!\n*   **Flamingos metabolize carotenoids:** As flamingos digest their food, they process these carotenoids and deposit them into their feathers.\n*   **The more they eat, the pinker they get:** The more carotenoids a flamingo consumes, the more vibrant its pink color becomes. A flamingo that doesn't get enough carotenoids in its diet will have paler feathers.\n\nIsn't that amazing? It's all thanks to what they eat!\n\n\n\nDo you have any other questions about flamingos, or any other animals you're curious about?"}]
