from illustrated_agents.llm import LLM, EmbeddingModel
from illustrated_agents.trajectory import Trajectory
from illustrated_agents.memory import Memory, RAGMemory

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

# RAGMemory with external documents
documents = [
    "Sarah works as a marine biologist studying coral reefs.",
    "Sarah lives in Lisbon, Portugal.",
    "Sarah's favorite hobby is rock climbing.",
    "Sarah favorite animal is flamingos.",
    "Sarah speaks fluent Spanish and Portuguese.",
    "Ilse is a software engineer at a renewable energy startup.",
    "Ilse lives in Amsterdam, the Netherlands.",
    "Ilse plays the cello in a local string quartet.",
    "Ilse's favorite author is Brandon Sanderson.",
    "Ilse's favorite animal is dolphins.",
]

# Create the Agent
llm = LLM(model="gemma3:12b")
embedding_model = EmbeddingModel(model="embeddinggemma")
memory = RAGMemory(documents=documents, embedding_model=embedding_model)
agent = TinyAgent(llm=llm, memory=memory)

# Run a query
response = agent.run("What is Sarah's favorite animal?")
print(response) # -> Sarah's favorite animal is flamingos.
