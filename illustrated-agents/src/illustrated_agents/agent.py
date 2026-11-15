from illustrated_agents.llm import LLM, Response
from illustrated_agents.trajectory import Trajectory
from illustrated_agents.memory import Memory
from illustrated_agents.tools import Tools, NativeTools


class TinyAgent:
    """
    最小限のモジュール式の教育エージェント フレームワーク

    大文字の単語の意味
    * THOUGHT：次に何をすべきかに関するLLMの推論
    * ACTION：LLMのアクションまたはツール呼び出しはJSON形式で構成されている
    * OBSERVATION：LLMが実行したACTIONの出力
    * ANSWER：1つ以上のステップを実行した後のLLMの最終回答
    """

    def __init__(self, llm: LLM, memory: Memory, tools: Tools) -> None:
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = None  # Chapter 6: Add Planning

        self.trajectory = Trajectory()

        # すべてのコンポーネントを含むビルドシステムプロンプト
        system_prompt = "You are a helpful assistant.\n\n"
        system_prompt += self.tools.prompt
        self.memory.add("system", system_prompt)

    def run(self, task: str) -> str:
        """タスクに対してエージェントを実行します"""
        self.memory.add("user", task)
        self.trajectory.initialize(task)
        return self._step(task)

    def _step(self, task: str) -> str:
        """単一のステップを実行します"""
        # THOUGHT: 応答を生成してメモリに追加する
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
        )
        self.memory.add("assistant", response.content, tool_call=response.tool_call)

        # Tool parsing
        response = self.tools.parse(response)

        # ANSWER: 停止メカニズム
        if self.tools.is_done(response):
            self.trajectory.add(response)
            return response.content

        return self._execute_action(response)

    def _execute_action(self, response: Response) -> str:
        """ツールアクションを実行します"""
        # ACTION: ツールを実行する
        result = self.tools.execute(response)

        # OBSERVATION: ツールの結果をメモリに追加して表示する
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)

        return observation


llm = LLM(model="gemma4:e4b")


# Register tool
tools = NativeTools()


def multiply(a: str, b: str) -> float:
    return float(a) * float(b)


tools.add_tool(
    name="multiply",
    func=multiply,
    description="Multiplies two numbers: multiply(a: str, b: str)",
)

# Memory
memory = Memory()

# Initialize Agent
agent = TinyAgent(llm=llm, memory=memory, tools=tools)

# ツールの使用が必要なクエリ
print(agent.run("What is 5.1 times 7.3?"))  # -> 37.23

print(agent.memory.get_messages())
# -> [{'role': 'system', 'content': 'You are a helpful assistant.\n\n'}, {'role': 'user', 'content': 'What is 5.1 times 7.3?'}, {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_zfi9cax9', 'index': 0, 'type': 'function', 'function': {'name': 'multiply', 'arguments': '{"a":"5.1","b":"7.3"}'}}]}, {'role': 'tool', 'content': '37.23'}]
