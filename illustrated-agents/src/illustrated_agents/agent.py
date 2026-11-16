from illustrated_agents.llm import LLM, Response
from illustrated_agents.trajectory import Trajectory
from illustrated_agents.memory import Memory
from illustrated_agents.tools import Tools, NativeTools
from illustrated_agents.toolbox import add, multiply, subtract
from illustrated_agents.planning import ReAct, NativeReAct


class TinyAgent:
    """
    最小限のモジュール式の教育エージェント フレームワーク

    大文字の単語の意味
    * THOUGHT：次に何をすべきかに関するLLMの推論
    * ACTION：LLMのアクションまたはツール呼び出しはJSON形式で構成されている
    * OBSERVATION：LLMが実行したACTIONの出力
    * ANSWER：1つ以上のステップを実行した後のLLMの最終回答
    """

    def __init__(self, llm: LLM, memory: Memory, tools: Tools, planner: ReAct) -> None:
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner

        self.trajectory = Trajectory()

        # すべてのコンポーネントを含むビルドシステムプロンプト
        system_prompt = "You are a helpful assistant.\n\n"
        system_prompt += self.planner.prompt
        system_prompt += self.tools.prompt
        self.memory.add("system", system_prompt)

    def run(self, task: str) -> str:
        """タスクに対してエージェントを実行します"""
        self.memory.add("user", task)
        self.trajectory.initialize(task)

        # *Autonomy* loop
        for step in range(self.planner.max_steps):
            result = self._step()
            if result is not None:
                return result

        return "Max steps reached without completion."

    def _step(self) -> str | None:
        """単一のステップを実行します"""
        # THOUGHT: 応答を生成してメモリに追加する
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
        )
        self.memory.add("assistant", response.content, tool_call=response.tool_call)

        # Tool parsing
        response = self.planner.parse(response)
        response = self.tools.parse(response)

        # ANSWER: 停止メカニズム
        if self.tools.is_done(response):
            self.trajectory.add(response)
            return response.content

        return self._execute_action(response)

    def _execute_action(self, response: Response) -> None:
        """ツールアクションを実行します"""
        # ACTION: ツールを実行する
        result = self.tools.execute(response)

        # OBSERVATION: ツールの結果をメモリに追加して表示する
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)

        return None


# Create the Agent
llm = LLM(model="gemma4:e4b")

# Register tools
tools = NativeTools()
tools.add_tool("add", add, "add(a: str, b: str)")
tools.add_tool("subtract", subtract, "subtract(a: str, b: str)")
tools.add_tool("multiply", multiply, "multiply(a: str, b: str)")

# Memory
memory = Memory()

# ReAct
react = NativeReAct(max_steps=10)

# Create agent
agent = TinyAgent(llm=llm, tools=tools, memory=memory, planner=react)

# Multi-step task with reasoning
print(agent.run("What is (4.6 + 6.685) x 4, and then subtract 3.14 from the result?"))
# -> The result is 42.0.

print(agent.trajectory.runs)
# ->
"""
[
    {
        "query": "What is (4.6 + 6.685) x 4, and then subtract 3.14 from the result?",
        "steps": [
            Step(
                thought="",
                action={"tool": "add", "kwargs": {"a": "4.6", "b": "6.685"}},
                observation="11.285",
                answer=None,
                metadata=None,
            ),
            Step(
                thought="",
                action={"tool": "multiply", "kwargs": {"a": "11.285", "b": "4"}},
                observation="45.14",
                answer=None,
                metadata=None,
            ),
            Step(
                thought="",
                action={"tool": "subtract", "kwargs": {"a": "45.14", "b": "3.14"}},
                observation="42.0",
                answer=None,
                metadata=None,
            ),
            Step(
                thought="",
                action=None,
                observation=None,
                answer="The result is 42.0.",
                metadata={
                    "model": "gemma4:e4b",
                    "prompt_tokens": 323,
                    "completion_tokens": 9,
                },
            ),
        ],
    }
]
"""

for index, step in enumerate(agent.trajectory.runs[0]["steps"]):
    print(f"-- Step {index+1} ---")
    if step.action:
        print(f"Tool: {step.action}")
    else:
        print(f"Answer: {step.answer}")
# ->
"""
-- Step 1 ---
Tool: {'tool': 'add', 'kwargs': {'a': '4.6', 'b': '6.685'}}
-- Step 2 ---
Tool: {'tool': 'multiply', 'kwargs': {'a': '11.285', 'b': '4'}}
-- Step 3 ---
Tool: {'tool': 'subtract', 'kwargs': {'a': '45.14', 'b': '3.14'}}
-- Step 4 ---
Answer: The result is 42.0.
"""
