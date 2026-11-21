from illustrated_agents.llm import LLM, Response
from illustrated_agents.trajectory import Trajectory
from illustrated_agents.memory import Memory
from illustrated_agents.tools import Tools
from illustrated_agents.planning import ReAct


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

    def run(self, task: str, image_data: str = None) -> str:
        """タスクに対してエージェントを実行します"""
        self.memory.add("user", task, image_data=image_data)
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
