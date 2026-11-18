import re
from dataclasses import dataclass
from typing import Callable
from illustrated_agents.llm import LLM
from illustrated_agents.memory import Memory
from illustrated_agents.tools import NativeTools
from illustrated_agents.planning import NativeReAct
from illustrated_agents.agent import TinyAgent

# Type hint for the scorers
##  (prediction: str, example: dict) -> bool | float
Scorer = Callable[[str, dict], bool | float]


@dataclass
class Benchmark:
    name: str
    examples: list[dict]
    scorer: Callable


class Evaluator:
    """ベンチマークに対して TinyAgent を実行し、結果を集計する"""

    def __init__(self, create_agent: Callable):
        """新しいエージェントインスタンスを作成する関数を使用して初期化する"""
        self.create_agent = create_agent

    def run(self, benchmark: Benchmark) -> dict:
        """ベンチマークの例でエージェントを実行し、結果をスコア付けする"""

        # Run each example and collect results
        results = []
        for example in benchmark.examples:
            agent = self.create_agent()
            prediction = agent.run(example["task"]) or ""
            passed = benchmark.scorer(prediction, example)
            results.append(
                {
                    "prediction": prediction,
                    "passed": passed,
                }
            )

        # Aggregate pass rate
        if results:
            pass_rate = sum(result["passed"] for result in results) / len(results)
        else:
            pass_rate = 0.0

        # 詳細な結果と全体的な合格率を返す
        return {
            "name": benchmark.name,
            "pass_rate": pass_rate,
            "results": results,
        }


# Gemma 4 E4B (with native thinking and tool calling)
llm = LLM(model="gemma4:e4b", think=True)


def create_agent():
    """Create a new instance of TinyAgent"""
    return TinyAgent(
        llm=llm,
        memory=Memory(),
        tools=NativeTools(),
        planner=NativeReAct(),
    )


def exact_match_scorer(prediction: str, example: dict) -> bool:
    """答えが予測と一致する場合は True を返し、そうでない場合は False を返す"""
    match = re.search(r"\b([A-J])\b", prediction.upper())
    return match.group(1) == example["expected"]


# MMLU Pro の 3 つの例
mmlu_pro = Benchmark(
    name="MMLU Pro",
    examples=[
        {
            "task": """Which of the following is the body cavity that contains
the pituitary gland?
A) Ventral B) Dorsal C) Buccal D) Thoracic E) Pericardial F) Abdominal
G) Spinal H) Pelvic I) Pleural J) Cranial
Answer with only the letter.""",
            "expected": "J",
        },
        {
            "task": """What is the approximate mean cranial capacity of
Homo erectus?
A) 1200 cc B) under 650 cc C) 1700 cc D) 1350 cc E) just under 1000 cc
F) 1500 cc G) under 500 cc H) about 800 cc I) just over 1100 cc J) about 900 cc
Answer with only the letter.""",
            "expected": "E",
        },
        {
            "task": """
According to Moore’s "ideal utilitarianism," the right action is the one that
brings about the greatest amount of:
A) wealth. B) virtue. C) fairness. D) pleasure. E) peace. F) justice.
G) happiness. H) power. I) good. J) knowledge.
Answer with only the letter.""",
            "expected": "I",
        },
    ],
    scorer=exact_match_scorer,
)

# Run evaluation
result = Evaluator(create_agent).run(mmlu_pro)
print(result)
# ->
"""
{
    "name": "MMLU Pro",
    "pass_rate": 0.3333333333333333,
    "results": [
        {"prediction": "J", "passed": True},
        {"prediction": "J", "passed": False},
        {"prediction": "G", "passed": False},
    ],
}
"""
