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


# 完全一致
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
# result = Evaluator(create_agent).run(mmlu_pro)
# print(result)
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


# プログラムによる検証
def programmatic_scorer(prediction: str, example: dict) -> bool:
    """Check a prediction against its related check."""
    return example["check"](prediction)


# Three examples from IFeval
ifeval = Benchmark(
    name="IFeval",
    examples=[
        {
            "task": "Write me a funny song with less than 10 sentences for a proposal to build a new playground at my local elementary school.",
            "check": lambda text: sum(1 for c in text if c in ".!?") < 10,
        },
        {
            "task": "Write an ad copy for a new product, a digital photo frame that connects to your social media accounts and displays your photos. Respond with at most 150 words.",
            "check": lambda text: len(text.split()) <= 150,
        },
        {
            "task": "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style.  You are not allowed to use any commas in your response.",
            "check": lambda text: "," not in text,
        },
    ],
    scorer=programmatic_scorer,
)

# Run evaluation
result = Evaluator(create_agent).run(ifeval)
print(result)
# ->
"""
{
    "name": "IFeval",
    "pass_rate": 1.0,
    "results": [
        {
            "prediction": '**(Set to a bouncy, quick tune like "Pop Goes the Weasel")**\n\nOh, the current playground is frankly a fright!\nWith rusty spin cycles and dull, tired sights.\nOur students demand more than chipped wood and dirt,\nThey need giant slides where they can just burst!\nSo give us a jungle gym, a soaring new climb!\nLet\'s build a great play place, one ring and one lime!\nFor laughter and grins, let the grand chaos begin!\n(Seriously, we really need better swings, am I wrong?)',
            "passed": True,
        },
        {
            "prediction": "**🏡 Never Miss a Memory. Never Stop the Joy. Introducing [Product Name]!**\n\nIs your perfect photo gallery trapped in the cloud? Stop scrolling through thousands of pictures and start living them.\n\n[Product Name] is the revolutionary digital photo frame that bridges the gap between your social life and your living room. Simply connect your favorite social media accounts, and let our frame do the heavy lifting.\n\nInstantly display family milestones, vacation highlights, and friend updates as they happen. This isn't just a display—it's a real-time memory curator, ensuring your loved ones are always front and center.\n\nImagine the warmth of your best memories, perfectly displayed, 24/7. Set it up once, and let the memories flow.\n\n✨ **More than just photos. It's connection, delivered.** ✨\n\n**Pre-order yours today and transform your space! [Website Link]**",
            "passed": True,
        },
        {
            "prediction": "Hark gentle traveler lend thy ear to this scroll\nA journey grand doth take thee far from soul.\nThis pilgrimage to Nippon’s storied land\nA tale of wonder grasped by guiding hand.\n\n***\n\n**THE SIX DAY GRAND ITINERARY**\n\n**DAY ONE TOKYO'S FIERY BREATH**\nThou shalt behold Tokyo a mighty maze\nWhere neon splendor lights all mortal days.\nBright towers rise a sight beyond compare\nGreat crowds doth throng with industry and care.\nAnon ye walk in Ginza's lavish square\nA modern spectacle beyond all prayer.\nEat sushi famed a feast of fish and spice\nAnd witness haste a marvelous device.\n\n**DAY TWO KYOTO'S SILKEN DECREE**\nFrom Tokyo’s frenzy seek ye quieter air\nTo Kyoto’s temples held with ancient prayer.\nHere zen serenity doth claim the day\nWhere ancient mossy wonders hold their sway.\nShinto shrines stand guarded by the stone\nTheir timeless quiet utterly is known.\nYe stroll Through gardens deep a perfect sight\nAnd bathe thy spirit in reflective light.\n\n**DAY THREE ARASHIYAMA AND BAMBOO'S CALL**\nToday the mighty Bamboo forest waits\nA green cathedral guarded by fate.\nYe walk the stalks that reach the heaven's dome\nA whispering forest far away from home.\nFushimi Inari’s great gatekeepers glow\nA thousand gates of vermillion overflow.\nAscend ye paths where fox spirits hold reign\nAnd feel the mystic beauty of the rain.\n\n**DAY FOUR HAKONE MOUNT'S HIGH SOAR**\nTo Fuji’s foothills journey ye afar\nWhere mystic waters shine beneath a star.\nHakone’s views a splendor all sublime\nYe seek the smoke a wonder beyond time.\nYe sail the lake where Hakone’s great mist drifts\nA scene of drama that the soul uplifted.\nBehold the mountain vast a sleeping god\nWhose silent glory doth the traveler laud.\n\n**DAY FIVE NARA'S GRACE A DEER'S SOFT HEAR**\nA day is spent in Nara where the deer\nDo roam like noble kings held ever near.\nThese gentle beasts with wisdom seem to speak\nA holy counsel that the hearts do seek.\nYe see the halls where massive Buddha sleeps\nA solemn watch the ancient temple keeps.\nA perfect peace in Nara ye shall find\nLeaving the modern noise and stress behind.\n\n**DAY SIX FAREWELL TILL ANOTHER TIME**\nThe final sun does set in golden gleam\nA waking vision a magnificent dream.\nWith memories rich and journeys finely done\nThy Nippon quest hath reached its noble run.\nGo hence good friend with heart both full and bright\nAnd carry forth this memory pure as light.",
            "passed": True,
        },
    ],
}
"""
