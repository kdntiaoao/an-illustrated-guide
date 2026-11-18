import re
import numpy as np
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


# ---------------------------------------------------------
# Exact match (完全一致)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Programmatic checks (プログラムによる検証)
# ---------------------------------------------------------
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
# result = Evaluator(create_agent).run(ifeval)
# print(result)
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


# ---------------------------------------------------------
# LLM-as-a-judge (審査員としてのLLM)
# ---------------------------------------------------------
judge = LLM(model="qwen3.5:9b")


def judge_scorer(prediction: str, example: dict) -> bool:
    """The LLM-as-a-judge scorer."""
    prompt = f"""
Score the response from 0.0 to 1.0.

Expected: {example["expected"]}
Response: {prediction}

Reply with only a single number.
"""
    response = judge.generate([{"role": "user", "content": prompt}])
    score = float(response.content.strip().split()[0])
    return score


# Three examples adapted from MMLU Pro
mmlu_pro = Benchmark(
    name="MMLU Pro",
    examples=[
        {"task": "下垂体はどの体腔にありますか？", "expected": "頭蓋腔"},
        {"task": "ホモ・エレクトスの平均頭蓋容量は？", "expected": "1000cc弱"},
        {"task": "ムーアの理想功利主義で正しい行為とは？", "expected": "善"},
    ],
    scorer=judge_scorer,
)

# Run evaluation
# result = Evaluator(create_agent).run(mmlu_pro)
# print(result)
# ->
"""
{
    "name": "MMLU Pro",
    "pass_rate": 0.9333333333333332,
    "results": [
        {
            "prediction": "下垂体（pituitary gland）は、一般的な意味での「体腔」（胸腔、腹腔など）に入っているという分類はしません。\n\n**最も正確に言うと、下垂体は「頭蓋骨（ずがいこつ）の中、脳の付け根（脳底）の部分」にあります。**\n\n### 詳しい説明\n\n1. **体腔（Body Cavity）との関係**\n   「体腔」とは、臓器を保護するために身体内に形成されている空間（例：肋骨の中の胸腔、腹部の腹腔など）を指します。下垂体は、体外の腔に存在するのではなく、**頭の中の非常に限られた空間**に位置しています。\n\n2. **解剖学的な位置**\n   下垂体は、**蝶形骨（ちょうけいこつ）という頭蓋骨の一部にある「鞍（くら）」と呼ばれるくぼみ（sella turcica）**に収まっています。\n\n3. **機能的な関係**\n   下垂体は、視床下部（シスラヘル）という脳の一部と非常に密接に機能的なつながり（下垂体-視床下部系）を持っている、特殊な内分泌腺です。\n\n**【まとめ】**\n* **伝統的な体腔分類**では当てはまりません。\n* **最も正確な位置**は、**頭蓋骨の脳の付け根（鞍）**です。",
            "passed": 0.9,
        },
        {
            "prediction": "ホモ・エレクトスの頭蓋容量について、**「これだけ」と断言できる単一の平均値は存在しません**。なぜなら、ホモ・エレクトスは非常に広範囲（アフリカ、アジア、ヨーロッパなど）で発見されているため、その地域や時代によって化石の特性が異なるからです。\n\nしかし、一般的な科学的知見に基づくと、平均的な推定値や範囲は以下のように示されています。\n\n### 🔬 一般的な推定値\n\n**約 700 cc 〜 1200 cc** の範囲で推測されています。\n\n*   **補足:**\n    *   早期の推定では700cc前後が主流でしたが、近年、特定の化石群や種を対象とした研究では、これよりも大きい容量を示すものも見られています。\n    *   この容量は、初期のホモ・ハビリス（*Homo habilis*）や後期旧人（ネアンデルタール人など）の平均的な頭蓋容量と比較すると、**「中期的な中間値」**にあたります。\n\n### 🌍 比較のための目安\n\n| 区分 | 頭蓋容量（目安） | 特徴 |\n| :--- | :--- | :--- |\n| **初期のホモ属** (例: ホモ・ハビリス) | 500 〜 700 cc | 脳が小さく、工具の使用が始まった時期。 |\n| **ホモ・エレクトス** | **700 〜 1200 cc** | 非常に広く分布し、火の使用や長距離移動など、高度な活動を開始した種。 |\n| **現生人類** (*Homo sapiens*) | 1200 〜 1600 cc以上 | 現代人の標準的な容量。 |\n\n**要約すると、「700ccから1200cc程度」という広い幅で捉えていただくのが最も正確です。**",
            "passed": 0.9,
        },
        {
            "prediction": "これは非常に高度で哲学的な質問であり、「ムーア（G.E. Moore）」という哲学者をどの側面で捉えるかによって、回答のニュアンスが大きく変わってきます。\n\nなぜなら、G.E.ムーアは、**「功利主義そのもの」を提唱した人物というよりも、むしろ功利主義や倫理学の前提を徹底的に批判した人物**だからです。\n\nしかし、ここでは「ムーアの倫理学が目指す、理想的かつ洗練された行為の決定方法」という視点から、正しい行為を定義します。\n\n---\n\n### 結論：ムーア的な倫理が示す「正しい行為」の定義\n\nムーアの視点から導かれる「正しい行為」とは、**単なる快楽の最大化を目指すのではなく、「真に善いもの（Good）」が発揮される結果をもたらし、それによって全集団の幸福（Well-being）を最大化する行為**です。\n\nこれは「結果主義」（コンシクエンシャリズム）に分類されますが、単なる「快楽最大化」とは一線を画す、より高次の幸福を目指します。\n\n### １. 正しい行為を決定する３つの原則\n\nムーア的アプローチは、以下の要素を総合的に評価し、最も「真の善」を最大化する選択肢を選ぶことを要求します。\n\n#### ① 最大多数の最大幸福（総体的な効用の最大化）\n最も基本的な原則です。行為の結果が、関わるすべての人々にとっての幸福（効用）の合計を最大化する方向に向かっているかを見ます。\n\n*   **評価のポイント:** 特定の一人にとっての苦痛や犠牲が、全体的な利益によって上回っているか。\n*   **例:** 少数派の利益を犠牲にしても、大多数の飢餓を救う行為は、総体的な効用を高めるため、正しいと判断されます。\n\n#### ② 感情的な快楽への依存からの脱却（非自然主義的アプローチ）\nここが、ムーア哲学の最も重要なポイントです。\n単に「気分が良い」「気持ちがいい」という感情的な快楽（hedonism）を基準にするのではなく、**「それが真に良い行為なのか？」という客観的な尺度**を用いて評価します。\n\nムーアは、「『善』とは快楽である」という単純な命題（自然主義的誤謬）が論理的に成り立たないことを鋭く指摘しました。したがって、正しい行為は、単なる快楽中毒的な行動（例：一夜限りの贅沢）ではなく、**倫理的かつ知的な充足感**を伴う行為である必要があります。\n\n#### ③ 真の善（Real Good）の実現\n正しい行為とは、一時的な快楽ではなく、自己や他者が「充実している」「満たされている」と感じられるような、高次の価値（真の善）を体現する行為である必要があります。\n\n*   **例：*単なる快楽* vs *真の善***\n    *   ただ薬を飲んで一時的に痛みを忘れること（一時的快楽）\n    *   苦しい努力をして社会に貢献し、深い自己肯定感と充足感を得ること（真の善）\n\nムーア的な倫理は、後者の方を「正しい行為」として高く評価します。\n\n---\n\n### ２. まとめ：正しい行為の決定プロセス\n\nムーア的倫理において「正しい行為」を判断するプロセスは、単なる感情や直感ではなく、**「効用計算と、善の質的評価」**を組み合わせた複雑なプロセスです。\n\n1.  **行為の特定:** 選択肢A、B、Cを洗い出す。\n2.  **結果の予測:** 各行為がもたらすすべての結果（幸福、苦痛、満足度など）を予測する。\n3.  **効用の計算（量的な評価）:** 予測された結果を総合的に比較し、最大多数の最大幸福に貢献する可能性が高いものを絞り込む。\n4.  **「善の質」の評価（質的な評価）：** 絞り込まれた選択肢が、単なる刺激や快楽ではなく、「真に価値あるもの」「倫理的であるもの」に結びついているか（ムーアの批判をクリアしているか）を吟味する。\n5.  **結論:** 量と質の両方で最も高いスコアを出したものが、「ムーアの理想功利主義」における正しい行為となります。\n\n**【極めて重要な補足】**\nご質問の「ムーアの理想功利主義」という言葉は、哲学用語としては厳密には存在しません。\n\n*   もし「功利主義（Utilitarianism）」全般を指すのであれば、上記の**「総体の幸福最大化」**が答えです。\n*   もし「ムーア（G.E. Moore）」を指すのであれば、彼は功利主義を根本的に批判し、**「『善』は快楽では測れない、もっと複雑で、客観的で、超越的な性質を持つものだ」**という立場を取っているため、「単なる行為のリスト」を出すのではなく、**「倫理的な直感や概念を、純粋な快楽という単純な尺度で還元することを拒否する倫理学」**として理解するのが適切です。",
            "passed": 1.0,
        },
    ],
}
"""


# ---------------------------------------------------------
# pass@k
# ---------------------------------------------------------
def pass_at_k(n_samples, n_correct_samples, k):
    """
    :param n_samples: total number of samples
    :param n_correct_samples: number of correct samples
    :param k: k in pass@k
    """
    if n_samples - n_correct_samples < k:
        return 1.0
    return 1.0 - np.prod(
        1.0 - k / np.arange(n_samples - n_correct_samples + 1, n_samples + 1)
    )


n_samples = 10
n_correct_samples = 0
task_example = {"task": "ホモ・エレクトスの平均頭蓋容量は？", "expected": "1000cc弱"}

for i in range(n_samples):
    agent = create_agent()  # 毎回新しいエージェントを作る
    prediction = agent.run(task_example["task"]) or ""

    # スコアラー（LLM-as-a-judgeなど）で正解判定
    score = judge_scorer(prediction, task_example)

    if score >= 1.0:  # 完璧に正解した場合
        n_correct_samples += 1

    # --- 実行結果の確認用 print 追加 ---
    print(f"--- [試行 {i + 1} / {n_samples}] ---")
    print(f"  問題: {task_example['task']}")
    print(f"  期待値: {task_example['expected']}")
    print(f"  エージェントの答え: {prediction}")
    print(f"  評価スコア: {score}")

print("---")
print(f"Success count: {n_correct_samples}/{n_samples}")
# -> Success count: 0/10

pass_1 = pass_at_k(n_samples, n_correct_samples, k=1)
print(f"pass@1:  {pass_1:.4f}")
# -> pass@1:  0.0000

pass_10 = pass_at_k(n_samples, n_correct_samples, k=10)
print(f"pass@10: {pass_10:.4f}")
# -> pass@10: 0.0000
