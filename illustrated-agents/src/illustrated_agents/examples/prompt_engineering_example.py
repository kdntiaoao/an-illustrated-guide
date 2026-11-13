from illustrated_agents.llm import LLM

# Gemma 3 12B (ネイティブな思考やツールの呼び出しはありません)
llm = LLM(model="gemma3:12b", temperature=1)

"""
# one-shot プロンプトの例
prompt_example = "I saw 6 flamingos. 2 flew away. 1 hid behind a tree. How many can I see? Be concise."  # 6羽のフラミンゴを見ました。2羽飛んでいきました。 1匹は木の陰に隠れました。何羽見れるでしょうか？簡潔にしてください。
prompt_answer = "3"

# モデルに尋ねたい質問
query = "I saw 9 penguins. 2 slid in the water and disappeared from sight while 4 waddled up from the shore. How many can I see?"  # 9羽のペンギンを見ました。2匹は水の中で滑って見えなくなり、４匹はよちよちと岸から上がってきた。何匹見れるでしょうか？

# Messages
messages = [
    {"role": "user", "content": prompt_example},
    {"role": "assistant", "content": prompt_answer},
    {"role": "user", "content": query},
]

# Generate response
response = llm.generate(messages)
print(response)
# -> Response(content='7', reasoning=None, tool_call=None, metadata={'model': 'gemma3:12b', 'prompt_tokens': 84, 'completion_tokens': 2})
# ※ 間違った回答になる
"""

"""
prompt_example = "I saw 6 flamingos. 2 flew away. 1 hid behind a tree. How many can I see?"  # 6羽のフラミンゴを見ました。 2匹飛んでいきました。1匹は木の陰に隠れました。何個見れるでしょうか？
prompt_answer = "You saw 6 flamingos 2 flew away. 6 - 2 = 4. Then, 1 flamingo hid. You see now 4 - 1 = 3."  # 6羽のフラミンゴ2が飛び去ったのが見えました。 6 - 2 = 4。すると、1 匹のフラミンゴが隠れました。 4 - 1 = 3 であることがわかります。
query = "I saw 9 penguins. 2 slid in the water and disappeared from sight while 4 waddled up from the shore. How many can I see?"  # 9羽のペンギンを見ました。2匹は水の中で滑って見えなくなり、４匹はよちよちと岸から上がってきた。何匹見れるでしょうか？

# Messages
messages = [
    {"role": "user", "content": prompt_example},
    {"role": "assistant", "content": prompt_answer},
    {"role": "user", "content": query},
]

# Generate response
response = llm.generate(messages)
print(response)
# -> Response(content="Okay, let's break it down:\n\n*   You started with 9 penguins.\n*   2 slid into the water and you can't see them: 9 - 2 = 7\n*   4 waddled up from the shore, and you *can* see them: 7 + 4 = 11\n\nSo you can see **11** penguins!", reasoning=None, tool_call=None, metadata={'model': 'gemma3:12b', 'prompt_tokens': 119, 'completion_tokens': 84})
# ※ 正しい回答になる
"""

# zero-shot プロンプトの例
query = "I saw 9 penguins. 2 slid in the water and disappeared from sight while 4 waddled up from the shore. How many can I see?"  # 9羽のペンギンを見ました。2匹は水の中で滑って見えなくなり、４匹はよちよちと岸から上がってきた。何匹見れるでしょうか？

# Messages
messages = [
    {"role": "user", "content": query + " Let's think step by step."},
]

# Generate response
response = llm.generate(messages)
print(response)
# -> Response(content="Okay, let's break this down step by step:\n\n1. **Start:** You initially see 9 penguins.\n2. **Penguins disappear:** 2 penguins slid into the water and are no longer visible, so 9 - 2 = 7 penguins.\n3. **Penguins appear:** 4 penguins waddled up from the shore, so now you see 7 + 4 = 11 penguins.\n\n**Answer:** You can see 11 penguins.\n", reasoning=None, tool_call=None, metadata={'model': 'gemma3:12b', 'prompt_tokens': 51, 'completion_tokens': 105})
