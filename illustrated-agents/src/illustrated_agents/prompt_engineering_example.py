from illustrated_agents.llm import LLM

# Gemma 3 12B (ネイティブな思考やツールの呼び出しはありません)
llm = LLM(model="gemma3:12b", temperature=1)

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
