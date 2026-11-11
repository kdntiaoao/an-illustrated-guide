from illustrated_agents.llm import LLM

test_cases = [
    ("III", 3),
    ("IV", 4),
    ("IX", 9),
    ("LVIII", 58),
    ("MCMXCIV", 1994),
    ("MMMCMXCIX", 3999),
    ("", 0),
    ("IIII", 0),       # invalid: four in a row
    ("VV", 0),         # invalid: V can't repeat
    ("IC", 0),         # invalid: I can only subtract from V or X
    ("ABC", 0),        # invalid: non-Roman characters
    ("MMMM", 0),       # invalid: exceeds 3999
]

# コードを評価する関数
def verify(answer):
    try:
        # 機能を抽出する
        namespace = {}
        exec(answer, namespace)
        fn = namespace["roman_to_int"]

        # テストケースで関数を評価する
        score = sum(fn(inp) == expected for inp, expected in test_cases) / len
        (test_cases)
        return score
    except Exception:
        # コードが有効な Python でない場合は 0 を返す
        return 0.0

llm = LLM(model="gemma3:12b", temperature=1)

responses = []
for _ in range(10):
    # **モデルに尋ねたい質問**
    # ローマ数字の文字列を整数に変換する Python 関数 `roman_to_int(s)` を作成します。関数は次のことを行う必要があります。
    # - 1 から 3999 までの標準ローマ数字を処理します。
    # - 減算表記を正しく処理します (例: IV=4、IX=9、XL=40、XC=90、CD=400、CM=900)。
    # - 無効な入力 (空の文字列、非ローマ字、または「IIII」や「VV」などの無効なパターン) の場合は 0 を返します。
    # 関数の定義のみを指定し、説明やマークダウン形式は指定しません。
    query = """Write a Python function `roman_to_int(s)` that converts a Roman
    numeral
string to an integer. The function should:
- Handle standard Roman numerals from 1 to 3999.
- Correctly handle subtractive notation (e.g., IV=4, IX=9, XL=40, XC=90, CD=400,
CM=900).
- Return 0 for invalid inputs (empty string, non-Roman characters, or invalid
patterns
  like "IIII" or "VV").

Give only the function definition, no explanation or markdown formatting."""

    # Generate response
    response = llm.generate([{"role": "user", "content": query}])

    # Verify the response and compute a score
    score = verify(response.content)

    # Store the response and its score
    responses.append((response.content, score))

# Get highest score and the corresponding answer
best_answer, best_score = max(responses, key=lambda x: x[1])

# Show all scores
[response[1] for response in responses]