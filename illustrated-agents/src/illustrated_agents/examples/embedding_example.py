from illustrated_agents.llm import EmbeddingModel

# Initialize EmbeddingGemma
embedding_model = EmbeddingModel(model="embeddinggemma")

# Create embeddings
embedding_a = embedding_model.embed("I love flamingos.")
embedding_b = embedding_model.embed("Dolphins use echolocation.")
embedding_c = embedding_model.embed("Flamingos are pink birds.")

# A と B の間のコサイン類似度を計算する
dot_ab = sum(x * y for x, y in zip(embedding_a, embedding_b))
norm_a = sum(x * x for x in embedding_a) ** 0.5
norm_b = sum(x * x for x in embedding_b) ** 0.5
similarity_ab = dot_ab / (norm_a * norm_b)

# A と C の間のコサイン類似度を計算する
dot_ac = sum(x * y for x, y in zip(embedding_a, embedding_c))
norm_c = sum(x * x for x in embedding_c) ** 0.5
similarity_ac = dot_ac / (norm_a * norm_c)

print(f"AとBの類似性: {similarity_ab}") # -> AとBの類似性: 0.35435322214952075
print(f"AとCの類似性: {similarity_ac}") # -> AとCの類似性: 0.6387643797699217