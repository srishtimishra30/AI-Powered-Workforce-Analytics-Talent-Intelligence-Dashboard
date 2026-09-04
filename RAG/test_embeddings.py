from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Employees working excessive hours may have higher retention risk."

embedding = model.encode(text)


print("Embedding generated successfully!")
print("Embedding type:", type(embedding))
print("Embedding dimensions:", len(embedding))

print("\nFirst 10 values:")
print(embedding[:10])
