from gpt4all import GPT4All
from pathlib import Path

MODEL_PATH = Path("models/llm")
MODEL_NAME = "mistral-7b-instruct-v0.1.Q4_K_S.gguf"

llm = GPT4All(
    MODEL_NAME,
    model_path=MODEL_PATH,
    allow_download=False  # IMPORTANT: offline guarantee
)

def generate(context, question):
    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{question}
"""
    return llm.generate(prompt, max_tokens=300)
