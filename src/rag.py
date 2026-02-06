from src.retrieve import retrieve
from src.llm import generate

def answer(query):
    docs = retrieve(query)
    context = "\n\n".join(docs)
    return generate(context, query)
