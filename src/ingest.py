from PyPDF2 import PdfReader
from pathlib import Path

def load_docs(path="data/docs"):
    texts = []
    for file in Path(path).glob("*"):
        if file.suffix == ".pdf":
            reader = PdfReader(file)
            texts.append("\n".join(p.extract_text() for p in reader.pages))
        elif file.suffix == ".txt":
            texts.append(file.read_text())
    return texts

def chunk(text, size=500, overlap=100):
    words = text.split()
    return [
        " ".join(words[i:i+size])
        for i in range(0, len(words), size-overlap)
    ]
