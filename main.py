import os
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === CARGA DOCUMENTOS ===
docs_dir = "docs"
chunks, referencias = [], []

for filename in os.listdir(docs_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
            texto = f.read()
            palabras = texto.split()
            for i in range(0, len(palabras), 30):
                fragmento = " ".join(palabras[i:i+30])
                chunks.append(fragmento)
                referencias.append(filename)

# === VECTOR BERT ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedder.encode(chunks)

index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(chunk_embeddings)

# === GPT HUGGING FACE ===
gpt_model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(gpt_model_id)
model = AutoModelForCausalLM.from_pretrained(gpt_model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === FUNCIONES ===
def buscar_chunks_relevantes(pregunta, k=3):
    pregunta_emb = embedder.encode([pregunta])
    _, indices = index.search(pregunta_emb, k)
    return [(chunks[i], referencias[i]) for i in indices[0]]

def construir_prompt(pregunta, resultados):
    contexto = "\n".join(
        [f"[{i+1}] ({doc}) \"{frag}\"" for i, (frag, doc) in enumerate(resultados)]
    )
    return f"""Estos son fragmentos de documentos relacionados con tu pregunta:

{contexto}

Pregunta: {pregunta}

Responde usando los fragmentos y cita de qué documento viene la información.
"""

def generar_respuesta(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# === EJECUCIÓN ===
if __name__ == "__main__":
    pregunta = input("❓ Escribe tu pregunta: ")
    resultados = buscar_chunks_relevantes(pregunta)
    prompt = construir_prompt(pregunta, resultados)
    respuesta = generar_respuesta(prompt)
    print("\n📄 Respuesta generada:\n")
    print(respuesta)
