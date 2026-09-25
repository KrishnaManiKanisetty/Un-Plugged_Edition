"""
UNPLLM - Stage 1: Core local inference service
Wraps a local Ollama model behind a simple FastAPI endpoint.

Setup:
    1. ollama serve                  (run in a separate terminal, keep it open)
    2. ollama pull phi4-mini         (or qwen3:4b / gemma3:4b depending on your RAM)
    3. Run it EITHER of these two ways:
         python3 main.py
         uvicorn main:app --reload --port 8000
       Both do the same thing. Either way, the terminal will show
       "Uvicorn running on http://127.0.0.1:8000" and then sit there
       waiting for requests — that's correct, it's a server, not a
       one-shot script. It won't return to your prompt; open a
       second terminal to test it or stop it with Ctrl+C.

Test:
    curl -X POST http://localhost:8000/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Explain recursion in two sentences."}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from rag_utils import retrieve_chunks
from query_logger import log_query

app = FastAPI(title="UNPLLM Core Inference Service")

# Change this to match the model you pulled in Week 2
DEFAULT_MODEL = "phi4-mini"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant running fully offline."


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    max_tokens: int = 512
    model: str = DEFAULT_MODEL


@app.get("/health")
def health():
    """Quick check that the service is up and which model is default."""
    return {"status": "ok", "default_model": DEFAULT_MODEL}


@app.post("/generate")
def generate(req: GenerateRequest):
    """Send a prompt to the local model and return its response."""
    try:
        result = ollama.chat(
            model=req.model,
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.prompt},
            ],
            options={
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama call failed: {e}")

    return {
        "response": result["message"]["content"],
        "model": req.model,
        "tokens_generated": result.get("eval_count"),
        "generation_time_ns": result.get("eval_duration"),
    }


class RagRequest(BaseModel):
    prompt: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    max_tokens: int = 512
    model: str = DEFAULT_MODEL
    top_k: int = 4


@app.post("/generate_rag")
def generate_rag(req: RagRequest):
    """Retrieve relevant chunks from the local knowledge base, then
    generate an answer grounded in that context. Falls back to a
    plain (non-augmented) prompt if the knowledge base is empty."""
    retrieved, query_embedding = retrieve_chunks(req.prompt, top_k=req.top_k)

    top_distance = retrieved[0][1] if retrieved else None
    top_chunk_preview = retrieved[0][0][:150] if retrieved else ""
    is_gap = log_query(
        query_text=req.prompt,
        query_embedding=query_embedding,
        top_distance=top_distance,
        retrieved_chunk_preview=top_chunk_preview,
    )

    if retrieved:
        context = "\n\n".join(f"- {chunk}" for chunk, _ in retrieved)
        augmented_prompt = (
            f"Use the following context if it's relevant to answer the "
            f"question. If the context doesn't help, answer normally.\n\n"
            f"Context:\n{context}\n\nQuestion: {req.prompt}"
        )
    else:
        augmented_prompt = req.prompt

    try:
        result = ollama.chat(
            model=req.model,
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": augmented_prompt},
            ],
            options={
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama call failed: {e}")

    return {
        "response": result["message"]["content"],
        "model": req.model,
        "retrieved_chunks": [
            {"text": chunk[:150] + "...", "distance": round(dist, 4)}
            for chunk, dist in retrieved
        ],
        "is_knowledge_gap": is_gap,
        "tokens_generated": result.get("eval_count"),
        "generation_time_ns": result.get("eval_duration"),
    }


if __name__ == "__main__":
    # Lets `python3 main.py` start the server directly, in addition
    # to `uvicorn main:app --reload --port 8000`. Same effect either way.
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
