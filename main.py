import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = "llama3.1"
local_llm = OllamaLLM(base_url=OLLAMA_URL, model=MODEL_NAME)

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    prompt = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
    response = local_llm.invoke(prompt)
    return JSONResponse(content={
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 1234567890,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response},
            "finish_reason": "stop"
        }],
    })

@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        content={
            "data": [
                {
                    "id": "langchain",
                    "object": "model",
                    "owned_by": "user"
                }
            ],
            "object": "list"
        }
    )
