import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.messages import AIMessage
from starlette.responses import StreamingResponse

from app.agent.nodes import agent


@asynccontextmanager
async def lifespan(s: FastAPI):
    print("Starting")
    yield

server = FastAPI(title="rag_agent", lifespan=lifespan)


@server.post('/answer')
def answer(message: str):
    input = {'messages': [message]}
    async def stream_generator():
        async for chunk in agent.astream(input, stream_mode='values'):
            message = chunk['messages'][-1]
            if message:
                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        continue
                    if message.content:
                        payload = json.dumps({
                            "type": "info",
                            "role": "system",
                            "content": message.content
                        })
                        yield f"data: {payload}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(server, host="0.0.0.0", port=8000, loop="asyncio")