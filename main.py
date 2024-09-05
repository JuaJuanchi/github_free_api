import os
import dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import uvicorn
from typing import List

# Configure loguru
logger.add("file_{time}.log", rotation="7 day")
# Load environment variables
app = FastAPI()

try:
    dotenv.load_dotenv()
except ImportError:
    logger.info("No environment variables found")
# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Define Pydantic models
class Message(BaseModel):
    role: str
    content: str

class RequestData(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False

class EmbeddingRequest(BaseModel):
    model: str
    input: str | List[str]

# Initialize OpenAI client
token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Hello World"}

@app.post("/v1/chat/completions")
async def chat_completions(request_data: RequestData):
    logger.info(f"Received chat request: {request_data}")
    try:
        if request_data.stream:
            # Define an asynchronous generator for streaming response
            async def event_stream():
                stream_response = client.chat.completions.create(
                    model=request_data.model,
                    messages=[message.dict() for message in request_data.messages],
                    stream=True
                )
                logger.info("Response:\n")
                for update in stream_response:
                    content = update.choices[0].delta.content if update.choices[0].delta.content else ""
                    logger.info(f"{content}")
                    yield f"data: {update.json()}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            # Non-streaming response
            response = client.chat.completions.create(
                model=request_data.model,
                messages=[message.dict() for message in request_data.messages],
                stream=False
            )
            logger.info(f"Response: {response}")
            return response
    except Exception as err:
        logger.error(f"Error occurred in chat completion: {err}")
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/v1/embeddings")
async def create_embedding(request_data: EmbeddingRequest):
    logger.info(f"Received embedding request: {request_data}")
    try:
        response = client.embeddings.create(
            model=request_data.model,
            input=request_data.input
        )
        logger.info(f"Embedding created successfully")
        return response
    except Exception as err:
        logger.error(f"Error occurred in embedding creation: {err}")
        raise HTTPException(status_code=500, detail=str(err))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
