from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import os
import logging
import time
from groq import Groq
from typing import Optional, List
from dotenv import load_dotenv
from jose import JWTError, jwt

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GENIT Chat API", version="3.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://mernat.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = 1.0
    max_tokens: int = 1024
    top_p: float = 1.0
    stream: bool = True
    system_prompt: Optional[str] = "You are a helpful AI assistant."

class ChatResponse(BaseModel):
    response: str
    processing_time: float

SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload  
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def generate_stream(messages, temperature, max_tokens, top_p):
    try:
        if messages[0].role != "system":
            messages.insert(0, Message(
                role="system",
                content="You are a helpful AI assistant."
            ))

        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
            stop=None,
        )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                yield f"data: {chunk.choices[0].delta.content}\n\n"
        
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}")
        yield f"data: Error: {str(e)}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest, token: str = Depends(oauth2_scheme)):
    try:
        verify_token(token)  # Verifies the token before processing the request
        
        if request.stream:
            return StreamingResponse(
                generate_stream(
                    request.messages,
                    request.temperature,
                    request.max_tokens,
                    request.top_p
                ),
                media_type="text/event-stream"
            )
        else:
            start_time = time.time()
            
            if request.messages[0].role != "system":
                request.messages.insert(0, Message(
                    role="system",
                    content=request.system_prompt
                ))

            formatted_messages = [{"role": m.role, "content": m.content} for m in request.messages]
            
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=formatted_messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stream=False,
            )
            
            response_text = completion.choices[0].message.content
            processing_time = time.time() - start_time
            
            return ChatResponse(
                response=response_text,
                processing_time=processing_time
            )
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

@app.get("/health")
async def health_check(token: str = Depends(oauth2_scheme)):
    try:
        verify_token(token)  # Verifies the token before responding with health status
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
            stream=False
        )
        return {
            "status": "healthy",
            "model": "llama3-8b-8192",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")
