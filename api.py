
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from itsdangerous import (
    URLSafeTimedSerializer,
    BadSignature,
    SignatureExpired
)
from pydantic import BaseModel, Field

from main import generate_answer
from memory import (
    create_conversation,
    get_history,
    save_message
)
from rag.context import RAGService


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("digital_twin_api")

rag_service = None

# 24 hours
HISTORY_MAX_AGE = 24 * 60 * 60

secret = os.getenv("CONVERSATION_SECRET")

if not secret or len(secret) < 32:
    raise RuntimeError(
        "Set a strong CONVERSATION_SECRET "
        "in the backend .env file."
    )

token_signer = URLSafeTimedSerializer(
    secret_key=secret,
    salt="digital-twin-conversations-v1"
)


# ==========================================
# CONVERSATION ACCESS
# ==========================================

def create_access_token(conversation_id):
    return token_signer.dumps({
        "conversation_id": conversation_id
    })


def verify_access_token(authorization):
    """
    Verify a signed conversation token.
    Its embedded conversation ID is trusted
    only after signature verification.
    """

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Conversation access token required."
        )

    scheme, separator, token = authorization.partition(" ")

    if (
        not separator
        or scheme.lower() != "bearer"
        or not token.strip()
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header."
        )

    try:
        payload = token_signer.loads(
            token.strip(),
            max_age=HISTORY_MAX_AGE
        )

    except SignatureExpired:
        raise HTTPException(
            status_code=401,
            detail="Conversation token expired."
        )

    except BadSignature:
        raise HTTPException(
            status_code=401,
            detail="Invalid conversation token."
        )

    conversation_id = payload.get("conversation_id")

    if not isinstance(conversation_id, str):
        raise HTTPException(
            status_code=401,
            detail="Invalid conversation token payload."
        )

    return conversation_id


# ==========================================
# APPLICATION LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service

    logger.info("Loading Digital Twin RAG...")
    rag_service = RAGService()
    logger.info("Digital Twin API ready.")

    yield

    rag_service = None


app = FastAPI(
    title="Digital Twin AI API",
    version="1.1.0",
    lifespan=lifespan
)


# ==========================================
# CORS
# ==========================================

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"]
)


# ==========================================
# REQUEST AND RESPONSE MODELS
# ==========================================

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000
    )


class SourceInfo(BaseModel):
    section: str
    source: str


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    access_token: str
    provider: str
    sources: list[SourceInfo]


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    conversation_id: str
    access_token: str
    messages: list[HistoryMessage]


# ==========================================
# HEALTH
# ==========================================

@app.get("/api/health")
def health_check():
    return {
        "status": (
            "ready"
            if rag_service is not None
            else "initializing"
        ),
        "service": "Digital Twin AI"
    }


# ==========================================
# RESTORE HISTORY
# ==========================================

@app.get(
    "/api/history",
    response_model=HistoryResponse
)
def get_conversation_history(
    authorization: str | None = Header(default=None)
):

    conversation_id = verify_access_token(
        authorization
    )

    messages = get_history(conversation_id)

    if not messages:
        raise HTTPException(
            status_code=410,
            detail="Conversation history expired."
        )

    return HistoryResponse(
        conversation_id=conversation_id,
        access_token=create_access_token(
            conversation_id
        ),
        messages=[
            HistoryMessage(
                role=message["role"],
                content=message["content"]
            )
            for message in messages
            if message.get("role") in {
                "user", "assistant"
            }
        ]
    )


# ==========================================
# CHAT
# ==========================================

@app.post(
    "/api/chat",
    response_model=ChatResponse
)
def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None)
):

    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="AI service is not ready."
        )

    question = request.message.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty."
        )

    try:
        if authorization:
            conversation_id = verify_access_token(
                authorization
            )

            # Do not silently reuse a
            # conversation whose Redis history
            # has already expired.
            if not get_history(conversation_id):
                raise HTTPException(
                    status_code=410,
                    detail="Conversation expired."
                )

        else:
            conversation_id = create_conversation()

        result = generate_answer(
            question=question,
            conversation_id=conversation_id,
            rag=rag_service
        )

        answer = result["answer"]

        save_message(
            conversation_id,
            "user",
            question
        )

        save_message(
            conversation_id,
            "assistant",
            answer
        )

        sources = [
            SourceInfo(
                section=source.get(
                    "section", "Unknown"
                ),
                source=source.get(
                    "source", "Unknown"
                )
            )
            for source in result["sources"]
        ]

        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            access_token=create_access_token(
                conversation_id
            ),
            provider=result["provider"],
            sources=sources
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Chat request failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to generate a response. "
                "Please try again."
            )
        )
