
import os
import uuid
from dotenv import load_dotenv
from upstash_redis import Redis
import json

load_dotenv()

redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
)

SESSION_TTL = 86400  # 24 hours


def create_conversation():
    conversation_id = str(uuid.uuid4())
    return conversation_id

def save_message(conversation_id, role, content):
    key = f"chat:{conversation_id}"

    message = {
        "role": role,
        "content": content
    }

    redis.rpush(key, json.dumps(message))
    redis.expire(key, SESSION_TTL)


def get_history(conversation_id):
    key = f"chat:{conversation_id}"

    messages = redis.lrange(key, 0, -1)

    return [json.loads(msg) for msg in messages]


def delete_conversation(conversation_id):
    key = f"chat:{conversation_id}"
    redis.delete(key)
