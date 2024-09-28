import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, Response

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()

headers = {
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    'content-type': 'application/json',
    'origin': 'https://apps.abacus.ai',
    'priority': 'u=1, i',
    'reai-ui': '1',
    'referer': 'https://apps.abacus.ai/chatllm/',
    'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'x-abacus-org-host': 'apps'
}
models_info_abacus = {
    "gpt-4o-mini": {
        "deploymentId": "c7fab66ee",
        "externalApplicationId": "1611427b5c",
        "llmName": "OPENAI_GPT4O_MINI",
    },
    "gpt-4": {
        "deploymentId": "c7fab66ee",
        "externalApplicationId": "1598d72ad2",
        "llmName": "OPENAI_GPT4_128K_LATEST",
    },
    "gpt-4o": {
        "deploymentId": "c7fab66ee",
        "externalApplicationId": "13b729e8aa",
        "llmName": "OPENAI_GPT4O_LATEST",
               },
    "o1-preview": {
        "deploymentId": "c7fab66ee",
        "externalApplicationId": "13d9c1197e",
        "llmName": "OPENAI_O1",
    },
    "o1-mini": {
        "deploymentId": "c7fab66ee",
        "externalApplicationId": "13a61ed8f4",
        "llmName": "OPENAI_O1_MINI",
    },
    "claude-3-5-sonnet": {
        "deploymentId": "c7fab66ee",
        "externalApplicationId": "142f953934",
        "llmName": "CLAUDE_V3_5_SONNET",
                          },
    "RouteLLM": {
        "deploymentId": "c7fab66ee",
        "externalApplicationId": "296d569f0",
        "llmName": "ROUTE_LLM",
               },
    "SearchLLM": {
        "deploymentId": "bd1ce4fc8",
        "externalApplicationId": "166888117e",
        "llmName": "OPENAI_GPT4O_LATEST",
               },
    "llama-3.1-405b": {
        "deploymentId": "c7fab66ee",
        "externalApplicationId": "181e006fa",
        "llmName": "LLAMA3_1_405B",
               },
    "gemini-1.5-pro-002": {
        "deploymentId": "c7fab66ee",
        "externalApplicationId": "14a80089be",
        "llmName": "GEMINI_1_5_PRO",
               },
    "Abacus.AI-Smaug": {
        "deploymentId": "c7fab66ee",
        "externalApplicationId": "15206bda48",
        "llmName": "ABACUS_SMAUG3",
    },
    "llama-3.1-70b": {
        "deploymentId": "c7fab66ee",
        "externalApplicationId": "1598d72ad2",
        "llmName": "LLAMA3_1_70B",
    },
}

APP_SECRET = os.getenv("APP_SECRET", "666")
COOKIES = os.getenv("COOKIES", "")
ALLOWED_MODELS = [
    {"id": "gpt-4o-mini", "name": "gpt-4o-mini"},
    {"id": "gpt-4o", "name": "gpt-4o"},
    {"id": "gpt-4", "name": "gpt-4"},
    {"id": "o1-preview", "name": "o1-preview"},
    {"id": "o1-mini", "name": "o1-mini"},
    {"id": "claude-3-5-sonnet", "name": "claude-3-5-sonnet"},
    {"id": "RouteLLM", "name": "RouteLLM"},
    {"id": "SearchLLM", "name": "SearchLLM"},
    {"id": "llama-3.1-405b", "name": "llama-3.1-405b"},
    {"id": "gemini-1.5-pro-002", "name": "gemini-1.5-pro-002"},
    {"id": "llama-3.1-70b", "name": "llama-3.1-70b"},
    {"id": "Abacus.AI-Smaug", "name": "Abacus.AI-Smaug"},
]
# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，您可以根据需要限制特定源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)
security = HTTPBearer()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False


def get_cookies(cookie_str):
    cookies = dict(item.partition('=')[::2] for item in cookie_str.split('; '))
    keys_remain = ['_a_p', '_ss_p', '_u_p', '_s_p']
    filtered_cookies = {key: cookies[key] for key in keys_remain if key in cookies}
    return filtered_cookies


def simulate_data(content, model):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "role": "assistant"},
                "finish_reason": None,
            }
        ],
        "usage": None,
    }


def stop_data(content, model):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }
    
    
def create_chat_completion_data(content: str, model: str, finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "role": "assistant"},
                "finish_reason": finish_reason,
            }
        ],
        "usage": None,
    }


def verify_app_secret(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != APP_SECRET:
        raise HTTPException(status_code=403, detail="Invalid APP_SECRET")
    return credentials.credentials


@app.options("/hf/v1/chat/completions")
async def chat_completions_options():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )


def replace_escaped_newlines(input_string: str) -> str:
    return input_string.replace("\\n", "\n")


async def create_conversation(model, cookies):
    url = 'https://apps.abacus.ai/cluster-proxy/api/createDeploymentConversation'
    data = {
        "deploymentId": models_info_abacus[model]['deploymentId'],
        "name": "New Chat",
        "externalApplicationId":  models_info_abacus[model]['externalApplicationId']
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, cookies=cookies, json=data)
            response.raise_for_status()  # Raises stored HTTPError, if one occurred.
            json_response = response.json()
            logger.info("Deployment Conversation Created Successfully")
            if json_response["success"]:
                return json_response["result"]["deploymentConversationId"]
        except httpx.HTTPStatusError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")


async def delete_conversation(cookies, deploymentId, deploymentConversationId):
    url = "https://apps.abacus.ai/cluster-proxy/api/deleteDeploymentConversation"
    payload = {
        "deploymentId": deploymentId,
        "deploymentConversationId": deploymentConversationId
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, cookies=cookies, json=payload)
            response.raise_for_status()  # Raises stored HTTPError, if one occurred.
            if response.json()["success"]:
                logger.info("Delete Conversation Successfully")
        except httpx.HTTPStatusError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")


@app.get("/hf/v1/models")
async def list_models():
    return {"object": "list", "data": ALLOWED_MODELS}


@app.get("/cluster-proxy/api/downloadAgentAttachment")
async def downloadAgentAttachment(request: Request = None):
    cookies = get_cookies(COOKIES)
    original_url = request.url
    new_scheme = "https"
    new_hostname = "apps.abacus.ai"
    parsed_url = urlparse(str(original_url))
    dst_url = parsed_url._replace(scheme=new_scheme, netloc=new_hostname).geturl()
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(dst_url, headers=headers, cookies=cookies)
            response.raise_for_status()
            logger.info("Get File Successfully")
            header = {"Content-Type": response.headers.get("Content-Type"),
                       "Content-Disposition": response.headers.get("Content-Disposition")}
            return Response(content=response.content, headers=header)
        except httpx.HTTPStatusError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")


@app.post("/hf/v1/chat/completions")
async def chat_completions(
    request: ChatRequest, app_secret: str = Depends(verify_app_secret),
    raw_request: Request = None
):
    logger.info(f"Received chat completion request for model: {request.model}")

    if request.model not in [model['id'] for model in ALLOWED_MODELS]:
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} is not allowed. Allowed models are: {', '.join(model['id'] for model in ALLOWED_MODELS)}",
        )

    cookies = get_cookies(COOKIES)
    deploymentConversationId = await create_conversation(request.model,cookies)
    # 使用 OpenAI API
    json_data = {
        "requestId": str(uuid.uuid4()),
        "deploymentConversationId": deploymentConversationId,
        "message": "\n".join(
            [
                f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                for msg in request.messages
            ]
        ),
        "chatConfig": {
            "timezone": "Asia/Hong_Kong",
            "language": "zh-CN"
        },
        "llmName": models_info_abacus[request.model]['llmName'],
        "externalApplicationId": models_info_abacus[request.model]['externalApplicationId']
    }
    host = raw_request.url.hostname
    port = raw_request.url.port
    scheme = raw_request.url.scheme

    async def generate():
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream('POST', 'https://apps.abacus.ai/api/_chatLLMSendMessageSSE', headers=headers, cookies=cookies, json=json_data, timeout=120.0) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line and (not json.loads(line).get("end")):
                            content = json.loads(line)
                            if content.get("type") == "image_url":
                                markdown_url = f"\n ![{content.get('imgGenerationPrompt', '')}]({content.get('segment', '')}) \n"
                                yield f"data: {json.dumps(create_chat_completion_data(markdown_url, request.model))}\n\n"
                            if content.get("type") == "text":
                                yield f"data: {json.dumps(create_chat_completion_data(content.get('segment', ''), request.model))}\n\n"
                            if content.get("type") == "attachments":
                                file_list = content.get("attachments")
                                for file in file_list:
                                    if port:
                                        attachments_url = f"\n [{file.get('filename')}]({scheme}://{host}:{port}/cluster-proxy/api/downloadAgentAttachment?deploymentId={models_info_abacus[request.model]['deploymentId']}&attachmentId={file.get('attachment_id')}) \n"
                                    else:
                                        attachments_url = f"\n [{file.get('filename')}]({scheme}://{host}/cluster-proxy/api/downloadAgentAttachment?deploymentId={models_info_abacus[request.model]['deploymentId']}&attachmentId={file.get('attachment_id')}) \n"
                                yield f"data: {json.dumps(create_chat_completion_data(attachments_url, request.model))}\n\n"
                    yield f"data: {json.dumps(create_chat_completion_data('', request.model, 'stop'))}\n\n"
                    yield "data: [DONE]\n\n"
                    await delete_conversation(cookies, models_info_abacus[request.model]["deploymentId"],
                                              deploymentConversationId)
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error occurred: {e}")
                await delete_conversation(cookies, models_info_abacus[request.model]["deploymentId"],
                                          deploymentConversationId)
                raise HTTPException(status_code=e.response.status_code, detail=str(e))
            except httpx.RequestError as e:
                logger.error(f"An error occurred while requesting: {e}")
                await delete_conversation(cookies, models_info_abacus[request.model]["deploymentId"],
                                          deploymentConversationId)
                raise HTTPException(status_code=500, detail=str(e))


    if request.stream:
        logger.info("Streaming response")
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        logger.info("Non-streaming response")
        full_response = ""
        async for chunk in generate():
            if chunk.startswith("data: ") and not chunk[6:].startswith("[DONE]"):
                # print(chunk)
                data = json.loads(chunk[6:])
                if data["choices"][0]["delta"].get("content"):
                    full_response += data["choices"][0]["delta"]["content"]
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_response},
                    "finish_reason": "stop",
                }
            ],
            "usage": None,
        }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
