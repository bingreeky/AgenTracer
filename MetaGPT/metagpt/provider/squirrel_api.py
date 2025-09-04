from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator

import aiohttp
from aiohttp import ClientSession, ClientTimeout
from tenacity import retry, stop_after_attempt, wait_random, retry_if_exception_type

from metagpt.configs.llm_config import LLMConfig, LLMType 
from metagpt.logs import log_llm_stream
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.llm_provider_registry import register_provider

logger = logging.getLogger(__name__)

class SquirrelContentTypeError(RuntimeError):
    pass

@register_provider(LLMType.SQUIRREL)
class SquirrelLLM(BaseLLM):
    """MetaGPT provider that wraps Squirrel 大模型平台的 Chat 接口."""

    TIMEOUT: int = 60

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.config = config  
        self.base_url: str = config.base_url.rstrip("/")
        self.api_key: str = config.api_key
        self.model: str = config.model or "gpt-4o-mini"
        self.stream: bool = bool(config.stream)
        extra: Dict[str, Any] = config.extra_fields or {}
        self.app_name: str = extra.get("app_name", "metagpt_default_app")

        # 懒加载 session，防止 RuntimeError
        self._session: Optional[ClientSession] = None
        print(f"--------------------------------SquirrelLLM initialized with base_url: {self.base_url}, api_key: {self.api_key}, model: {self.model}, stream: {self.stream}, app_name: {self.app_name}")

    async def _get_session(self) -> ClientSession:
        """懒加载 aiohttp session，确保在 asyncio 环境内创建。"""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=64))
        return self._session

    async def _achat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return await self.acompletion_text(messages, **kwargs)

    async def _achat_completion_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        async for chunk in self.astream_completion(messages, **kwargs):
            yield chunk

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),  # 最多重试6次
        wait=wait_random(min=1, max=2),  # 每次重试间隔1-2秒
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, SquirrelContentTypeError)),
    )
    async def acompletion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        payload = self._build_payload(messages, stream=False)
        headers = self._build_headers()
        session = await self._get_session()

        async with session.post(self.base_url, json=payload, headers=headers) as resp:
            # 打印 Content-Type 以便调试
            print("Response Content-Type:", resp.headers.get('Content-Type'))

            # 如果返回的 Content-Type 不匹配，可以打印出响应内容进行检查
            content_type = resp.headers.get("Content-Type")
            if not content_type:
                # 记录日志，方便排查
                logger.error("No Content-Type in LLM response headers, headers: %s", resp.headers)
                raise SquirrelContentTypeError("No Content-Type in LLM response headers")
            if "application/json" not in content_type:
                # 打印响应体内容用于调试
                print("Unexpected Content-Type:", content_type)
                text_response = await resp.text()
                print("Response Body:", text_response)
            else:
                # 如果是 JSON 格式，正常解析
                body = await resp.json()

        if body.get("code") != 0:
            raise RuntimeError(f"Squirrel API error {body.get('code')}: {body.get('msg')}")

        answer: str = body.get("data", "")
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "model": self.model,
            "object": "chat.completion",
        }

    async def acompletion_text(self, messages: List[Dict[str, str]], **kwargs) -> str:
        resp = await self.acompletion(messages, **kwargs)
        return resp["choices"][0]["message"]["content"]

    async def astream_completion(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        payload = self._build_payload(messages, stream=True)
        headers = self._build_headers()
        session = await self._get_session()

        async with session.post(self.base_url, json=payload, headers=headers) as resp:
            async for line in resp.content:
                if not line:
                    continue
                try:
                    decoded = line.decode(errors="ignore").strip()
                    if decoded.startswith("data:"):
                        data_str = decoded[len("data:"):].strip()
                        data_json = json.loads(data_str)
                        chunk_text = data_json.get("answer", "")
                        if chunk_text:
                            yield chunk_text
                            log_llm_stream(chunk_text)
                except json.JSONDecodeError:
                    continue

    def _build_headers(self) -> Dict[str, str]:
        return {
            "authorization": self.api_key,
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: List[Dict[str, str]], stream: bool) -> Dict[str, Any]:
        if not messages:
            raise ValueError("messages is empty")

        *history, last_msg = messages
        history_chat = [
            {
                "input": m["content"],
                "output": m.get("response", "")  # Ensure there's a field to hold the model's response
            } for m in history if m["role"] in ("user", "assistant")
        ]

        prompt: str = last_msg["content"]

        inputs = {
            "msg": prompt,
            "stream": stream,
        }
        if history_chat:
            inputs["history_chat"] = history_chat
        
        return {
            "name": self.app_name,
            "model": self.model,
            "inputs": inputs,
        }

    async def aclose(self):
        if self._session and not self._session.closed:
            await self._session.close()
            await self.aclose()

    @staticmethod
    def _retry_decorator():
        return retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_random(min=1, max=2),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        )

    async def __aenter__(self):
        # 可选：初始化 session
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    def __del__(self):
        # 尽量避免在 __del__ 里做异步操作，但可以尝试关闭 session
        if hasattr(self, "_session") and self._session and not self._session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._session.close())
                else:
                    loop.run_until_complete(self._session.close())
            except Exception:
                pass
