from typing import Any, List, Optional, Dict, Iterator, Mapping, Type, Sequence, Union, Callable
from ..chat.api_rpc import model_message, post
from ...config.model_config import model_config
import requests
import json
from langchain_core.language_models import LanguageModelInput
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel, generate_from_stream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "user", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict.get("content", "") or "")
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = _dict["tool_calls"]

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return model_config.base_model

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response, _ = next(model_message(prompt))
        return response


class CustomChat(BaseChatModel):
    streaming: bool = False
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def _default_params(self) -> Dict[str, Any]:
        normal_params = {
            "model": model_config.base_model,
            "stream": self.streaming,
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def _llm_type(self) -> str:
        return model_config.base_model

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        res = self._chat(messages, **kwargs)
        if res.status_code != 200:
            raise ValueError(f"Error from response: {res}")
        response = res.json()
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        res = self._chat(messages, **kwargs)
        if res.status_code != 200:
            raise ValueError(f"Error from Baichuan api response: {res}")
        default_chunk_class = AIMessageChunk
        for chunk in res.iter_lines():
            chunk = chunk.decode("utf-8").strip("\r\n")
            parts = chunk.split("data: ", 1)
            chunk = parts[1] if len(parts) > 1 else None
            if chunk is None:
                continue
            if chunk == "[DONE]":
                break
            response = json.loads(chunk)
            for m in response.get("choices"):
                chunk = _convert_delta_to_message_chunk(
                    m.get("delta"), default_chunk_class
                )
                default_chunk_class = chunk.__class__
                cg_chunk = ChatGenerationChunk(message=chunk)
                yield cg_chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
        res = post([_convert_message_to_dict(m) for m in messages], self.streaming)
        return res

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for c in response["choices"]:
            message = _convert_dict_to_message(c["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)

        token_usage = response["usage"]
        llm_output = {"token_usage": token_usage, "model": model_config.base_model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
        function_call: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        from langchain.chains.openai_functions.base import convert_to_openai_function

        formatted_functions = [convert_to_openai_function(fn) for fn in functions]
        if function_call is not None:
            if len(formatted_functions) != 1:
                raise ValueError(
                    "When specifying `function_call`, you must provide exactly one "
                    "function."
                )
            if formatted_functions[0]["name"] != function_call:
                raise ValueError(
                    f"Function call {function_call} was specified, but the only "
                    f"provided function was {formatted_functions[0]['name']}."
                )
            function_call_ = {"name": function_call}
            kwargs = {**kwargs, "function_call": function_call_}
        return super().bind(
            functions=formatted_functions,
            **kwargs,
        )


llm = CustomChat()
