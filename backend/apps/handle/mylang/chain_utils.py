from typing import Any, List, Optional
from ..chat.api_rpc import model_message
from ...config.model_config import model_config

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


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
