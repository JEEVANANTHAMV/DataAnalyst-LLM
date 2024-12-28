import os
from typing import Dict, Literal, Final

from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic



# This is uniquely identified by the model name and the API key
def get_groq_client(model_name, max_tokens=1000) -> ChatGroq:
    # TODO remove hardcoded key
    return ChatGroq(
        model=model_name,
        groq_api_key="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",
    )


def get_anthropic_client(model_name, max_tokens=1000) -> ChatAnthropic:
    return ChatAnthropic(model=model_name, max_tokens=max_tokens)


def get_llm_client_info(
    client_name: Literal["azure_openai", "openai", "groq", "gemini", "SONNET"]
) -> Dict:
    client_info = {
        "groq": {
            "client_name": "groq",
            "api_key": "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",
        },
        "SONNET": {
            "client_name": "anthropic",
            "model": "claude-3-5-sonnet-20240620",
        },
    }

    return client_info[client_name]


class AvailableModels:
    VISUO_GPT_35_TURBO: Final = "visuo-gpt-35-turbo"
    LLAMA3_70B_8192: Final = "llama3-70b-8192"
    GPT_4O: Final = "gpt-4o-2024-08-06"
    GPT_4O_MINI: Final = "gpt-4o-mini-2024-07-18"
    SONNET: Final = "claude-3-5-sonnet-20240620"
    GEMINI_PRO: Final = "gemini-pro"
    GPT_4_TURBO: Final = "gpt-4-turbo"
    GEMINI_PRO_15: Final = "gemini-1.5-pro"
    GEMINI_FLASH_15: Final = "gemini-1.5-flash"
    GEMINI_PRO_15_EX: Final = "gemini-pro-experimental"
    GEMINI_FLASH_15_EX: Final = "gemini-flash-experimental"
