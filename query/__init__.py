from typing import Dict, Literal, Final

from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic



# This is uniquely identified by the model name and the API key
def get_groq_client(model_name, max_tokens=1000) -> ChatGroq:
    # TODO remove hardcoded key
    return ChatGroq(
        model=model_name,
        groq_api_key="",
    )


def get_anthropic_client(model_name, max_tokens=1000) -> ChatAnthropic:
    return ChatAnthropic(model=model_name, max_tokens=max_tokens,api_key='')


def get_llm_client_info(
    client_name: Literal["azure_openai", "openai", "groq", "gemini", "SONNET"]
) -> Dict:
    client_info = {
        "groq": {
            "client_name": "groq",
            "api_key": "",
        },
        "SONNET": {
            "client_name": "anthropic",
            "model": "claude-3-5-sonnet-20240620",
        },
    }

    return client_info[client_name]


class AvailableModels:
    LLAMA3_70B_8192: Final = "llama-3.3-70b-versatile"
    SONNET: Final = "claude-3-5-sonnet-20240620"
    GEMINI_FLASH_15_EX: Final = "gemini-flash-experimental"
