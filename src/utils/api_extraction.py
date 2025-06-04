# given that our server is running (`ollama serve`):
from ollama import ChatResponse, chat
from pydantic.types import JsonSchemaValue
from typing import Optional
from json import loads

"""
Adapted from: https://tollefj.folk.ntnu.no/books/local-llm/content/01-landing.html
"""


def generate(
    system_prompt: str,
    prompt: str,
    model: str,
    schema: Optional[JsonSchemaValue] = None,
    parse: bool = True,
    num_ctx: int = 32000,
    num_predict: int = 4000,
    temperature: float = 0.0,
) -> str:
    response: ChatResponse = chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        options={
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_k": 100,
            "top_p": 0.8,
            "temperature": temperature,
            "seed": 0,  # this is not needed when temp is 0
            "repeat_penalty": 1.3,  # remain default for json outputs, from experience.
        },
        format=schema,
        stream=False,
    )
    res = response.message.content
    if parse and schema:
        try:
            res = loads(res)
        except Exception:
            res = None
    return res