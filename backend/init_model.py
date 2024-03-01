import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from apps.config.model_config import model_config, chat_config
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import json
from sse_starlette import EventSourceResponse


class ChatModel(BaseModel):
    model: str
    messages: List[Dict]
    stream: bool


app = FastAPI()
model_name = model_config.base_model


print('+------------------------------------------------+\n'
      '|                  开始载入模型                  |\n'
      '|当前使用的模型为{model_name:^32}|\n'
      '+------------------------------------------------+'.format(model_name=model_name))
tokenizer = AutoTokenizer.from_pretrained(
    r'../pretrained_models/{model_name}'.format(model_name=model_name),
    use_fast=False,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    r'../pretrained_models/{model_name}'.format(model_name=model_name),
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

if chat_config['local'][model_name]['need_generation_config']:
    model.generation_config = GenerationConfig.from_pretrained(r'../pretrained_models/{model_name}'.format(model_name=model_name),)

print('+------------------------------------------------+\n'
      '|                  载入模型成功                  |\n'
      '+------------------------------------------------+')


@app.post("/v1/chat/completions")
async def chat(item: ChatModel):
    model_name = item.model
    messages = item.messages
    stream = item.stream
    content = model.chat(tokenizer, messages, stream)

    async def generator():
        for chunk in content:
            yield json.dumps({'model': model_name, 'choices': [{'delta': {'role': 'assistant', 'content': chunk}}]})
    if stream:
        return EventSourceResponse(generator())
    else:
        return {'model': model_name, 'choices': [{'message': {'role': 'assistant', 'content': content}}]}

print('+------------------------------------------------+\n'
      '|                  服务启动成功                  |\n'
      '+------------------------------------------------+')

uvicorn.run(app, host=chat_config['local'][model_name]['ip'], port=chat_config['local'][model_name]['port'])
