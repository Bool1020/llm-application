from ...config.model_config import model_config, chat_config
from ...handle.retrieval.retrieval_utils import Search
from .prompt import advice, naive_rag
import requests
import json


model_name = model_config.base_model


def post(history, is_stream=False):
    data = {
        'model': model_name,
        'messages': history,
        'stream': is_stream
    }
    json_data = json.dumps(data)
    if model_config.is_online:
        response = requests.post(
            chat_config['online'][model_name]['url'],
            data=json_data,
            headers=chat_config['online'][model_name]['headers'],
            timeout=60,
            stream=is_stream
        )
    else:
        response = requests.post(
            'http://{ip}:{port}/v1/chat'.format(ip=chat_config['local'][model_name]['ip'],
                                                port=str(chat_config['local'][model_name]['port'])),
            data=json_data,
            timeout=60,
            stream=is_stream
        )
    return response


def model_message(query, history=[], is_stream=False):
    history.append(
        {
            'role': 'user',
            'content': query
        }
    )
    response = post(history, is_stream=is_stream)
    if is_stream:
        history.append({'role': 'assistant', 'content': ''})
        for line in response.iter_lines(decode_unicode=True):
            if 'data: ' in line:
                if line.replace('data: ', '') == '[DONE]':
                    break
                else:
                    result = json.loads(line.replace('data: ', ''))['choices'][0]['delta']
                    history[-1]['content'] += result['content']
                    yield result['content'], history
    else:
        result = response.json()['choices'][0]['message']
        history.append(result)
        yield result['content'], history


def chat(query, db=None, is_stream=False):
    if db:
        retriever = Search(db, 5)
        content = retriever.search_for_content(query)
        content = '\n\n'.join(content)
        if is_stream:
            return model_message(naive_rag.format(content=content, query=query), is_stream=is_stream)
        else:
            response, history = next(model_message(naive_rag.format(content=content, query=query), is_stream=is_stream))
            return response
    else:
        if is_stream:
            return model_message(query, is_stream=is_stream)
        else:
            response, history = next(model_message(query, is_stream=is_stream))
            return response
