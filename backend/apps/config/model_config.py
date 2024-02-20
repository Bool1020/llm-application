api_key = {
    'baichuan': 'your_apikey'
}


chat_config = {
    'local': {
        'Baichuan2-13B-Chat': {
            'ip': '0.0.0.0',
            'port': 5001,
            'need_generation_config': True
        },
        'Qwen-14B-Chat': {
            'ip': '0.0.0.0',
            'port': 5001,
            'need_generation_config': False
        },
        'chatglm3-6b': {
            'ip': '0.0.0.0',
            'port': 5001,
            'need_generation_config': False
        }
    },
    'online': {
        'Baichuan2-Turbo': {
            'url': 'https://api.baichuan-ai.com/v1/chat/completions',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + api_key['baichuan']
            }
        },
        'Baichuan2-Turbo-192k': {
            'url': 'https://api.baichuan-ai.com/v1/chat/completions',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + api_key['baichuan']
            }
        },
        'Baichuan2-53B': {
            'url': 'https://api.baichuan-ai.com/v1/chat/completions',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + api_key['baichuan']
            }
        }
    }
}


class ModelConfig:
    embedding_model = 'bge-large-zh-v1.5'
    base_model = 'Baichuan2-Turbo'
    is_online = base_model in chat_config['online']


model_config = ModelConfig()
