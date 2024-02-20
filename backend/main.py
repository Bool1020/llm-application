from apps.handle.chat.api_rpc import chat

if __name__ == '__main__':
    for i, _ in chat('达美康的国内适应症是什么？缓释制剂初始剂量是多少？', is_stream=True):
        print(i)
