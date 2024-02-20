from apps.handle.knowledge.knowledge_utils import create_knowledge, embedding_model
import os


if __name__ == '__main__':
    if not os.path.exists('knowledge_base/vector_db'):
        os.mkdir('knowledge_base/vector_db')
    print('+------------------------------------------------+\n'
          '|                开始构建向量数据库                |\n'
          '|当前使用的embedding模型为{embedding_model:^24}|\n'
          '+------------------------------------------------+'.format(embedding_model=embedding_model))
    create_knowledge(dir_path=r'knowledge_base/content', kg_name='test')
    print('+------------------------------------------------+\n'
          '|                     构建成功                    |\n'
          '+------------------------------------------------+')
