# llm-application

> 学习如何基于大模型开发应用

## 🚀 Quick Start
项目结构如下，下面这个例子中使用的LLM为Baichuan2-13B-Chat，embedding模型为bge-large-zh-v1.5
```
├───backend
│   ├───apps
│   │   ├───config
│   │   ├───core
│   │   └───handle
│   └───knowledge_base
│       ├───content
│       └───vector_db
└───pretrained_models
    ├───Baichuan2-13B-Chat
    └───bge-large-zh-v1.5
```
### Step 1
按照上述的文件树存放好你要使用的模型，包括问答模型和embedding模型（现在还没有加入在线embedding的功能，所以后者是必要的）
### Step 2
进入后端的目录
```
cd backend
```
并安装后端所需要的依赖
```
pip install -r requirements.txt
```
### Step 3
如果此时你还没有初始化过向量数据库，将文件放入**backend/knowledge_base/content/{知识库名称}**中，然后执行以下代码
```
python init_faiss.py
```
### Step 4(可选)
如果你要使用本地模型，则在修改config.py文件后，执行以下代码
```
python init_model.py
```
### Step 5
修改main.py并运行
