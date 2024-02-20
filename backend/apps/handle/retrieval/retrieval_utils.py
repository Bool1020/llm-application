class Search(object):
    def __init__(self, db, top_k=5):
        self.retriever = db.as_retriever(search_kwargs={"k": top_k})

    def __call__(self, query):
        return self.retriever.get_relevant_documents(query)

    def search_for_content(self, query):
        docs = self(query)
        for i, doc in enumerate(docs):
            docs[i] = doc.page_content
        return docs

    def search_for_qa(self, query):
        docs = self(query)
        for i, doc in enumerate(docs):
            docs[i] = {'question': doc.page_content, 'answer': doc.metadata['answer']}
        return docs

    def search_for_docs(self, query):
        docs = self(query)
        docs_set = set()
        for i, doc in enumerate(docs):
            docs_set.add(doc.metadata['source'])
        return docs_set

