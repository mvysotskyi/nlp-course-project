import weaviate
from langchain_weaviate import WeaviateVectorStore
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import boto3
import dotenv
import os

class SupRemeCourtRAG:
    
    def __init__(self):
        dotenv.load_dotenv()
        CLASS_NAME = "SupremeCourtDecision"
        self.client = weaviate.connect_to_local(grpc_port=50051)
        self.embeddings = BedrockEmbeddings(
            model_id="cohere.embed-multilingual-v3",
            region_name="us-east-1",
            aws_access_key_id=os.getenv("aws_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_key")
        )
        self.vector_store = WeaviateVectorStore(
            client=self.client,
            index_name=CLASS_NAME,
            embedding=self.embeddings,
            text_key="summary", 
            attributes=["case_number", "court_name", "date", "analysis"]
        )
        self.llm = ChatBedrock(
            model_id="openai.gpt-oss-120b-1:0",
            model_kwargs={"temperature": 0.1},
            region_name="us-east-1",
            aws_access_key_id=os.getenv("aws_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_key")
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            formatted = []
            for doc in docs:
                meta = doc.metadata
                content = f"Case: {meta.get('case_number')}\nDate: {meta.get('date')}\nSummary: {doc.page_content}"
                formatted.append(content)
            return "\n\n".join(formatted)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    def ask_csv_rag(self, query):

        return self.rag_chain.invoke(query)
    
    def close(self):
        self.client.close()


if __name__ == "__main__":
    rag = SupRemeCourtRAG()
    question = "Справа Вібросепаратор"
    response = rag.ask_csv_rag(question)
    print(f"Question: {question}")
    print(f"Answer: {response}")
    rag.close()
