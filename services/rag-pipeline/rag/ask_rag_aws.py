import weaviate
from langchain_weaviate import WeaviateVectorStore
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import boto3
import dotenv
import os

class AskRAGAWS:
    def __init__(self):
        dotenv.load_dotenv()

        # --- 1. Connect to Local Weaviate ---
        self.client = weaviate.connect_to_local(
            grpc_port=50051
        )
        self.client.is_ready()

        # --- 2. Define Class Name ---
        class_name = "CohereBedrockEmbed"

        # --- 3. Initialize Embedding Model ---
        print("Loading Cohere embedding model on AWS Bedrock...")
        embeddings = BedrockEmbeddings(
            model_id="cohere.embed-multilingual-v3",
            region_name="us-east-1",
            aws_access_key_id=os.getenv("aws_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_key")
        )
        print("Embedding model loaded.")

        # --- 4. Connect to Existing Vector Store ---
        vector_store = WeaviateVectorStore(
            client=self.client,
            index_name=class_name,
            embedding=embeddings,
            text_key="text"
        )

        # --- 5. Initialize AWS Bedrock LLM ---
        aws_region = "us-east-1"
        boto3.setup_default_session(region_name=aws_region)

        llm = ChatBedrock(
            model_id="openai.gpt-oss-120b-1:0",
            model_kwargs={"temperature": 0.1},
            region_name=aws_region,
            aws_access_key_id=os.getenv("aws_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_key")
        )

        # --- 6. Create Retriever ---
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        # --- 7. Define RAG Chain ---
        template = """
        You are an assistant for question-answering tasks.
        Answer the question based ONLY on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs[:2]])

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
    def retrieve(self, query):
        return self.retriever.invoke(query)
    
    def ask_rag_aws(self, query):
        return self.chain.invoke(query)

    def close(self):
        self.client.close()


if __name__ == "__main__":
    rag = AskRAGAWS()

    question = "Стаття про пошкодження авто та відповідальність за неї."
    retrieved_docs = rag.retrieve(question)
    response = rag.ask_rag_aws(question)

    print(f"Question: {question}")
    print(f"Answer: {response}")

    rag.close()
