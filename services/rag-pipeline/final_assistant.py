import weaviate
from langchain_weaviate import WeaviateVectorStore
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import boto3
import dotenv
import os
import json

class FinalLegalAssistantRAG:
        
    def __init__(self):
        dotenv.load_dotenv()

        # --- Configuration ---
        CODECS_CLASS_NAME = "CohereBedrockEmbed"
        COURT_CLASS_NAME = "SupremeCourtDecision"
        AWS_REGION = "us-east-1"

        # --- 1. Connect to Local Weaviate ---
        self.client = weaviate.connect_to_local(grpc_port=50051)

        # --- 2. Initialize Embedding Model ---
        print("Loading Cohere embedding model on AWS Bedrock...")
        embeddings = BedrockEmbeddings(
            model_id="cohere.embed-multilingual-v3",
            region_name=AWS_REGION,
            aws_access_key_id=os.getenv("aws_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_key")
        )

        # --- 3. Connect to Vector Stores ---
        self.codecs_store = WeaviateVectorStore(
            client=self.client,
            index_name=CODECS_CLASS_NAME,
            embedding=embeddings,
            text_key="text"
        )

        self.court_store = WeaviateVectorStore(
            client=self.client,
            index_name=COURT_CLASS_NAME,
            embedding=embeddings,
            text_key="summary",
            attributes=["case_number", "court_name", "date", "analysis"]
        )

        # --- 4. Initialize LLM ---
        # Using Amazon Titan as fallback if Claude is not enabled
        self.llm = ChatBedrock(
            model_id="openai.gpt-oss-120b-1:0",
            model_kwargs={"temperature": 0.1},
            region_name=AWS_REGION,
            aws_access_key_id=os.getenv("aws_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_key")
        )

        # --- 5. Router ---
        router_template = """You are an expert legal assistant router. 
        Your task is to decide whether a user's query requires consulting the Criminal Code (laws, articles, definitions of crimes) or Court Decisions (precedents, specific cases, judicial reasoning).

        Query: {query}

        Return ONLY a JSON object with a key "source" цшер value either "codecs" or "court_decisions"
        and "queries" list of strings that represent sub-queries to use for the selected source. This queries should be simple, few words that would describe what the law should be about if "codecs" or a brief description of a legal case for "court_decisions".
        For example, if the query is about defining a crime, select "codecs". If it's about understanding how a law was applied in a case, select "court_decisions".
        If codecs, make subqueries that would refer to possible articles. So if user killed a dog, it should be something like ["animal cruelty", etc
        Subqueries should be in ukrainian language.
        """

        router_prompt = ChatPromptTemplate.from_template(router_template)
        self.router_chain = router_prompt | self.llm | StrOutputParser()
        final_template = """You are a helpful legal assistant. Answer the question based ONLY on the following context. 
            If the answer is not in the context, say "I cannot answer this based on the provided context."
            Do not use your internal knowledge to answer the question.

            Context:
            {context}

            Question: {question}
            """

        self.final_prompt = ChatPromptTemplate.from_template(final_template)
    # --- 6. RAG Chains ---

    def get_retriever(self, source):
        if source == "codecs":
            return self.codecs_store.as_retriever(search_kwargs={"k": 5})
        else:
            return self.court_store.as_retriever(search_kwargs={"k": 5})

    def format_docs(self, docs):
        formatted = []
        for d in docs:
            content = d.page_content
            if hasattr(d, 'metadata') and d.metadata:
                meta = d.metadata
                # Add metadata to context if it's a court decision
                if 'case_number' in meta:
                    content = f"Case: {meta.get('case_number')}\nCourt: {meta.get('court_name')}\nDate: {meta.get('date')}\nSummary: {content}\nAnalysis: {meta.get('analysis')}"
            formatted.append(content)
        return "\n\n".join(formatted)

    def run_pipeline(self, query):
        #print(f"Analyzing query: {query}")
        
        # 1. Route
        route_response = self.router_chain.invoke({"query": query})
        # try:
            # Clean up potential markdown formatting in response
        clean_response = route_response.strip().replace('```json', '').replace('```', '')
        route_data = clean_response.split('</reasoning>')[-1]
        #print(f"Router returned: {route_data}")

        route_data = json.loads(route_data)
        source = route_data.get("source") 
        queries = route_data.get("queries")

        
        # 2. Retrieve
        retriever = self.get_retriever(source)
        docs = []
        if queries:
            for q in queries:
                docs.append(retriever.invoke(q))
        
        docs_final = []
        for i in range(3):
            for doc_list in docs:
                docs_final.append(doc_list[i])
        docs = docs_final

        # Deduplicate docs
        unique_docs = {}
        for doc in docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
        docs = list(unique_docs.values())[:3]

        context = self.format_docs(docs)
        
        # 3. Generate
        chain = self.final_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": query})

        final_response = response.split('</reasoning>')[-1].strip()
        
        return final_response
    
    def close(self):
        self.client.close()

if __name__ == "__main__":
    assistant = FinalLegalAssistantRAG()
    question = "Що отримає мій друг за викрадення продукту з супермаркету?"
    response = assistant.run_pipeline(question)
    print(f"Question: {question}")
    print(f"Answer: {response}")
    assistant.close()
