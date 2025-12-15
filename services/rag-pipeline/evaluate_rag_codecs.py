import json
import sys
import os
import dotenv
import re
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add rag directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag'))
from ask_rag_aws import AskRAGAWS

dotenv.load_dotenv()

def parse_score(score_str):
    match = re.search(r'\b(10|[0-9](\.[0-9]+)?)\b', score_str)
    if match:
        return float(match.group(1))
    return 0.0

def evaluate_rag_codecs(eval_data_path, output_path):
    with open(eval_data_path, 'r') as f:
        eval_data = json.load(f)
        
    print("Initializing RAG (Codecs)...")
    rag = AskRAGAWS()
    
    llm = ChatBedrock(
        model_id="openai.gpt-oss-120b-1:0",
        model_kwargs={"temperature": 0},
        region_name="us-east-1",
        aws_access_key_id=os.getenv("aws_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_key")
    )
    
    # Relevance Metric
    relevance_prompt = ChatPromptTemplate.from_template("""
    You are an expert judge. Evaluate if the predicted answer is relevant to the question.
    
    Question: {question}
    Predicted Answer: {predicted_answer}
    
    Rate the relevance on a scale of 0 to 10, where 10 is perfectly relevant and answers the question directly, and 0 is completely irrelevant.
    
    Output ONLY the number (0-10).
    """)
    relevance_chain = relevance_prompt | llm | StrOutputParser()

    results = []
    total_top_k = 0
    total_relevance = 0
    
    print(f"Evaluating {len(eval_data)} samples...")
    
    for i, item in enumerate(eval_data):
        question = item['question']
        ground_truth = item['ground_truth']
        target_article_number = item.get('article_number')
        
        print(f"Processing {i+1}/{len(eval_data)}: {question[:50]}...")
        
        try:
            # Check Top-k Retrieval
            # Temporarily increase k to 15
            rag.retriever.search_kwargs['k'] = 15
            retrieved_docs = rag.retrieve(question)
            
            # Restore k to 2 (default in AskRAGAWS)
            rag.retriever.search_kwargs['k'] = 2
            
            rank = -1
            # Check if target article number is in retrieved docs
            # We look for "Стаття {target_article_number}" in the text
            target_str = f"Стаття {target_article_number}"
            
            for idx, doc in enumerate(retrieved_docs):
                # Check if the document starts with or contains the article header
                # The split logic preserves "Стаття X..." at the start usually
                if target_str in doc.page_content:
                    rank = idx + 1
                    break
            
            top_k_score = 0.0
            k_threshold = 3 # We can use 3 as standard even if RAG uses 2
            if rank != -1 and rank <= k_threshold:
                top_k_score = 1.0
            
            # Generate Answer
            predicted_answer = rag.ask_rag_aws(question)
            
            # Calculate Relevance
            relevance_score_str = relevance_chain.invoke({
                "question": question,
                "predicted_answer": predicted_answer
            })
            
            relevance_score = parse_score(relevance_score_str)
                
            total_top_k += top_k_score
            total_relevance += relevance_score
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "target_article_number": target_article_number,
                "rank": rank,
                "predicted_answer": predicted_answer,
                "top_k_score": top_k_score,
                "relevance_score": relevance_score
            })
            
            print(f"  Scores - Rank: {rank}, Top-3 Recall: {top_k_score}, Relevance: {relevance_score}")
            
        except Exception as e:
            print(f"Error evaluating item {i}: {e}")
            import traceback
            traceback.print_exc()
            
    rag.close()
    
    avg_top_k = total_top_k / len(results) if results else 0
    avg_relevance = total_relevance / len(results) if results else 0
    
    print("\nEvaluation Complete.")
    print(f"Average Top-3 Recall: {avg_top_k:.2f}")
    print(f"Average Relevance: {avg_relevance:.2f}/10")
    
    final_output = {
        "metrics": {
            "average_top_3_recall": avg_top_k,
            "average_relevance": avg_relevance
        },
        "details": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    evaluate_rag_codecs('eval_data_codecs.json', 'eval_results_codecs.json')
