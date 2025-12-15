import json
import sys
import os
import dotenv
import re
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add rag directory to path to import SupRemeCourtRAG
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag'))
from ask_csv_rag import SupRemeCourtRAG

dotenv.load_dotenv()

def parse_score(score_str):
    # Look for a number between 0 and 10
    match = re.search(r'\b(10|[0-9](\.[0-9]+)?)\b', score_str)
    if match:
        return float(match.group(1))
    return 0.0

def evaluate_rag(eval_data_path, output_path):
    with open(eval_data_path, 'r') as f:
        eval_data = json.load(f)
        
    print("Initializing RAG...")
    rag = SupRemeCourtRAG()
    
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
        target_case_number = item.get('case_number')
        
        print(f"Processing {i+1}/{len(eval_data)}: {question[:50]}...")
        
        try:
            # Check Top-k Retrieval
            # Temporarily increase k to 15 to check rank
            rag.retriever.search_kwargs['k'] = 15
            retrieved_docs = rag.retriever.invoke(question)
            
            # Restore k to 3 for generation (standard RAG behavior)
            rag.retriever.search_kwargs['k'] = 3
            
            retrieved_case_numbers = [doc.metadata.get('case_number') for doc in retrieved_docs]
            
            rank = -1
            if target_case_number in retrieved_case_numbers:
                rank = retrieved_case_numbers.index(target_case_number) + 1 # 1-based rank
            
            top_k_score = 0.0
            k_threshold = 3
            if rank != -1 and rank <= k_threshold:
                top_k_score = 1.0
            
            # Generate Answer
            predicted_answer = rag.ask_csv_rag(question)
            
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
                "target_case_number": target_case_number,
                "retrieved_case_numbers": retrieved_case_numbers,
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
    evaluate_rag('eval_data.json', 'eval_results.json')
