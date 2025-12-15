import pandas as pd
import json
import os
import dotenv
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

import re

def generate_eval_data(csv_path, output_path, num_samples=20):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, nrows=num_samples)
    
    llm = ChatBedrock(
        model_id="openai.gpt-oss-120b-1:0",
        model_kwargs={"temperature": 0.7}, # Higher temp for generation
        region_name="us-east-1",
        aws_access_key_id=os.getenv("aws_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_key")
    )

    prompt_template = """You are an expert legal assistant. 
    Given the following summary of a Supreme Court decision, generate a realistic question that a legal professional might ask to find this case or understand its ruling.
    The question should be answerable based ONLY on the information in the summary.
    
    Summary: {summary}
    
    Output format:
    Question: [Your question here]
    Answer: [Your answer here]
    
    Do not include any reasoning or other text. Only the Question and Answer.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    
    eval_data = []
    
    print("Generating evaluation pairs...")
    for index, row in df.iterrows():
        summary = row['summary']
        case_number = row['case_number']
        
        if not isinstance(summary, str) or len(summary) < 10:
            continue
            
        try:
            response = chain.invoke({"summary": summary})
            
            # Parse response using regex
            question_match = re.search(r"Question:\s*(.*?)(?=Answer:|$)", response, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r"Answer:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
            
            if question_match and answer_match:
                question = question_match.group(1).strip()
                answer = answer_match.group(1).strip()
                
                # Clean up if answer contains reasoning block at the end
                if "</reasoning>" in answer:
                     answer = answer.split("</reasoning>")[-1].strip()

                # Validation
                if len(question) < 15 or not any(c.isalpha() for c in question):
                    print(f"Skipping invalid question for case {case_number}: {question}")
                    continue

                eval_data.append({
                    "case_number": case_number,
                    "summary": summary,
                    "question": question,
                    "ground_truth": answer
                })
                print(f"Generated pair for case {case_number}")
            else:
                print(f"Failed to parse response for case {case_number}: {response[:100]}...")
                
        except Exception as e:
            print(f"Error processing case {case_number}: {e}")
            
    with open(output_path, 'w') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(eval_data)} evaluation pairs to {output_path}")

if __name__ == "__main__":
    generate_eval_data(
        csv_path='2024-5K-supreme-court-decisions-analyzed.csv',
        output_path='eval_data.json',
        num_samples=20
    )
