import json
import os
import dotenv
import re
import random
import sys
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add rag directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag'))
from pdf_reader import PDFReader

dotenv.load_dotenv()

def generate_eval_data_codecs(pdf_path, output_path, num_samples=20):
    print(f"Reading {pdf_path}...")
    pdf_reader = PDFReader()
    try:
        document = pdf_reader.read_pdf(pdf_path)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    print("Splitting document by 'Стаття' (articles)...")
    # Regex from populate_rag_aws.py
    pattern = r'(Стаття\s+\d+[^\n]*(?:\n(?!Стаття\s+\d+)[^\n]*)*)'
    matches = re.findall(pattern, document, re.MULTILINE)
    
    if not matches:
        print("No articles found with regex. Trying fallback split.")
        parts = re.split(r'(?=Стаття)', document)
        matches = [part.strip() for part in parts if part.strip() and 'Стаття' in part]

    print(f"Found {len(matches)} articles.")
    
    # Filter out very short articles or noise
    valid_articles = [m for m in matches if len(m) > 100]
    print(f"Found {len(valid_articles)} valid articles (>100 chars).")
    
    if len(valid_articles) > num_samples:
        sampled_articles = random.sample(valid_articles, num_samples)
    else:
        sampled_articles = valid_articles
        
    print(f"Selected {len(sampled_articles)} articles for generation.")

    llm = ChatBedrock(
        model_id="openai.gpt-oss-120b-1:0",
        model_kwargs={"temperature": 0.7},
        region_name="us-east-1",
        aws_access_key_id=os.getenv("aws_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_key")
    )

    prompt_template = """You are an expert legal assistant. 
    Given the following text of a legal article from a Ukrainian code, generate a realistic question in Ukrainian that a legal professional might ask to find this article or understand its content.
    The question should be answerable based ONLY on the provided text.
    The Answer must also be in Ukrainian.
    
    Article Text: {text}
    
    Output format:
    Question: [Your question in Ukrainian]
    Answer: [Your answer in Ukrainian]
    
    Do not include any reasoning or other text. Only the Question and Answer.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    
    eval_data = []
    
    print("Generating evaluation pairs...")
    for i, article_text in enumerate(sampled_articles):
        # Extract article number for reference if possible
        article_num_match = re.search(r'Стаття\s+(\d+)', article_text)
        article_num = article_num_match.group(1) if article_num_match else f"unknown_{i}"
        
        try:
            response = chain.invoke({"text": article_text})
            
            # Parse response using regex
            question_match = re.search(r"Question:\s*(.*?)(?=Answer:|$)", response, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r"Answer:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
            
            if question_match and answer_match:
                question = question_match.group(1).strip()
                answer = answer_match.group(1).strip()
                
                if "</reasoning>" in answer:
                     answer = answer.split("</reasoning>")[-1].strip()

                # Validation
                if len(question) < 15 or not any(c.isalpha() for c in question):
                    print(f"Skipping invalid question for article {article_num}")
                    continue

                eval_data.append({
                    "article_number": article_num,
                    "article_text": article_text, # Store text for reference/debugging
                    "question": question,
                    "ground_truth": answer
                })
                print(f"Generated pair for article {article_num}")
            else:
                print(f"Failed to parse response for article {article_num}")
                
        except Exception as e:
            print(f"Error processing article {article_num}: {e}")
            
    with open(output_path, 'w') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(eval_data)} evaluation pairs to {output_path}")

if __name__ == "__main__":
    pdf_path = os.path.join(os.path.dirname(__file__), 'rag/docs/codecs_ukr.pdf')

    pdf_path = 'docs/codecs_ukr.pdf'
    
    generate_eval_data_codecs(
        pdf_path=pdf_path,
        output_path='eval_data_codecs.json',
        num_samples=20
    )
