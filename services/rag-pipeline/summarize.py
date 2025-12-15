import os
import json
import dotenv
import pandas as pd
import concurrent.futures
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm.auto import tqdm

def extract_analysis_fields(analysis_str):
    try:
        #extract with regex  ```json   ```
        import re
        match = re.search(r'```json(.*?)```', analysis_str, re.DOTALL)
        if match:       
            analysis_str = match.group(1).strip()
        else:
            analysis_str = analysis_str.strip() 
        analysis_json = json.loads(analysis_str)
        return pd.Series({
            'case_number': analysis_json.get('case_number', None),
            'date': analysis_json.get('date', None),
            'court_name': analysis_json.get('court_name', None),
            'article_codes_and_documents': analysis_json.get('article_codes_and_documents', None),
            'summary': analysis_json.get('summary', None)
        })
    except Exception as e:
        return pd.Series({
            'case_number': None,
            'date': None,
            'court_name': None,
            'article_codes_and_documents': None,
            'summary': None
        })

def summarize_legal_decisions(df):
    dotenv.load_dotenv()

    aws_region = "us-east-1"
    llm = ChatBedrock(
        model_id="openai.gpt-oss-120b-1:0",
        model_kwargs={"temperature": 0.1},
        region_name=aws_region,
        aws_access_key_id=os.getenv("aws_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_key")
    )
    template = """
    You are an expert legal assistant. Analyze the following court decision text.
    Extract the following information and return it in valid JSON format:
    {{
        "case_number": "The case number",
        "date": "The date of the decision",
        "court_name": "Name of the court in Ukrainian language",
        "article_codes_and_documents": "codes and docs of relevant legal articles mentioned (as a list) in Ukrainian language",
        "summary": "A concise summary of the decision in English"
    }}

    Text:
    {text}

    JSON Output:
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    def analyze_text(text):
        if not text:
            return None
        try:
            truncated_text = text[:15000]
            return chain.invoke({"text": truncated_text})
        except Exception as e:
            return f"Error: {str(e)}"

    tqdm.pandas()

    # Parallelize the analysis
    with concurrent.futures.ThreadPoolExecutor(max_workers=13) as executor:
        df['analysis'] = list(tqdm(executor.map(analyze_text, df['text']), total=len(df), desc="Analyzing"))
    
    
    analysis_fields = df['analysis'].progress_apply(extract_analysis_fields)
    df = pd.concat([df, analysis_fields], axis=1)

    return df


if __name__ == "__main__":
    df = pd.read_parquet('2024-5K-supreme-court-decisions-deduplicated.parquet')
    df = summarize_legal_decisions(df)
    df.to_csv('2024-5K-supreme-court-decisions-analyzed.csv', index=False)
    