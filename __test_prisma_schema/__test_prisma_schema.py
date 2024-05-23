from pypdf import PdfReader
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


# reading the pdf file and splitting pages
reader = PdfReader("./__test_prisma_schema/Prisma Schema.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]


_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

openai_client = OpenAI()


# making gpt from a modle that remebers facts into a model that processes information
def rag(pdf_texts, model="gpt-4"):
    information = "\n\n".join(pdf_texts)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful database designer. You have designed a database for SAAS, B2B sustainability solution. Delimited by three backticks."
            "Make a human readable explanation of the database schema."
            "Use exact filed names and types"
            "explain the relationships between the tables with the exact field names and types."
            f"```{information}```"
        },
    ]
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content


output = rag(pdf_texts=pdf_texts)

print(output)
