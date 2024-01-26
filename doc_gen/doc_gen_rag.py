# Importing the required libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI

from dotenv import load_dotenv
import os

import warnings
warnings.filterwarnings('ignore')

# Loading the env variables
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
api_org = os.getenv('OpenAI-Organization')


def load_docs():
    embeddings = HuggingFaceEmbeddings()
    # Loading the vector embedding stored in faiss vdb
    new_db = FAISS.load_local("faiss_index", embeddings)
    return embeddings, new_db
    pass

embeddings = HuggingFaceEmbeddings()

new_db = FAISS.load_local("faiss_index", embeddings)

doc_template = """ I want you to act as a recruiter. I will provide some information about job openings, and it will be your job to come up with creating a perfect job description based on the given context. This could include rephrasing the language in a professional sense, providing the necessary details, following the proper structure of job description and if required you can make up some details on your own so that draft looks professional. The standard structure of job decription includes details such as about the company, brief about the role and responsibilies, the impact you will have, what we look for, benefits, about company, and our commitment to diversity and inclusion.  

Following is the context you have to follow: '{context}'
Following are the details job details: '{details}'

Now please generate a Job decription based on the above instructions.

Job Description: """

prompt = PromptTemplate(template=doc_template, input_variables=["context","details"])


llm = OpenAI(openai_api_key=api_key, openai_organization=api_org, temperature=0.9)  # model_name="text-davinci-003"

llm_chain = LLMChain(prompt=prompt, llm=llm)

details = """5 to 15 Years
M.Tech
$59K-$99K
Douglas
Isle of Man
Machine Learning Intern
"""
# RAG
docs = new_db.similarity_search(details)

doclist = []
docs_count = 0
# Consider top 3 results only
for _documents in docs:
    docs_count +=1
    doclist.append(_documents.page_content)
    if(docs_count>=3):
      break

for _results in doclist:
  print(llm_chain.run({'context': _results, 'details': details}))
  print("\n\n")
