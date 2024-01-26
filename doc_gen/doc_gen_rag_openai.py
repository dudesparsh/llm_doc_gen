# Importing the required libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI

from dotenv import load_dotenv
import os

import logging
logging.basicConfig(filename='./logs/LLM_RAG.log', level=logging.DEBUG, format='%(asctime)s-%(process)d-%(levelname)s-%(message)s') 

import warnings
warnings.filterwarnings('ignore')
logging.info('\nLogs are being generated in LLM_RAG.log \n')

# Loading the env variables
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
api_org = os.getenv('OpenAI-Organization')


def load_docs():
    """
    Method for loading FAISS vector embeddings
    """
    try:
    
        embeddings = HuggingFaceEmbeddings()
        # Loading the vector embedding stored in faiss vdb
        new_db = FAISS.load_local("faiss_index", embeddings)
        
        return embeddings, new_db
    
    except Exception as e:
        logging.debug("Exception occured in load_docs %s", e)
    
    pass

# embeddings = HuggingFaceEmbeddings()

# new_db = FAISS.load_local("faiss_index", embeddings)

def prompt_eng(prompt_no=1):
    """
    Method for performing prompt engineering
    
    ----Includes----
    Prompt_templates
    LLMs
    Chains
    
    """
    try:
       
        if prompt_no==1:
        # Following prompt is used for Job description document generation
            doc_template = """ I want you to act as a recruiter. I will provide some information about job openings, and it will be your job to come up with creating a perfect job description based on the given context. This could include rephrasing the language in a professional sense, providing the necessary details, following the proper structure of job description and if required you can make up some details on your own so that draft looks professional. The standard structure of job decription includes details such as about the company, brief about the role and responsibilies, the impact you will have, what we look for, benefits, about company, and our commitment to diversity and inclusion.  

            Following is the context you have to follow: '{context}'
            Following are the details job details: '{details}'

            Now please generate a Job decription based on the above instructions.

            Job Description: """

        prompt = PromptTemplate(template=doc_template, input_variables=["context","details"])


        llm = OpenAI(openai_api_key=api_key, openai_organization=api_org, temperature=0.9)  # model_name="text-davinci-003"

        llm_chain = LLMChain(prompt=prompt, llm=llm)

        return llm_chain    
    
    except Exception as e:
        logging.debug("Exception occured in promp_end method as %s", e)
    
    pass

def user_input():
    """
    Method for obtaining inputs from the user
    """
    deafult_details = """5 to 15 Years
    M.Tech
    $59K-$99K
    Douglas
    Isle of Man
    Machine Learning Intern
    """ 
    return deafult_details
    pass



# Module starts
def main():
    try:
        # Obtaining input details from the user
        details = user_input()

        # RAG
        embeddings, new_db = load_docs()
        docs = new_db.similarity_search(details)

        # For Keeping track of vector db query results in a list
        doclist = []
        docs_count = 0
        # Consider top 3 results only
        for _documents in docs:
            docs_count +=1
            doclist.append(_documents.page_content)
            if(docs_count>=3):
                break
        llm_chain = prompt_eng(1)
        # Generating RAG based results for top_k=3 ( RAG )
        for _results in doclist:
            print(llm_chain.run({'context': _results, 'details': details}))
            print("\n\n")

    except Exception as e:
        logging.debug("Exception occured in main as %s", e)


if __name__== "__main__":
    main()        