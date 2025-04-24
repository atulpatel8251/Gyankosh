from langchain.embeddings.openai import OpenAIEmbeddings
#__import__('pysqlite3')
import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import OpenAI
from openai import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import (
    ConversationalRetrievalChain,ConversationChain
)
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
import pytesseract
from pytesseract import Output, TesseractError
from functions import convert_pdf_to_txt_pages, convert_pdf_to_txt_file, save_pages, displayPDF, images_to_txt
from prompts import filter_prompt2,initialise_prompt,master_prompt,lang_prompt,format_prompt,ai_prompt,aiformat_prompt,mcq_test_prompt,key_term_prompt,learn_outcome_prompt,student_prompt
import chromadb
import os
import nltk
import constants
import logging
import streamlit as st
import docx
import tempfile
import json
import re


languages = {
    'English': 'eng',
    'Hindi': 'hi'
}


# Initialize logging with the specified configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(constants.LOGS_FILE),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

# load documents from the specified directory using a DirectoryLoader object

# values = {
#     "openai_api_key": "4b81012d55fb416c9e398f6149c3071e",
# }

# api_key = '4b81012d55fb416c9e398f6149c3071e',  
# api_version = os.getenv("OPENAI_API_VERSION"),
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

def load_pdf_text(path,name):
    with st.spinner('Processing OCR'):
        st.session_state.filename=[]
        texts, nbPages = images_to_txt(path.getvalue(),languages['English'])
        st.session_state.filename.append(name)
        return texts
    
def correct_bhashni_translations(text, lowercase_dict):
    corrected_text = []
    words = text.split()  # Tokenize the input text into words
    for word in words:
        # Check if the word needs correction
        if word in lowercase_dict:
            corrected_text.append(str(lowercase_dict[word]))
        else:
            corrected_text.append(str(word))  # If no correction needed, keep the word unchanged
    return ' '.join(corrected_text)


def get_text(filename):
    doc = docx.Document(filename)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def load_documents(files=[]):
    with st.spinner('Processing'):
        if len(files) == 0:
            loader = DirectoryLoader(constants.FILE_DIR)
            documents = loader.load()
        st.session_state.filename=[]
        documents = []
        for f in files:
            print(f)
            temp_dir = tempfile.TemporaryDirectory()
            temp_filepath = os.path.join(temp_dir.name, f.name)
            with open(temp_filepath, "wb") as fout:
                fout.write(f.read())
            fname = f.name
            st.session_state.filename.append(fname)
            print('file name is')
            print(fname)
            if fname.endswith('.pdf'):
                loader = PyPDFLoader(temp_filepath)
                documents.extend(loader.load())
            elif fname.endswith('.docx'):
                loader = Docx2txtLoader(temp_filepath)
                documents.extend(loader.load())
            elif fname.endswith('.txt'):
                loader = TextLoader(temp_filepath)
                documents.extend(loader.load())
            elif fname.endswith('.md'):
                loader = UnstructuredMarkdownLoader(temp_filepath)
                documents.extend(loader.load())
            elif fname.endswith('.ppt'):
                loader = UnstructuredPowerPointLoader(temp_filepath)
                documents.extend(loader.load())

    # print(documents[0].page_content)
        
        text = " ".join([re.sub('\s+', ' `  ', d.page_content) for d in documents])
        return text, documents

def create_doc_embeddings(documents) -> any:
    
    # split the text to chunks of of size 1000
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # create a vector store from the chunks using an OpenAIEmbeddings object and a Chroma object
    #embeddings = AzureOpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    docsearch = Chroma.from_documents(texts, embeddings)
    return docsearch


def get_cache_vectorstore(text_file,file_name):
    
    default_directory = "chroma_custom_store"
    os.makedirs(default_directory,exist_ok=True)
    ''' if this directory has files
        then use those db directly to created vectorstore and return
        else create new db '''
   
    chroma_directory = os.path.join(default_directory,file_name)
    if not hasattr(st.session_state,"embeddings"):
        st.session_state.embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    if not os.path.exists(chroma_directory):
        # text_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text = text_splitter.split_text(text_file)
        # text_chunks.append(text)

        print("Initializing embedding model...")
        print("No vectorstore for {} file found creating new vectorstore...".format(file_name))
        os.makedirs(chroma_directory,exist_ok=True)
       
        vectorstore = Chroma.from_texts(texts=text,embedding=st.session_state.embeddings,persist_directory=chroma_directory)
        return vectorstore
   
    else:
        print("Existing vectorstore for {} file found...".format(file_name))
        vectorstore = Chroma(persist_directory=chroma_directory, embedding_function=st.session_state.embeddings)
        return vectorstore
 

def load_chain(docsearch):
    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    # Create a prompt template using a template from the config module and input variables
    # representing the context and question.
    # prompt_template = PromptTemplate(template=constants.prompt_template, input_variables=["context", "question"])
    # Load a QA chain using an OpenAI object, a chain type, and a prompt template.
    
    doc_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            openai_api_key = "sk-proj-NduY6NQbG0f2uR6n22F9T3BlbkFJqNztRCFEE5TPFh4vB6Ym",
            model = "gpt-3.5-turbo",
            temperature=0.7
        ),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,    
    )
    return doc_chain

# Define answer generation function
def answer(user_input: str, persist_directory: str = constants.PERSIST_DIR) -> str:
    
    # Log a message indicating that the function has started
    LOGGER.info(f"Start answering based on prompt: {user_input}.")

    
    # Log a message indicating the number of chunks to be considered when answering the user's query.
    LOGGER.info(f"The top {constants.k} chunks are considered to answer the user's query.")
    
    # Create a VectorDBQA object using a vector store, a QA chain, and a number of chunks to consider.
    #qa = VectorDBQA(vectorstore=docsearch, combine_documents_chain=doc_chain, k=constants.k)

    # Call the VectorDBQA object to generate an answer to the prompt.
    #print("prompt going = ",filter_prompt2.format(st.session_state.filename,user_input))
    print(master_prompt.format(
                               st.session_state.mode_of_questions,
                               st.session_state.complexity,
                               st.session_state.no_of_questions,
                               st.session_state.type_of_questions,
                               st.session_state.mode_of_questions,
                               st.session_state.filename,))
    
    
    result = st.session_state.llm({"question": master_prompt.format(
                                                                    st.session_state.mode_of_questions,
                                                                    # st.session_state.filename,
                                                                    st.session_state.complexity,
                                                                     
                                                                    st.session_state.no_of_questions,
                                                                    st.session_state.type_of_questions,
                                                                    st.session_state.mode_of_questions,
                                                                    st.session_state.filename,
                                                                    )})

    answer = result["answer"]
    print("Answer  - ",answer)
    with open('dictionary.json','r') as f:
                existing_dictionary = json.load(f)
    lowercase_dict = {key.lower(): value for key, value in existing_dictionary.items()}                 
    answer_2=correct_bhashni_translations(answer,lowercase_dict)
    translated_response = ""
    if st.session_state.language == 'English' or st.session_state.language =='English and Hindi':
        print("Prompt went of translation ---------------------------",lang_prompt.format(st.session_state.language,answer_2))
        translated_response = st.session_state.language_chain.predict(input=lang_prompt.format(st.session_state.language,answer_2))
        # translated_response = translated_response["response"]
    # Log a message indicating the answer that was generated
    LOGGER.info(f"The returned answer is: {translated_response}")
    # Log a message indicating that the function has finished and return the answer.
    LOGGER.info(f"Answering process completed.")
    return answer,translated_response
def answerq(user_input: str,context:str, persist_directory: str = constants.PERSIST_DIR) -> str:
    # Log a message indicating that the function has started
    LOGGER.info(f"Start answering based on prompt: {user_input}.")
    # Log a message indicating the number of chunks to be considered when answering the user's query.
    LOGGER.info(f"The top {constants.k} chunks are considered to answer the user's query.")
    # Create a VectorDBQA object using a vector store, a QA chain, and a number of chunks to consider.
    #qa = VectorDBQA(vectorstore=docsearch, combine_documents_chain=doc_chain, k=constants.k)
    # Call the VectorDBQA object to generate an answer to the prompt.
    #print("prompt going = ",filter_prompt2.format(st.session_state.filename,user_input))
    print(student_prompt.format(user_input,context))
    result = st.session_state.llm({"question": student_prompt.format(user_input,context)})
    answer = result["answer"]
    return answer
def format_response(result: str, persist_directory: str = constants.PERSIST_DIR) -> str:
     print("prompt going = ",format_prompt.format(result))
     formatted_response = st.session_state.format_chain.predict(input=format_prompt.format(result))
     return formatted_response
def aiformat_response(result: str, persist_directory: str = constants.PERSIST_DIR) -> str:
     ##print("prompt going = ",format_prompt.format(result))
     formatted_response = st.session_state.aiformat_chain.predict(input=aiformat_prompt.format(result))
     return formatted_response

def ai_response(text: str, persist_directory: str = constants.PERSIST_DIR) -> str:
     #print("prompt going = ",ai_prompt.format(text))
     formatted_response = st.session_state.ai_chain.predict(input=ai_prompt.format(text))
     return formatted_response

def mcq_response(result: str, persist_directory: str = constants.PERSIST_DIR) -> str:
     #print("prompt going = ",mcq_test_prompt.format(result))
     formatted_response = st.session_state.mcq_chain.predict(input=mcq_test_prompt.format(result))
     st.write(formatted_response)
     return formatted_response

def key_term(result: str, persist_directory: str = constants.PERSIST_DIR) -> str:
     #print("prompt going = ",mcq_test_prompt.format(result))
     formatted_response = st.session_state.key_term_chain.predict(input=key_term_prompt.format(result))
     st.write(formatted_response)
     return formatted_response

def learn_outcome_term(result: str, persist_directory: str = constants.PERSIST_DIR) -> str:
     #print("prompt going = ",mcq_test_prompt.format(result))
     formatted_response = st.session_state.learn_outcome_chain.predict(input=learn_outcome_prompt.format(result))
     st.write(formatted_response)
     return formatted_response



