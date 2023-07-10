import os
#from langchain.llms import OpenAI # Import OpenAI as main LLM service
import streamlit as st # Bring in streamlit for UI/app interface
# from langchain.document_loaders import PyPDFLoader # Import PDF document loaders...there's other ones as well!
#from langchain.vectorstores import Chroma # Import chroma as the vector store 
import PyPDF2
from io import BytesIO
import openai

# # Import vector store stuff
# from langchain.agents.agent_toolkits import (
#     create_vectorstore_agent,
#     VectorStoreToolkit,
#     VectorStoreInfo
# )

# # Set APIkey for OpenAI Service / Can sub this out for other LLM providers
# os.environ['OPENAI_API_KEY'] = st.secrets["openai_password"]
# Set the OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_password"]

# # Create instance of OpenAI LLM
# llm = OpenAI(temperature=0.1, verbose=True)

# st.write("yo")
# Create and load PDF Loader
# loader = PyPDFLoader("breast-invasive-patient.pdf")
# # loader = PyPDFLoader('/content/drive//My Drive/3_MY_WORK/1_My_projects/3b_langchain_experiment/git_repo/LangchainDocuments/gal.pdf')
uploaded_files = st.sidebar.file_uploader("",accept_multiple_files=True, type=['pdf'])
# Initialize an empty list to store the extracted text from the uploaded files
data = []
filenames = []
if uploaded_files:
    st.sidebar.write("You have uploaded the following files:")
    for file in uploaded_files:
        st.sidebar.write(file.name)
        # Open the file as a stream
        file_stream = BytesIO(file.read())
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfFileReader(file_stream)
        text = ""
        # Loop over each page in the PDF and extract the text
        for page in range(pdf_reader.getNumPages()):
            text += pdf_reader.getPage(page).extract_text()
        # Append the text to the data list
        data.append(text)
        # Append the filename to the filenames list
        filenames.append(file.name)

if data:
    st.write(data[:50])

# # # # Split pages from pdf 
# # pages = loader.load_and_split()
# document = loader.load() # load sinlge page
# # document
# # Load documents into vector database aka ChromaDB
# store = Chroma.from_documents(document, collection_name='breast_cancer')
# # store
# # Create vectorstore info object - metadata repo?
# vectorstore_info = VectorStoreInfo(
#     name="breast_cancer",
#     description="breast cancer information",
#     vectorstore=store
# )
# # Convert the document store into a langchain toolkit
# toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
# # Add the toolkit to an end-to-end LC
# agent_executor = create_vectorstore_agent(
#     llm=llm,
#     toolkit=toolkit,
#     verbose=True
# )
# st.title('Your journey begins here') # headline of the website
# # Create a text input box for the user
# prompt = st.text_input('respond like an oncologist that specializes in breast cancer - with 30 years experience and IQ 120 and very high emotional intelligence')

# # If the user hits enter
# if prompt:
#     # Then pass the prompt to the LLM
#     response = agent_executor.run(prompt)
#     # ...and write it out to the screen
#     st.write(response)

#     # With a streamlit expander  
#     with st.expander('Document Similarity Search'):
#         # Find the relevant pages
#         search = store.similarity_search_with_score(prompt) 
#         # Write out the first 
#         st.write(search[0][0].page_content) 
