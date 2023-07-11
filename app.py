import streamlit as st
import pandas as pd
import numpy as np 

st.title("Let's explores session states and callback functions")

# Create a 4x4 numpy array with random numbers
data = np.random.rand(4, 4)

# Create a pandas DataFrame
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])

increment = st.button("show more columns")
if increment:
        random_numbers = np.random.rand(100)
        if df:
            time.sleep(5)
            # st.write(data[0][:50])
            random_numbers = np.random.rand(100)
            plt.plot(random_numbers)
            st.pyplot(plt)

# import os
# import streamlit as st 
# import PyPDF2
# from io import BytesIO
# import openai
# import time
# import matplotlib.pyplot as plt
# import numpy as np

# openai.api_key = st.secrets["openai_password"]

# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] =  "gpt-3.5-turbo-16k" 

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if prompt := st.chat_input("What is up?"):
#     # If the user inputs a message, clear previous messages and append the new one with the role "user"
#     st.session_state.messages = [{"role": "user", "content": prompt}]
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
#         for response in openai.ChatCompletion.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         ):
#             full_response += response.choices[0].delta.get("content", "")
#             message_placeholder.markdown(full_response + "▌")
#         message_placeholder.markdown(full_response)
#     st.session_state.messages.append({"role": "assistant", "content": full_response})

# uploaded_files = st.sidebar.file_uploader("",accept_multiple_files=True, type=['pdf'])
# data = []
# filenames = []
# if uploaded_files:
#     st.sidebar.write("You have uploaded the following files:")
#     for file in uploaded_files:
#         st.sidebar.write(file.name)
#         file_stream = BytesIO(file.read())
#         pdf_reader = PyPDF2.PdfFileReader(file_stream)
#         text = ""
#         for page in range(pdf_reader.getNumPages()):
#             text += pdf_reader.getPage(page).extract_text()
#         data.append(text)
#         filenames.append(file.name)
        
# random_numbers = np.random.rand(100)
# if data:
#     time.sleep(5)
#     st.write(data[0][:50])
#     random_numbers = np.random.rand(100)
#     plt.plot(random_numbers)
#     st.pyplot(plt)



















# import os
# import streamlit as st # Bring in streamlit for UI/app interface
# import PyPDF2
# from io import BytesIO
# import openai
# import time
# import matplotlib.pyplot as plt
# import numpy as np

# openai.api_key = st.secrets["openai_password"]

# ###############################################
# ##### I THINK THIS IS THE CHAT BAR #¢##########
# # Initialize the session state for the OpenAI model if it doesn't exist, with a default value of "gpt-4"
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] =  "gpt-3.5-turbo-16k" # "gpt-4"

# # Initialize the session state for the messages if it doesn't exist, as an empty list
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display all the existing messages in the chat, with the appropriate role (user or assistant)
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Wait for the user to input a message
# if prompt := st.chat_input("What is up?"):
#     # If the user inputs a message, append it to the session's messages with the role "user"
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display the user's message
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Prepare for the assistant's message
#     with st.chat_message("assistant"):
#         # Create a placeholder for the assistant's message
#         message_placeholder = st.empty()
#         # Initialize an empty string to build up the assistant's response
#         full_response = ""
#         # Generate the assistant's response using OpenAI's chat model, with the current session's messages as context
#         # The response is streamed, which means it arrives in parts that are appended to the full_response string
#         for response in openai.ChatCompletion.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         ):
#             # Append the content of the new part of the response to the full_response string
#             full_response += response.choices[0].delta.get("content", "")
#             # Update the assistant's message placeholder with the current full_response string, appending a "▌" to indicate it's still typing
#             message_placeholder.markdown(full_response + "▌")
#         # Once the full response has been received, update the assistant's message placeholder without the "▌"
#         message_placeholder.markdown(full_response)
#     # Append the assistant's full response to the session's messages with the role "assistant"
#     st.session_state.messages.append({"role": "assistant", "content": full_response})
# ####################################################
# ####################################################

# uploaded_files = st.sidebar.file_uploader("",accept_multiple_files=True, type=['pdf'])
# # Initialize an empty list to store the extracted text from the uploaded files
# data = []
# filenames = []
# if uploaded_files:
#     st.sidebar.write("You have uploaded the following files:")
#     for file in uploaded_files:
#         st.sidebar.write(file.name)
#         # Open the file as a stream
#         file_stream = BytesIO(file.read())
#         # Create a PDF file reader object
#         pdf_reader = PyPDF2.PdfFileReader(file_stream)
#         text = ""
#         # Loop over each page in the PDF and extract the text
#         for page in range(pdf_reader.getNumPages()):
#             text += pdf_reader.getPage(page).extract_text()
#         # Append the text to the data list
#         data.append(text)
#         # Append the filename to the filenames list
#         filenames.append(file.name)
        
# # Generate an array of random numbers
# random_numbers = np.random.rand(100)
# if data:
#     time.sleep(5)
#     st.write(data[0][:50])
#     # Generate an array of random numbers
#     random_numbers = np.random.rand(100)
#     plt.plot(random_numbers)
#     st.pyplot(plt)












# # # # # Split pages from pdf 
# # # pages = loader.load_and_split()
# # document = loader.load() # load sinlge page
# # # document
# # # Load documents into vector database aka ChromaDB
# # store = Chroma.from_documents(document, collection_name='breast_cancer')
# # # store
# # # Create vectorstore info object - metadata repo?
# # vectorstore_info = VectorStoreInfo(
# #     name="breast_cancer",
# #     description="breast cancer information",
# #     vectorstore=store
# # )
# # # Convert the document store into a langchain toolkit
# # toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
# # # Add the toolkit to an end-to-end LC
# # agent_executor = create_vectorstore_agent(
# #     llm=llm,
# #     toolkit=toolkit,
# #     verbose=True
# # )
# # st.title('Your journey begins here') # headline of the website
# # # Create a text input box for the user
# # prompt = st.text_input('respond like an oncologist that specializes in breast cancer - with 30 years experience and IQ 120 and very high emotional intelligence')

# # # If the user hits enter
# # if prompt:
# #     # Then pass the prompt to the LLM
# #     response = agent_executor.run(prompt)
# #     # ...and write it out to the screen
# #     st.write(response)

# #     # With a streamlit expander  
# #     with st.expander('Document Similarity Search'):
# #         # Find the relevant pages
# #         search = store.similarity_search_with_score(prompt) 
# #         # Write out the first 
# #         st.write(search[0][0].page_content) 
