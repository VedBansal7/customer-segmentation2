


# Imports
import os 
import pandas as pd
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message
from langchain_community.llms import OpenAI
from langchain_experimental import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import creds


# Loading the dataset
data = pd.read_csv('C:/Users/aayus/OneDrive/Desktop/customer-segmentation2-main/New folder (5)/online_retail_store.csv')

# the first few rows of your data
print(data.head())

api_key = (creds.api_key)
model_id = 'gpt-3.5-turbo'


# Define prompt to query data
prompt = PromptTemplate(
    input_variables=[high spenders],
    template="Give insights about the customer segment {high spenders}."
)
# Create a LangChain with the prompt
chain = LLMChain(llm=llm, prompt=prompt)
llm = ChatOpenAI(model_name = model_id, temperature=0)
df = pd.read_csv('online_retail_store.csv')
agent = create_pandas_dataframe_agent(llm, df, verbose=True, max_iterations=6)

# Query example
segment_insight = chain.run(segment="high spenders")
print(segment_insight)
while True:
    user_input = input("Ask about a customer segment: ")
    if user_input.lower() == "exit":
        break
    response = chain.run(segment=user_input)
    print(response)
llm = ChatOpenAI(model_name = model_id, temperature=0)
df = pd.read_csv('online_.csv')
agent = create_pandas_dataframe_agent(llm, df, verbose=True, max_iterations=6)

# Setup streamlit app
# Display the page title and the text box for the user to ask the question
st.title('U+2728 Query your Data ')
prompt = st.text_input("Enter your question to query your PDF documents")