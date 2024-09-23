


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
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sns


# Loading the dataset
data = pd.read_csv('C:/Users/aayus/OneDrive/Desktop/customer-segmentation2-main/New folder (5)/online_retail_store.csv')

# the first few rows of your data
print(data.head())


# Define prompt to query data
prompt = PromptTemplate(
    input_variables=["segment"],
    template="Give insights about the customer segment {segment}."
)

# Load OpenAI LLM (using your API key)
llm = OpenAI(api_key= "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")

# Create a LangChain with the prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Query example
segment_insight = chain.run(segment="high spenders")
print(segment_insight)
while True:
    user_input = input("Ask about a customer segment: ")
    if user_input.lower() == "exit":
        break
    response = chain.run(segment=user_input)
    print(response)
