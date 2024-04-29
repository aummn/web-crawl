################################################################################
### Step 1
################################################################################
from openai import OpenAI
import requests
import re
import urllib.request
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
# from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

client = OpenAI()
GPT_MODEL = 'gpt-4-turbo'
# GPT_MODEL = "gpt-3.5-turbo-1106"
EMBEDDING_MODEL = 'text-embedding-3-large'
MAX_INPUT_TOKENS = 4096  # GPT's maximum input token limit
MAX_OUTPUT_TOKENS = 150  # GPT's maximum output token limit

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Define root domain to crawl
domain = "apromore.com"
full_url = "https://apromore.com/"








################################################################################
### Step 11
################################################################################

def getEmbeddingsDataFrame():
    # df = pd.read_csv('../../processed/embeddings.csv', index_col=0)
    df = pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
    return df


# df=pd.read_csv('processed/embeddings.csv', index_col=0)
# df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
df = getEmbeddingsDataFrame()
df.head()

################################################################################
### Step 12
################################################################################

# iterates over each embedding in the embeddings list and calculates the distance to the query_embedding using the specified distance_metric.
# The distances are then returned as a list. The supported distance metrics are "cosine" and "euclidean".
def distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine"):
    distances = []
    for embedding in embeddings:
        if distance_metric == "cosine":
            distances.append(distance.cosine(query_embedding, embedding))
        elif distance_metric == "euclidean":
            distances.append(distance.euclidean(query_embedding, embedding))
        else:
            raise ValueError("Unsupported distance metric: " + distance_metric)
    return distances


def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = client.embeddings.create(input=question, model=EMBEDDING_MODEL).data[0].embedding

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def ensure_fit(question, context):
    total_length = len(question.split()) + len(context.split())
    if total_length > MAX_INPUT_TOKENS:
        # If the total length is too long, truncate the context
        context = context.split()[:MAX_INPUT_TOKENS - len(question.split())]
        context = ' '.join(context)
    return question, context

def answer_question(
    df,
    model=GPT_MODEL,
    question="",
    context=None,
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    dataframe_context = create_context(question, df, max_len=max_len, size=size);
    
    if dataframe_context is None:
        dataframe_context = ""

    if context is None:
        context = ""        

    # Combine the contexts if a context was passed in
    context = dataframe_context + " " + context
    question, context = ensure_fit(question, context)

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        
        # messages = [
        #     {
        #         "role": "user",
        #         "content": "The system should generate a unique digest ID for each answer to represent the details of the answer. This digest ID should be prepended to the beginning of the answer so that it can be easily seen. If two answers have the same digest ID, it means they are the same, and a new answer with a different digest ID should be presented."
        #     },
        #     {
        #         "role": "user",
        #         # "content": f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"
        #         "content": f"Answer the question based on the context below, \n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"
        #     }
        # ]

        messages = [
            # {
            #     "role": "system",
            #     "content": {context}
            # },            
            {
                "role": "user",
                "content": f"Answer the question based on the context below, \n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"
            }
        ]    

        # Create a completions using the questin and context
        response = client.chat.completions.create(
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        # return response.choices[0].text.strip()
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""



def answer_questions(question, frontend_context):
    # df = getEmbeddingsDataFrame()
    df.head()
    return answer_question(
        df,
        model=GPT_MODEL,
        question=question,
        context=frontend_context,  # Pass the context into the function
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=MAX_OUTPUT_TOKENS,
        stop_sequence=None
    )

################################################################################
### Step 13
################################################################################

# print(answer_question(df, question="can you tell me top 10 news on Apromore in 2024?"))

# print(answer_question(df, question="what's apromore ?"))

# print(answer_question(df, question="can you list job titles ?"))

# print(answer_question(df, question="please let me know top 5 partners ?"))

# print(answer_question(df, question="what products does Apromore provide ?"))
# print(answer_question(df, question="where'Estonia office ?"))
# print(answer_question(df, question="do you know Apromore office business address ?"))
# print(answer_question(df, question="is Apromore in Carlton (Melbourne), VIC 3053, Australia ?"))

# print(answer_question(df, question="is Clear Analytics a business partner ?"))
# print(answer_question(df, question="can you list top 5 Apromore business partners ?"))
# print(answer_question(df, question="can you list top 5 Apromore business partners in America ?"))
# print(answer_question(df, question="can you list top 5 Apromore business partners in Australia ?"))
# print(answer_question(df, question="who are the founders of Apromore ?"))
# print(answer_question(df, question="is Marlon Dumas the founder of Apromore ?"))
# print(answer_question(df, question="is Simon Raboczi the founder of Apromore ?"))
# print(answer_question(df, question="is Dr. Ilya Verenich the founder of Apromore ?"))

# print(answer_question(df, question="who's marlon ?"))

# print(answer_question(df, question="what are the key features of Apromore product ?"))

# print(answer_question(df, question="Is GPT-4 the newest GPT model?"))

# print(answer_question(df, question="which one is the newest GPT model ?"))

# print(answer_question(df, question="Seems GPT-3 the newest GPT model?"))

# print(answer_question(df, question="What day is it?", debug=False))

# print(answer_question(df, question="What is our newest embeddings model?"))
