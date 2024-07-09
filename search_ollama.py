import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import ollama

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')

# Streamlit UI
st.title("DIY SEMANTIC CODE SEARCH")

# Input fields for query, number of top files to retrieve, and index name
query = st.text_input("Enter your query:")
depth = st.number_input("Enter the number of top files to retrieve:", min_value=1, value=4)
name = st.text_input("Enter the index name:")

def load_index_map(name):
    """
    Load the index to file path mapping from a text file.
    
    Args:
        name (str): The name of the index file.
    
    Returns:
        dict: A dictionary mapping index numbers to file paths.
    """
    index_map = {}
    with open('index_map_' + name + '.txt', 'r') as f:
        for line in f:
            idx, path = line.strip().split('\t')
            index_map[int(idx)] = path
    return index_map

def query_top_files(query, top_n=4):
    """
    Retrieve the top_n most similar files to the query using the Annoy index.
    
    Args:
        query (str): The search query.
        top_n (int): The number of top files to retrieve.
    
    Returns:
        list: A list of tuples containing file paths and their distances from the query.
    """
    # Load the Annoy index and the index map
    t = AnnoyIndex(768, 'angular')
    t.load(name + '_mpnet.ann')
    index_map = load_index_map(name)
    
    # Get embeddings for the query
    query_embedding = model.encode(query)
    
    # Search in the Annoy index for the top_n most similar files
    indices, distances = t.get_nns_by_vector(query_embedding, top_n, include_distances=True)
    
    # Fetch file paths for these indices
    files = [(index_map[idx], dist) for idx, dist in zip(indices, distances)]
    return files

def get_file_contents(path):
    """
    Read and return the contents of a file.
    
    Args:
        path (str): The file path.
    
    Returns:
        str: The contents of the file.
    """
    try:
        with open(path, 'r') as f:
            return f.read()
    except:
        return ""

def get_LLM_response(path, query):
    """
    Get a response from the LLM based on the file contents and the query.
    
    Args:
        path (str): The file path.
        query (str): The search query.
    
    Returns:
        str: The response from the LLM.
    """
    content = get_file_contents(path)
    if not content.strip():
        return "Empty file. Please choose another file."
    
    # Format the content and create the message for the LLM
    file_content = f"Path: {path}\n{content}"

    prompt_message = f"""You have been asked to perform a specific task related to a codebase. 
    
    Tasks: 
    1) Identify the Purpose of the File: {path} based on Contents of the File below
    - Carefully read through the contents of the provided file. 
    - Summarize what the file is about. Explain the primary functionality or purpose of the code in this file. 
    2) Determine Necessary Changes: 
        - Based on the user's question and the current contents of the file, identify what changes need to be made to accomplish the task specified by the user.
        - Provide a step-by-step outline of the modifications required, including specific code snippets if applicable. 
    Step by Step instruction
    Understanding the File: 
        - Start by examining the file’s structure and content. 
        - Identify key functions, classes, and comments that indicate the purpose and functionality of the file. 
        - Summarize the primary purpose and functionality of the file in a clear and concise manner. 
    Analyzing the User's Question: 
        - Understand the user’s question and the specific task they want to accomplish.
        - Determine how the requested changes align with the current functionality of the file.
    Identifying Changes: 
        - Identify the specific sections of the file that need modification to accomplish the task. 
        - Outline the steps needed to implement these changes, including any new code that needs to be added or existing code that needs to be modified.
        - Provide detailed code snippets for each step, ensuring clarity and correctness.

    Context: 
    User's Question: {query} 
    Top Matching File: {path} 
    Contents of the File: {content} 
        """
    

    response = ""

    ###print(prompt_message)
    
    # Stream the response from the LLM
    #stream = ollama.chat(model='llama3:latest', messages=[{'role': 'user', 'content': prompt_message}], stream=True)
    stream = ollama.chat(model='codestral:latest', messages=[{'role': 'user', 'content': prompt_message}], stream=True)

    for chunk in stream:
        response += chunk['message']['content']
    
    return response


def get_llm_summary(path,query):
    content = get_file_contents(path)
    prompt = f"""
            Path: {path}
            Code: 
            {content}
            You response should accomplish following 2 tasks after carefully reading throught the contents of the code pasted above.
            Tasks:
            1. Summarize the code so that a new engineer can understand in 3 lines
            2. Explain what needs to be changed to accomplish the user's task of \" {query}\" in 2 lines
            """
    
    ###print(prompt)
    
    # Stream the response from the LLM
    response = ""

    stream = ollama.chat(model='yi:latest', messages=[{'role': 'user', 'content': prompt}], stream=True)
    for chunk in stream:
        response += chunk['message']['content']
    
    return response
    
# Main logic to handle user input and display results
if query and name:
    results = query_top_files(query, depth)
    
    # Display the results as search engine links
    st.write("Files you might want to read:")
    counter = 1
    for path,dist in results:
        st.markdown(f"###### File {counter}")
        st.markdown(f" <font color=red> {path}  </font> <font color= green> (Cosine Distance: {dist}) </font>", unsafe_allow_html = True)
        counter+=1

    st.markdown("--------------------------------")

    st.markdown("<font color=red> LLMS are known to hallucinate </font><br><br> read the summary with a grain of salt", unsafe_allow_html = True)
    st.markdown("--------------------------------")

    st.markdown("#### Summary of each file:")

    counter = 1
    for path, dist in results:
        st.markdown(f"###### File {counter}")
        st.markdown(f" <font color=red> {path}  </font> <font color= green> (Cosine Distance: {dist}) </font>", unsafe_allow_html = True)
        summary = get_LLM_response(path, query)
        st.markdown(summary)
        st.markdown("--------------------------------")
        counter+=1


 