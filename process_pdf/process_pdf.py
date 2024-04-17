from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
import os
import getpass
from typing import Optional 
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import tiktoken
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print
from langchain.globals import set_debug
import numpy as np
import umap
import pandas as pd
from sklearn.mixture import GaussianMixture
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import concurrent.futures
from retrying import retry
import random
import time
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
import markdown
from IPython.display import display, HTML
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from typing import List, Dict
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField


app = FastAPI()


loader = PyMuPDFLoader(
    "data/TCEQ_AERRSensitivity_Final_01262024.pdf", 
    extract_images=True
)
docs = loader.load()
print(len(docs))
print(docs)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

def split_text(
    text: str, tokenizer: tiktoken.get_encoding("cl100k_base"), max_tokens: int, overlap: int = 0
):
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    
    Args:
        text (str): The text to be split.
        tokenizer (CustomTokenizer): The tokenizer to be used for splitting the text.
        max_tokens (int): The maximum allowed tokens.
        overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 0.
    
    Returns:
        List[str]: A list of text chunks.
    """
    # Split the text into sentences using multiple delimiters
    # delimiters = [".", "!", "?", "\n"]
    delimiters = ["...","\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)
    
    # Calculate the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if not sentence.strip():
            continue
        
        # If the sentence is too long, split it into smaller parts
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in sub_sentences]
            
            sub_chunk = []
            sub_length = 0
            
            for sub_sentence, sub_token_count in zip(sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                    sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])
                
                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count
            
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        
        # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count
        
        # Otherwise, add the sentence to the current chunk
        else:
            current_chunk.append(sentence)
            current_length += token_count
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

texts = [doc.page_content for doc in docs]
new_docs = '\n'.join(texts)

texts = split_text(new_docs, tiktoken.get_encoding("cl100k_base"), 100)
print(texts)


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

counts = [num_tokens_from_string(t) for t in texts]
print(len(counts))



plt.figure(figsize=(10, 6))
plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
plt.title("Histogram of Token Counts")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)
plt.show

d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)
print(
    "Num tokens in all context: %s"
    % num_tokens_from_string(concatenated_content)
)

os.environ["OPENAI_API_KEY"] = getpass.getpass('Enter your OpenAI API key: ')
model = ChatOpenAI(model="gpt-3.5-turbo-0125")

embedding_model = OpenAIEmbeddings()

class Embedding:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        
    def get_embeddings(self, texts):
        with ThreadPoolExecutor() as executor:
            # Submit all embed_query tasks to the executor
            future_to_txt = {executor.submit(self.embedding_model.embed_query, txt): txt for txt in texts}
            
            global_embeddings = []
            for future in as_completed(future_to_txt):
                data = future.result()  # Retrieve the result from the future
                global_embeddings.append(data)
                
        return global_embeddings
    
embedder = Embedding(embedding_model)
global_embeddings = embedder.get_embeddings(texts)

global_embeddings[0]

set_debug(True)

print(len(global_embeddings))

class EmbeddingReducer:
    def __init__(self, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine"):
        """
        Initialize the EmbeddingReducer with UMAP configuration.
        
        Args:
            dim (int): Target dimensionality of the reduced space.
            n_neighbors (Optional[int]): The number of neighbors to consider for each point. Defaults to the square root of the number of embeddings minus one.
            metric (str): The metric to use for computing distances in high dimensional space. Defaults to "cosine".
        """
        self.dim = dim
        self.n_neighbors = n_neighbors  # This will be dynamically set in `reduce_embeddings` if None.
        self.metric = metric
        self.reducer = None

    def reduce_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce the dimensionality of embeddings using UMAP.
        
        Args:
            embeddings (np.ndarray): The high-dimensional data points to reduce.
        
        Returns:
            np.ndarray: The reduced dimensionality embeddings.
        """
        # Dynamically set `n_neighbors` based on the data if not explicitly provided
        if self.n_neighbors is None:
            self.n_neighbors = int((len(embeddings) - 1) ** 0.5)

        # Initialize the UMAP reducer if it hasn't been initialized already
        if self.reducer is None:
            self.reducer = umap.UMAP(
                n_neighbors=self.n_neighbors, 
                n_components=self.dim, 
                metric=self.metric
            )
        
        # Perform the dimensionality reduction
        return self.reducer.fit_transform(embeddings)
    
dim = 10

# Initialize the EmbeddingReducer with the desired dimensionality
embedding_reducer = EmbeddingReducer(dim)

# Reduce the dimensionality of global embeddings
global_embeddings_reduced = embedding_reducer.reduce_embeddings(global_embeddings)

# Access the first reduced embedding as an example
global_embeddings_reduced[0]

def plot_embeddings(embeddings: np.ndarray, title: str = "Global Embeddings",
                    xlabel: str = "Dimension 1", ylabel: str = "Dimension 2",
                    figsize: tuple = (10, 8), alpha: float = 0.5):
    plt.figure(figsize=figsize)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


plot_embeddings(global_embeddings_reduced, title="Global Embeddings Reduced to 2D",
                xlabel="Dimension 1", ylabel="Dimension 2", figsize=(10, 8), alpha=0.5)

def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 1234):
    max_clusters = min(max_clusters, len(embeddings))
    bics = [GaussianMixture(n_components=n, random_state=random_state).fit(embeddings).bic(embeddings)
            for n in range(1, max_clusters)]
    return np.argmin(bics) + 1

def gmm_clustering(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

labels, _ = gmm_clustering(global_embeddings_reduced, threshold=0.5)

plot_labels = np.array([label[0] if len(label) > 0 else -1 for label in labels])
plt.figure(figsize=(10, 8))

unique_labels = np.unique(plot_labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    mask = plot_labels == label
    plt.scatter(global_embeddings_reduced[mask, 0], global_embeddings_reduced[mask, 1], color=color, label=f'Cluster {label}', alpha=0.5)

plt.title("Cluster Visualization of Global Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()

simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]

df = pd.DataFrame({
    'Text': texts,
    'Embedding': list(global_embeddings_reduced),
    'Cluster': simple_labels
})
df.head(3)

def format_cluster_texts(df):
    clustered_texts = {}
    for cluster in df['Cluster'].unique():
        cluster_texts = df[df['Cluster'] == cluster]['Text'].tolist()
        clustered_texts[cluster] = " ----- ".join(cluster_texts)
    return clustered_texts

clustered_texts = format_cluster_texts(df)

for key, value in clustered_texts.items():
    print("===========================================")
    print(f"Size of the Cluster is : {num_tokens_from_string(value)}")
    print(f"""Cluster name: {key}, cluster data : {value}""" )
    print("===========================================")


template = """
You are a helpful assistant.
Write a summary of the following, including as many key details as possible: {text}:
also add important keywords from the input that you think would be neccessary for RAG operation as a list at the end of the summary
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model | StrOutputParser()

def retry_if_rate_limit_exceeded(exception):
    """Return True if we should retry (in this case when it's a rate limit exceeded error), False otherwise."""
    return "rate_limit_exceeded" in str(exception)

# Apply the retry decorator
@retry(retry_on_exception=retry_if_rate_limit_exceeded, stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def summarize_text(text):
    """Invoke chain function that may fail."""
    return chain.invoke({"text": text})

summaries = {}
all_info = {}

# Use ThreadPoolExecutor with a fixed number of workers
# You can choose any number between 5 and 10 based on your specific requirements and system capabilities
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_cluster_text = {}
    for cluster, text in clustered_texts.items():
        future = executor.submit(summarize_text, text)
        future_to_cluster_text[future] = (cluster, text)
    
    for future in concurrent.futures.as_completed(future_to_cluster_text):
        cluster, text = future_to_cluster_text[future]
        try:
            summary = future.result()
        except Exception as exc:
            print(f'Generated an exception: {exc}')
            summary = None  # Optionally handle the failure
        summaries[cluster] = summary
        all_info[cluster] = {'text': text, 'summary': summary}

for key, value in all_info[3].items():
    print(f"{key}:")
    print(value)
    print("\n") 

indices_list = list(all_info.keys())  # Collect all keys (indices) from the dictionary into a list

# Now sort the list of indices
indices_list.sort()

# Print the sorted list of indices
print("Sorted indices:")
print(indices_list)

summaries

summary_values = [summary for summary in summaries.values()]
embedded_summaries = embedder.get_embeddings(summary_values)
embedded_summaries_np = np.array(embedded_summaries)

labels, _ = gmm_clustering(embedded_summaries_np, threshold=0.5)

simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]

plt.figure(figsize=(10, 8))

unique_labels = np.unique(simple_labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    mask = plot_labels == label
    plt.scatter(global_embeddings_reduced[mask, 0], global_embeddings_reduced[mask, 1], color=color, label=f'Cluster {label}', alpha=0.5)

plt.title("Cluster Visualization of Clustered Summaries Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()

clustered_summaries = {}
for i, label in enumerate(simple_labels):
    if label not in clustered_summaries:
        clustered_summaries[label] = []
    clustered_summaries[label].append(list(summaries.values())[i])

    clustered_summaries

final_summaries = {}
for cluster, texts in clustered_summaries.items():
    combined_text = ' '.join(texts)
    summary = chain.invoke({"text": combined_text})
    final_summaries[cluster] = summary

final_summaries

texts_from_df = df['Text'].tolist()
texts_from_clustered_texts = list(clustered_texts.values())
texts_from_final_summaries = list(final_summaries.values())

combined_texts = texts_from_df + texts_from_clustered_texts + texts_from_final_summaries

# Now, use all_texts to build the vectorstore with Chroma
vectorstore = Chroma.from_texts(texts=combined_texts, embedding=embedding_model, persist_directory='./rag7')

final_number = 30  

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": final_number})

def input_received(input, max_token=15000):
    input_list = []
    token_size = 0
    for doc in input:
        print("================================")
        print(f"Token size before adding new document: {token_size}")
        # print(doc)
        current_doc_tokens = num_tokens_from_string(doc.page_content)
        print(f"Token size after adding document: {token_size}")
        if (token_size + current_doc_tokens) < max_token:
            input_list.append(doc)
            token_size += current_doc_tokens
        else:
            print("Max token limit reached with this document. Stopping further addition.")
            break
        print("================================")

    return input_list

template = """
You are Question Answering Portal
Given Context: {context} Give the best full answer amongst the option to question {question}
if information give can be put into points then display as bullet points
if possible provide output in 3500 words
"""
prompt = ChatPromptTemplate.from_template(template)


rag_chain = (
    {"context": retriever | input_received, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain.invoke("provide detailed summary of the report?")

html_output = markdown.markdown(result)

# Display the HTML in Jupyter Notebook
display(HTML(html_output))

document = {
    "Introduction": ["Background information on the topic", "Problem statement/research questions", "Significance of the study", "Objectives/purpose of the research", "Overview of the paper's structure"],
    "Literature Review": ["Review of relevant literature", "Summary of existing research findings", "Identification of gaps in the literature" , "Theoretical framework (if applicable)"],
    "Methodology": ["Description of research design", "Explanation of data collection methods", "Sampling techniques (if applicable)", "Data analysis procedures"],
    "Results": ["Presentation of research findings", "Use of tables, graphs, or figures to illustrate data" , "Objective reporting of results without interpretation"]
}


for section, subsection in document.items():
    print(f"## {section}:-")
    for sub in subsection:
        print(f"- {sub}")


print(summaries)

token_list = [num_tokens_from_string(text) for text in summaries.values()]
print(token_list)


WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501


RESEARCH_REPORT_TEMPLATE = """Information: 
--------
{research_summary}
--------
`Section: {section}`
Based on the above information, focus specifically on the section and sub-section titled "{sub_section}". \
Extract and elaborate on this part in detail. \
The response should be structured, informative, and contain in-depth analysis with facts and numbers where available. \
Aim for a minimum of 3,500 words.

- Ensure the response is strictly confined to the requested sub-section.
- Use markdown syntax for the response.
- Do not insert personal opinions or stray from the data presented in the research summary.
- Include references in APA format from the data provided without duplicating sources.
- This response is crucial for a detailed understanding of "{sub_section}" within the context of "{topic}". 

Please concentrate on providing a thorough and precise analysis of the requested sub-section.
Format output in below format
# section
    ## sub_section
    ## sub_section
    ## sub_section
    ## sub_section
"""



# model = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)
chain = prompt | model | StrOutputParser()

topic = "DETERMINING THE IMPACT OF ZERO EMISSIONS VEHICLES ON TRAFFIC-RELATED AIR POLLUTION EXPOSURE IN DISADVANTAGED COMMUNITIES"
section_details =""
gen_out = []
for section, subsection in document.items():
    
    # gen_out.append(f"##{section}")
    subsections_formatted = "\t\n-" + ", \t\n-".join(subsection)
    print(f"## {section} - {subsections_formatted}")
    out = chain.invoke({
        "research_summary" : summaries,
        "section" : section,
        "sub_section" : subsections_formatted,
        "topic" : topic
    })
    print("====================================================================")
    print("====================================================================")
    print("====================================================================")
    print("====================================================================")
    print(out)
    print("====================================================================")
    print("====================================================================")
    print("====================================================================")
    print("====================================================================")
    gen_out.append(out)

for out in gen_out:
    html_out = markdown.markdown(out)
    display(HTML(html_out)) 

@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = Form(...), document_id: str = Form(...)):
    # Save the uploaded PDF file
    with open(f"{document_id}.pdf", "wb") as buffer:
        buffer.write(await file.read())
    
    # Load the PDF file
    loader = PyMuPDFLoader(f"{document_id}.pdf", extract_images=True)
    docs = loader.load()
   
    return {"status": "success", "message": "PDF processed successfully."}

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  