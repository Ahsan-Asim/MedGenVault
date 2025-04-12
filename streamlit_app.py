
# import streamlit as st
# import faiss
# import pickle
# import numpy as np
# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from sentence_transformers import SentenceTransformer

# # Load LLM model (local folder)
# @st.cache_resource
# def load_llm():
#     model_path = "./Generator_Model"
#     tokenizer = T5Tokenizer.from_pretrained(model_path)
#     model = T5ForConditionalGeneration.from_pretrained(model_path)
#     return tokenizer, model

# # Load embedding model (local folder)
# @st.cache_resource
# def load_embedding_model():
#     embed_model_path = "./Embedding_Model1"
#     return SentenceTransformer(embed_model_path)

# # Load FAISS index and embeddings
# @st.cache_resource
# def load_faiss():
#     # Load FAISS index
#     faiss_index = faiss.read_index("faiss_index_file.index")
    
#     # Load the texts (raw data)
#     with open("texts.pkl", "rb") as f:
#         data = pickle.load(f)
        
#     # Load the embeddings
#     embeddings = np.load("embeddings_file.npy", allow_pickle=True)
    
#     return faiss_index, data, embeddings

# # Search function to find top-k contexts based on query
# def search(query, embed_model, index, data, k=5):
#     # Generate query embedding
#     query_embedding = embed_model.encode([query]).astype('float32')
    
#     # Perform FAISS search
#     _, I = index.search(query_embedding, k)  # Top-k results
#     results = [data[i] for i in I[0] if i != -1]
#     return results

# # Generate response using the LLM model (T5 model)
# def generate_response(context, query, tokenizer, model):
#     input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
#     inputs = tokenizer.encode(input_text, return_tensors="pt")
#     outputs = model.generate(inputs, max_length=512, do_sample=True, temperature=0.7)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # Streamlit App
# def main():
#     st.title("Local LLM + FAISS + Embedding Search App")
#     st.markdown("🔍 Ask a question, and get context-aware answers!")

#     # Load everything once
#     tokenizer, llm_model = load_llm()
#     embed_model = load_embedding_model()
#     faiss_index, data, embeddings = load_faiss()

#     query = st.text_input("Enter your query:")

#     if query:
#         with st.spinner("Processing..."):
#             # Search for relevant contexts based on the query
#             contexts = search(query, embed_model, faiss_index, data)
#             combined_context = " ".join(contexts)

#             # Generate an answer using the LLM model
#             response = generate_response(combined_context, query, tokenizer, llm_model)

#             st.subheader("Response:")
#             st.write(response)

#             # st.subheader("Top Retrieved Contexts:")
#             # for idx, ctx in enumerate(contexts, 1):
#             #     st.markdown(f"**{idx}.** {ctx}")

# if __name__ == "__main__":
#     main()

###########################################
import os
import streamlit as st
import faiss
import pickle
import numpy as np
import torch
import gdown
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Install gdown if needed
os.system('pip install gdown')

# Function to download a full folder from Google Drive
def download_folder_from_google_drive(folder_url, output_path):
    if not os.path.exists(output_path):
        gdown.download_folder(url=folder_url, output=output_path, quiet=False, use_cookies=False)

# Download individual files
def download_file_from_google_drive(file_id, destination):
    if not os.path.exists(destination):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)

# Setup models and files
@st.cache_resource
def setup_files():
    os.makedirs("models/embedding_model", exist_ok=True)
    os.makedirs("models/generator_model", exist_ok=True)
    os.makedirs("models/files", exist_ok=True)

    # Download embedding model (folder)
    download_folder_from_google_drive(
        "https://drive.google.com/drive/folders/1GzPk2ehr7rzOr65Am1Hg3A87FOTNHLAM?usp=sharing",
        "models/embedding_model"
    )

    # Download generator model (folder)
    download_folder_from_google_drive(
        "https://drive.google.com/drive/folders/1338KWiBE-6sWsTO2iH7Pgu8eRI7EE7Vr?usp=sharing",
        "models/generator_model"
    )

    # Download FAISS index, texts.pkl, embeddings.npy
    download_file_from_google_drive("11J_VI1buTgnvhoP3z2HM6X5aPzbBO2ed", "models/files/faiss_index_file.index")
    download_file_from_google_drive("1RTEwp8xDgxLnRUiy7ClTskFuTu0GtWBT", "models/files/texts.pkl")
    download_file_from_google_drive("1N54imsqJIJGeqM3buiRzp1ivK_BtC7rR", "models/files/embeddings.npy")

# Paths
EMBEDDING_MODEL_PATH = "./models/embedding_model"
GENERATOR_MODEL_PATH = "./models/generator_model"
FAISS_INDEX_PATH = "./models/files/faiss_index_file.index"
TEXTS_PATH = "./models/files/texts.pkl"
EMBEDDINGS_PATH = "./models/files/embeddings.npy"

# Load LLM model (Generator model)
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_PATH)
    return tokenizer, model

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_PATH)

# Load FAISS index and embeddings
@st.cache_resource
def load_faiss():
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        data = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return faiss_index, data, embeddings

# Search top-k contexts
def search(query, embed_model, index, data, k=5):
    query_embedding = embed_model.encode([query]).astype('float32')
    _, I = index.search(query_embedding, k)
    results = [data[i] for i in I[0] if i != -1]
    return results

# Generate response
def generate_response(context, query, tokenizer, model):
    input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=512, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app
def main():
    st.set_page_config(page_title="Clinical QA with RAG", page_icon="🩺")
    st.title("🔎 Clinical QA System (RAG + FAISS + T5)")

    st.markdown(
        """
        Enter your **clinical question** below.  
        The system will retrieve relevant context and generate an informed answer using a local model. 🚀
        """
    )

    # Download + Load everything
    setup_files()
    tokenizer, llm_model = load_llm()
    embed_model = load_embedding_model()
    faiss_index, data, embeddings = load_faiss()

    query = st.text_input("💬 Your Question:")

    if query:
        with st.spinner("🔍 Retrieving and Generating..."):
            contexts = search(query, embed_model, faiss_index, data)
            combined_context = " ".join(contexts)
            response = generate_response(combined_context, query, tokenizer, llm_model)

            st.success("✅ Answer Ready!")
            st.subheader("📄 Response:")
            st.write(response)

if __name__ == "__main__":
    main()
