import boto3
import streamlit as st
import os

##AWS_REGION 
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

## s3_client
s3_client = boto3.client("s3", region_name=AWS_REGION)
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse




## import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client( service_name="bedrock-runtime",region_name=AWS_REGION )
bedrock_embeddings = BedrockEmbeddings( model_id="amazon.titan-embed-text-v2:0",client=bedrock_client )

folder_path="/tmp/"

def get_unique_id():
    return str(uuid.uuid4())

## load index
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

def get_llm():
    return ChatBedrockConverse(
        model_id=os.getenv(
            "BEDROCK_MODEL_ID",
            "amazon.nova-lite-v1:0"
        ),
        region_name=AWS_REGION,
        max_tokens=512,
        temperature=0,
    )

# get_response()
def get_response(llm, vectorstore, question):
    docs = vectorstore.similarity_search(question, k=5)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
Use only the context below to answer the question.

If the answer cannot be found in the context, say:
"I don't know based on the indexed document."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content


## main method
def main():
    st.set_page_config(
        page_title="Bedrock RAG Assistant",
        page_icon="🤖",
        layout="centered",
    )

    st.title("🤖 Bedrock RAG Document Assistant")

    st.caption(
        "Ask questions grounded in your indexed documents using "
        "Amazon Bedrock, Titan embeddings, FAISS, and semantic retrieval."
    )

    load_index()

    #dir_list = os.listdir(folder_path)
    #st.write(f"Files and Directories in {folder_path}")
    #st.write(dir_list)

    ## create index
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path = folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.success("Knowledge base loaded and ready.")
    question = st.text_input(
        "Ask a question about the indexed document",
        placeholder="What are the main ideas discussed in this document?"
    )

    if st.button("Ask Document", type="primary"):

        if not question.strip():
            st.warning("Enter a question before submitting.")
            return

        with st.spinner("Searching the document and generating an answer..."):
            llm = get_llm()

            answer = get_response(llm, faiss_index, question)

            st.subheader("Answer")
            st.write(answer)

            st.success("Response generated successfully.")

if __name__ == "__main__":
    main()