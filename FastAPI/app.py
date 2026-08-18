import hashlib
import os
import shutil
import time
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "amazon.nova-lite-v1:0",
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

UPLOAD_LIMIT = 2
QUESTION_LIMIT = 20
RATE_LIMIT_WINDOW_SECONDS = 24 * 60 * 60  # 24 hours

RATE_LIMIT_TABLE = os.getenv(
    "RATE_LIMIT_TABLE",
    "bedrock-rag-demo-rate-limits",
)

TEMP_ROOT = Path("/tmp/rag_sessions")

TEMP_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# AWS Clients
# ---------------------------------------------------------

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
)

dynamodb_client = boto3.client(
    "dynamodb",
    region_name=AWS_REGION,
)

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
)


# ---------------------------------------------------------
# AI Models
# ---------------------------------------------------------

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_client,
)

llm = ChatBedrockConverse(
    model_id=BEDROCK_MODEL_ID,
    region_name=AWS_REGION,
    max_tokens=512,
    temperature=0,
)


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="Bedrock RAG Document Assistant",
    description="Upload a PDF and ask grounded questions about it.",
    version="2.0.0",
)


# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------

class QuestionRequest(BaseModel):
    document_id: str
    question: str


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_session_directory(document_id: str) -> Path:
    session_dir = TEMP_ROOT / document_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")

    if forwarded_for:
        return forwarded_for.split(",")[-1].strip()

    if request.client:
        return request.client.host

    return "unknown"


def get_visitor_id(request: Request) -> str:
    client_ip = get_client_ip(request)

    return hashlib.sha256(
        client_ip.encode("utf-8")
    ).hexdigest()


def consume_rate_limit(
    visitor_id: str,
    action: str,
    limit: int,
) -> str:
    now = int(time.time())
    expires_at = now + RATE_LIMIT_WINDOW_SECONDS

    rate_key = f"{action}#{visitor_id}"

    key = {
        "rate_key": {
            "S": rate_key,
        }
    }

    # Try twice because another request may reset an
    # expired window between our conditional operations.
    for _ in range(2):
        try:
            dynamodb_client.update_item(
                TableName=RATE_LIMIT_TABLE,
                Key=key,
                UpdateExpression=(
                    "SET expires_at = "
                    "if_not_exists(expires_at, :expires) "
                    "ADD #count :one"
                ),
                ConditionExpression=(
                    "attribute_not_exists(#count) "
                    "OR "
                    "(expires_at > :now AND #count < :limit)"
                ),
                ExpressionAttributeNames={
                    "#count": "count",
                },
                ExpressionAttributeValues={
                    ":one": {
                        "N": "1",
                    },
                    ":limit": {
                        "N": str(limit),
                    },
                    ":now": {
                        "N": str(now),
                    },
                    ":expires": {
                        "N": str(expires_at),
                    },
                },
            )

            return rate_key

        except ClientError as exc:
            error_code = exc.response[
                "Error"
            ]["Code"]

            if (
                error_code
                != "ConditionalCheckFailedException"
            ):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Demo rate-limit service "
                        "is temporarily unavailable."
                    ),
                ) from exc

        # The first update may have failed because
        # the existing 24-hour window has expired.
        try:
            dynamodb_client.update_item(
                TableName=RATE_LIMIT_TABLE,
                Key=key,
                UpdateExpression=(
                    "SET #count = :one, "
                    "expires_at = :expires"
                ),
                ConditionExpression=(
                    "expires_at <= :now"
                ),
                ExpressionAttributeNames={
                    "#count": "count",
                },
                ExpressionAttributeValues={
                    ":one": {
                        "N": "1",
                    },
                    ":now": {
                        "N": str(now),
                    },
                    ":expires": {
                        "N": str(expires_at),
                    },
                },
            )

            return rate_key

        except ClientError as exc:
            error_code = exc.response[
                "Error"
            ]["Code"]

            if (
                error_code
                != "ConditionalCheckFailedException"
            ):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Demo rate-limit service "
                        "is temporarily unavailable."
                    ),
                ) from exc

    if action == "upload":
        detail = (
            "Demo upload limit reached. "
            "Maximum 2 PDFs per visitor "
            "every 24 hours."
        )
    else:
        detail = (
            "Demo question limit reached. "
            "Maximum 20 questions per visitor "
            "every 24 hours."
        )

    raise HTTPException(
        status_code=429,
        detail=detail,
        headers={
            "Retry-After": str(
                RATE_LIMIT_WINDOW_SECONDS
            ),
        },
    )


def release_rate_limit(rate_key: str) -> None:
    try:
        dynamodb_client.update_item(
            TableName=RATE_LIMIT_TABLE,
            Key={
                "rate_key": {
                    "S": rate_key,
                }
            },
            UpdateExpression=(
                "ADD #count :minus_one"
            ),
            ConditionExpression=(
                "attribute_exists(#count) "
                "AND #count > :zero"
            ),
            ExpressionAttributeNames={
                "#count": "count",
            },
            ExpressionAttributeValues={
                ":minus_one": {
                    "N": "-1",
                },
                ":zero": {
                    "N": "0",
                },
            },
        )

    except ClientError:
        # Cleanup must never hide the original
        # upload-processing error.
        return


def create_vector_store(pdf_path: Path, document_id: str) -> int:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No readable text was found in the PDF.")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    session_dir = get_session_directory(document_id)

    vectorstore.save_local(
        folder_path=str(session_dir),
        index_name="index",
    )

    # Persist vector index in S3
    s3_prefix = f"sessions/{document_id}"

    s3_client.upload_file(
        str(session_dir / "index.faiss"),
        BUCKET_NAME,
        f"{s3_prefix}/index.faiss",
    )

    s3_client.upload_file(
        str(session_dir / "index.pkl"),
        BUCKET_NAME,
        f"{s3_prefix}/index.pkl",
    )

    return len(chunks)


def load_vector_store(document_id: str):
    session_dir = get_session_directory(document_id)

    faiss_file = session_dir / "index.faiss"
    pkl_file = session_dir / "index.pkl"

    # If container restarted, restore index from S3
    if not faiss_file.exists() or not pkl_file.exists():
        s3_prefix = f"sessions/{document_id}"

        try:
            s3_client.download_file(
                BUCKET_NAME,
                f"{s3_prefix}/index.faiss",
                str(faiss_file),
            )

            s3_client.download_file(
                BUCKET_NAME,
                f"{s3_prefix}/index.pkl",
                str(pkl_file),
            )

        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail="Document session was not found.",
            ) from exc

    return FAISS.load_local(
        folder_path=str(session_dir),
        index_name="index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def generate_answer(vectorstore, question: str) -> str:
    docs = vectorstore.similarity_search(
        question,
        k=5,
    )

    context = "\n\n".join(
        document.page_content
        for document in docs
    )

    prompt = f"""
You are a document question-answering assistant.

Use only the supplied document context to answer the question.

If the answer cannot be determined from the context, respond exactly:

"I don't know based on the indexed document."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content


# ---------------------------------------------------------
# API Routes
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "bedrock-rag-document-assistant",
    }


@app.post("/api/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    if not BUCKET_NAME:
        raise HTTPException(
            status_code=500,
            detail="BUCKET_NAME is not configured.",
        )

    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents are supported.",
        )

    visitor_id = get_visitor_id(request)

    rate_key = consume_rate_limit(
        visitor_id,
        "upload",
        UPLOAD_LIMIT,
    )

    document_id = str(uuid.uuid4())
    session_dir = get_session_directory(document_id)

    pdf_path = session_dir / "document.pdf"

    size = 0

    try:
        with pdf_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)

                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 10 MB upload limit.",
                    )

                output.write(chunk)

        chunk_count = create_vector_store(
            pdf_path,
            document_id,
        )

    except HTTPException:
        release_rate_limit(
            rate_key,
        )

        shutil.rmtree(
            session_dir,
            ignore_errors=True,
        )
        raise

    except Exception as exc:
        release_rate_limit(
            rate_key,
        )

        shutil.rmtree(
            session_dir,
            ignore_errors=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Unable to process document: {exc}",
        ) from exc

    finally:
        await file.close()

    # Remove raw PDF after indexing.
    if pdf_path.exists():
        pdf_path.unlink()

    return {
        "document_id": document_id,
        "filename": filename,
        "chunks": chunk_count,
        "status": "ready",
    }


@app.post("/api/ask")
def ask_question(
    request: Request,
    payload: QuestionRequest,
):
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    visitor_id = get_visitor_id(request)

    consume_rate_limit(
        visitor_id,
        "ask",
        QUESTION_LIMIT,
    )

    vectorstore = load_vector_store(
        payload.document_id
    )

    answer = generate_answer(
        vectorstore,
        question,
    )

    return {
        "document_id": payload.document_id,
        "question": question,
        "answer": answer,
    }


# ---------------------------------------------------------
# Browser UI
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Bedrock RAG Document Assistant</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f5f7fa;
    margin: 0;
}

.container {
    max-width: 760px;
    margin: 60px auto;
    background: white;
    padding: 38px;
    border-radius: 14px;
    box-shadow: 0 6px 25px rgba(0,0,0,0.08);
}

h1 {
    margin-bottom: 5px;
}

.subtitle {
    color: #666;
    margin-bottom: 30px;
}

.section {
    margin-top: 30px;
}

input[type=file] {
    margin-top: 10px;
}

textarea {
    width: 100%;
    min-height: 100px;
    padding: 12px;
    margin-top: 10px;
    box-sizing: border-box;
    resize: vertical;
}

button {
    margin-top: 15px;
    padding: 12px 20px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 15px;
}

.status {
    margin-top: 15px;
    padding: 12px;
    background: #f1f3f5;
    border-radius: 7px;
}

.answer {
    margin-top: 20px;
    padding: 18px;
    background: #f1f3f5;
    border-radius: 8px;
    white-space: pre-wrap;
}

.hidden {
    display: none;
}

</style>
</head>

<body>

<div class="container">

<h1>Bedrock RAG Document Assistant</h1>

<p class="subtitle">
Upload a PDF and ask questions grounded in your document.
</p>


<div class="section">

<h3>1. Upload Document</h3>

<input
    type="file"
    id="pdfFile"
    accept=".pdf,application/pdf"
>

<br>

<button onclick="uploadDocument()">
Process Document
</button>

<div
    id="uploadStatus"
    class="status hidden">
</div>

</div>


<div
    id="questionSection"
    class="section hidden">

<h3>2. Ask Your Document</h3>

<textarea
    id="question"
    placeholder="What are the main findings of this document?">
</textarea>

<br>

<button onclick="askQuestion()">
Ask Question
</button>

<div
    id="answer"
    class="answer hidden">
</div>

</div>


</div>


<script>

let documentId = null;


async function uploadDocument() {

    const fileInput =
        document.getElementById("pdfFile");

    const statusBox =
        document.getElementById("uploadStatus");

    const questionSection =
        document.getElementById("questionSection");


    if (!fileInput.files.length) {

        statusBox.classList.remove("hidden");

        statusBox.innerText =
            "Please select a PDF document.";

        return;
    }


    const formData = new FormData();

    formData.append(
        "file",
        fileInput.files[0]
    );


    statusBox.classList.remove("hidden");

    statusBox.innerText =
        "Processing document and creating embeddings...";


    questionSection.classList.add("hidden");


    try {

        const response = await fetch(
            "/api/upload",
            {
                method: "POST",
                body: formData
            }
        );


        const data =
            await response.json();


        if (!response.ok) {

            statusBox.innerText =
                data.detail || "Upload failed.";

            return;
        }


        documentId =
            data.document_id;


        statusBox.innerText =
            `Document ready. ${data.chunks} chunks indexed.`;


        questionSection.classList.remove(
            "hidden"
        );


    } catch (error) {

        statusBox.innerText =
            "Unable to process document.";

    }
}



async function askQuestion() {

    const question =
        document
        .getElementById("question")
        .value;


    const answerBox =
        document.getElementById("answer");


    if (!question.trim()) {

        answerBox.classList.remove(
            "hidden"
        );

        answerBox.innerText =
            "Please enter a question.";

        return;
    }


    if (!documentId) {

        answerBox.classList.remove(
            "hidden"
        );

        answerBox.innerText =
            "Upload a document first.";

        return;
    }


    answerBox.classList.remove(
        "hidden"
    );

    answerBox.innerText =
        "Searching your document and generating an answer...";


    try {

        const response = await fetch(
            "/api/ask",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    document_id: documentId,
                    question: question
                })
            }
        );


        const data =
            await response.json();


        if (!response.ok) {

            answerBox.innerText =
                data.detail ||
                "Unable to generate answer.";

            return;
        }


        answerBox.innerText =
            data.answer;


    } catch (error) {

        answerBox.innerText =
            "Unable to generate answer.";

    }
}

</script>

</body>

</html>
"""
