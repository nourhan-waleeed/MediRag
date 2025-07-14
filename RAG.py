from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
import logging

logger = logging.getLogger(__name__)


genai.configure(api_key="AIzaSyB0ONNZHqS0-ZA0Ms59X1iQU3Nog4XsFbU")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    source: str


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

persist_directory = r"C:\Users\nourh\OneDrive\Desktop\apps\MediRag\db"
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name ="collection"
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print('base retriever',base_retriever)
async def make_rag_prompt(question: str, context: str)->str:
    prompt = (
        f"You are a medical assistant specialized in radiology report generation. "
        f"Your task is to review the following clinical or radiological information:\n\n"
        f"{context}\n\n"
        f"Based on this information, answer the following question accurately and clearly:\n"
        f"{question}\n\n"
        f"Respond in a professional and structured medical report format, in clear way. "
        f"Include clinical observations, possible diagnoses, and any recommendations. "
        f"If the context is insufficient, ask for more details.\n\n"
        f"Your report:"
    )

    return prompt

async def generate_response(question: str,direct=False) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")

    if direct:
        print('direct model')

        prompt = (
            f"You are a medical assistant specialized in radiology report generation. "
            f"{question}\n\n"
            f"Respond in a professional and structured medical report format, in clear English. "
            f"Include clinical observations, possible diagnoses, and any recommendations. "
            f"If the context is insufficient, ask for more details.\n\n"
            f"Your report:"
        )

        return model.generate_content(prompt).text
    return model.generate_content(question).text


collection = vectorstore.get()

print("=" * 50)
print("BASIC VECTORSTORE INSPECTION")
print("=" * 50)

print(f"Total documents: {len(collection['ids'])}")
print(f"Collection keys: {collection.keys()}")

if collection['ids']:
    print(f"First 5 document IDs: {collection['ids'][:5]}")
    print(f"Sample metadata: {collection['metadatas'][:2] if collection['metadatas'] else 'None'}")
    print(f"Sample documents: {collection['documents'][:2] if collection['documents'] else 'None'}")


async def get_answer(question: str) -> AnswerResponse:
    print('into get answer')
    scores =[]
    context_lis = []
    combined_context = " ".join(context_lis)
    print('context',combined_context)
    relevant_docs = vectorstore.similarity_search_with_score(
        query=question, k=3,
    )
    if not relevant_docs:
        print('No relevant documents found, using direct model')
        direct_answer = await generate_response(question, direct=True)
        return AnswerResponse(
            answer=direct_answer,
            source="Direct Model - No Relevant Context"
        )
    rag_prompt =  await make_rag_prompt(question, combined_context)
    rag_answer = await generate_response(rag_prompt)

    for doc, score in relevant_docs:
        print(f"* [SIM={score / 10}] {doc.page_content} [{doc.metadata}]")
        scores.append(score/10)
        context_lis.append(doc.page_content)
        if scores[0] >40:
            print('first score',scores[0])
            direct_answer =  await generate_response(question, direct=True)
            print('direct',direct_answer)
            # await insert_caches(question,direct_answer)
            return AnswerResponse(
                answer=direct_answer,
                source="Direct Model",

            )
        # await insert_caches(question,rag_answer)
        return AnswerResponse(
            answer=rag_answer,
            source="RAGG",
        )




@app.get("/")
async def root():
    return {"message": "RAG"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    print('questionnnn')
    print(request.question)
    return await get_answer(request.question)


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.10.185", port=8888)