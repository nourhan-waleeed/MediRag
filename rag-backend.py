import os
import glob
from typing import List
import textract
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import win32com.client


# def extract_text_from_doc(file_path: str) -> str:
#     try:
#         text = textract.process(file_path).decode('utf-8')
#         return text
#     except Exception as e:
#         print(f"Error extracting text from {file_path}: {e}")
#         return ""


def extract_text_from_doc(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == '.docx':
            text = textract.process(file_path).decode('utf-8')
            return text
        elif ext == '.doc':
            try:
                word = win32com.client.Dispatch("Word.Application")
                word.visible = False
                wb = word.Documents.Open(os.path.abspath(file_path))
                doc = word.ActiveDocument
                text = doc.Content.Text
                wb.Close()
                word.Quit()
                return text
            except Exception as inner_e:
                print(f"Could not process .doc file with Word: {inner_e}")
                try:
                    import docx2txt
                    text = docx2txt.process(file_path)
                    return text
                except Exception as docx2txt_e:
                    print(f"Could not process with docx2txt: {docx2txt_e}")
                    return ""
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def process_docs_folder(folder_path: str) -> List[Document]:
    documents = []
    doc_files = glob.glob(os.path.join(folder_path, "*.doc"))
    docx_files = glob.glob(os.path.join(folder_path, "*.docx"))
    all_files = doc_files + docx_files

    print(f"Found {len(doc_files)} .doc files and {len(docx_files)} .docx files.")

    for doc_file in all_files:
        text = extract_text_from_doc(doc_file)
        if text:
            doc = Document(
                page_content=text,
                metadata={
                    "source": doc_file,
                    "file_name": os.path.basename(doc_file),
                    "extension": os.path.splitext(doc_file)[1]
                }
            )
            documents.append(doc)

    return documents




if __name__ == "__main__":
    docs_folder = "datatest"
    db_directory = "databasetest"

    documents = process_docs_folder(docs_folder)
    print(f"Extracted text from {len(documents)} documents.")

    print("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating Chroma vector store...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_directory,
        collection_name="collection"

    )

    vectorstore.persist()

    print("Successfully created and persisted Chroma vector database!")

    print("\nExample retrieval:")
    query = "Your example query here"
    docs = vectorstore.similarity_search(query, k=3)
    print(f"Found {len(docs)} relevant documents for query: '{query}'")
    for i, doc in enumerate(docs):
        print(f"\nResult {i + 1}:")
        print(f"Source: {doc.metadata['file_name']} ({doc.metadata['extension']})")
        print(f"Content preview: {doc.page_content[:200]}...")
