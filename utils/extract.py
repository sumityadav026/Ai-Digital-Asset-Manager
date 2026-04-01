from PyPDF2 import PdfReader
from docx import Document

def extract_text(file_path):
    text = ""

    try:
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content

        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text

    except Exception as e:
        print("Error extracting:", e)

    return text