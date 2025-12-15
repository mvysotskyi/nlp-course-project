import fitz 

class PDFReader:
    def __init__(self):
        pass
    def read_pdf(self, file_path: str) -> str:
        document = fitz.open(file_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text