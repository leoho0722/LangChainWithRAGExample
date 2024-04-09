import os
import socket  # 用來抓區網 IP

from flask import Flask, jsonify, redirect, request
from werkzeug.utils import secure_filename

from apis.api_model import AnswerResponse, ErrorResponse, GeneralResponse
from pdf_extractor import PDFExtractor
from rag import RAG

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # 設定上傳檔案的資料夾
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS  # 設定允許上傳的檔案副檔名
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 設定檔案上限為 100MB


@app.route('/rag/qa', methods=['POST'])
def rag_qa():
    """RAG QA

    Request Body Sample:
    ```json
    {
        "question": "What is the capital of Taiwan?"
    }
    ```
    """

    # 取得使用者傳入的問題
    question = request.json["question"]

    # 讀取 PDF 檔案
    # 如果有上傳 PDF 檔案，則使用上傳的檔案
    # 否則使用預設的 PDF 檔案路徑 (driver_book.pdf/駕駛人手冊.pdf)
    try:
        pdf_filepath = _pdf_filepath if _pdf_filepath != "" else os.environ["PDF_PATH"]
    except:
        pdf_filepath = os.environ["PDF_PATH"]

    # 讀取 PDF 檔案內容
    documents = PDFExtractor.extract_text(pdf_filepath)

    # 將 PDF 內容傳入 RAG 內進行向量化
    rag = RAG(texts=documents)

    # 並使用 LLM OpenAI GPT 3.5 Turbo 取得問題答案
    answer = ""
    for chunk in rag.chain.stream(question):
        print(chunk)
        if chunk != "":
            answer += chunk
    print(answer)

    return jsonify(
        AnswerResponse(
            answer=answer
        ).to_json()
    )


@app.route('/upload/pdf', methods=['POST'])
def upload_pdf():
    # 檢查 Request 是否包含檔案
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)
    file = request.files['file']

    # 檢查檔案是否有檔名，如果為空則未選取檔案
    if file.filename == '':
        print('No selected file')
        return redirect(request.url)

    # 檢查檔案是否為 PDF
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            if not os.path.exists(UPLOAD_FOLDER):
                os.mkdir(UPLOAD_FOLDER)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            global _pdf_filepath
            _pdf_filepath = path
            return jsonify(
                GeneralResponse(
                    message=f"Upload Success, File Path: {path}"
                ).to_json()
            )
        except Exception as e:
            return jsonify(
                ErrorResponse(
                    message=f"Upload Failed, Error: {str(e)}"
                ).to_json()
            )
    else:
        return jsonify(
            ErrorResponse(
                message="File Type Not Allowed, current only allow pdf file"
            ).to_json()
        )


def allowed_file(filename):
    """判斷上傳的檔案副檔名是否包含在允許的副檔名清單內"""

    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_local_ip():
    """取得區網 IP"""

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    print("IP：{}".format(ip))
    s.close()

    return ip


if __name__ == '__main__':
    ip = get_local_ip()
    app.run(debug=True, host="{}".format(ip), port=8000)
