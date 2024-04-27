import os

from flask import Flask, jsonify, redirect, request
from werkzeug.utils import secure_filename

from apis.api_model import AnswerResponse, ErrorResponse, GeneralResponse
from extractors.documents import DocumentsExtractor
from llm.rag import RAG
from utils import config
from utils.files import allowed_file
from utils.local_ip import get_local_ip

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = config.ALLOWED_EXTENSIONS  # 設定允許上傳的檔案副檔名
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
    print("Question: ", question)

    # 讀取資料夾內所有的 PDF 檔案
    pdf_dir = config.PDF_DIR

    # 讀取 PDF 檔案內容
    documents_extractor = DocumentsExtractor()
    documents = documents_extractor.extract(pdf_dir)

    # 將 PDF 內容傳入 RAG 內進行向量化
    rag = RAG(texts=documents)

    # 並使用 LLM OpenAI GPT 3.5 Turbo 取得問題答案
    answer = ""
    for chunk in rag.chain.stream(question):
        print(chunk)
        if chunk != "":
            answer += chunk
    print("Answer: ", answer)

    return jsonify(
        AnswerResponse(
            answer=answer
        ).to_json()
    )


@app.route('/upload/file', methods=['POST'])
def upload_file():
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
            path = os.path.join(config.PDF_DIR, filename)
            file.save(path)
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


if __name__ == '__main__':
    ip = get_local_ip()
    app.run(
        host="{}".format(ip),
        port=8000,
        debug=True
    )
