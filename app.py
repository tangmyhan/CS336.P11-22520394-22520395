from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from Chatbot import initialize, process_question
from brain import format_docs  # Import format_docs function
from connectdb import connect_to_postgresql, execute_query
import logging
import signal
import sys
import os
import csv
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Tạo thư mục lưu trữ tạm thời cho tệp CSV
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.DEBUG)

components = initialize()
chat_history = []

@app.route("/")
def index():
    default_message = "Tôi là chatbot hỗ trợ giải đáp an toàn giao thông đường bộ, rất mong có thể giúp bạn giải đáp thắc mắc có liên quan."
    return render_template('chat.html', default_message=default_message, title="Chatbot")

@app.route("/chat", methods=["POST"])
def chatbot():
    """
    Route to handle chatbot messages.
    """
    try:
        logging.debug("Received request: %s", request.data)
        data = request.json

        question = data.get('question')
        if not question:
            logging.error("No question provided")
            return jsonify({"error": "No question provided"}), 400

        # Call the chatbot function
        answer, elapsed_time, doc = process_question(question, chat_history, components)
        # logging.debug("Chatbot response: %s", answer)
        
        # Replace newline characters with <br> tags for HTML display
        formatted_answer = answer.replace("\n", "<br>")
        
        response = jsonify({"answer": formatted_answer, "retrieval_time": elapsed_time, "context": format_docs(doc)})
        logging.debug("Sending response: %s", response.get_data(as_text=True))
        return response

    except Exception as e:
        logging.error(f"Error in chatbot route: {e}")
        return jsonify({"error": str(e)}), 500


# @app.route("/upload", methods=["GET", "POST"])
# def upload_csv():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return jsonify({"error": "No file part"}), 400
        
#         file = request.files["file"]
#         if file.filename == "":
#             return jsonify({"error": "No selected file"}), 400
        
#         if file and file.filename.endswith(".csv"):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)
            
#             # Xử lý tệp CSV và cập nhật cơ sở dữ liệu
#             try:
#                 update_database_with_csv(file_path)
#                 global components
#                 components = initialize()  # Reinitialize components to use updated data
#                 return jsonify({"success": "File uploaded and database updated successfully"}), 200
#             except Exception as e:
#                 logging.error(f"Error processing CSV: {e}")
#                 return jsonify({"error": str(e)}), 500
#         else:
#             return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

#     return render_template('upload.html', title="Upload CSV")

@app.route("/upload", methods=["GET", "POST"])
def upload_csv():
    message = None  # Biến lưu thông báo để hiển thị trên giao diện
    message_type = "success"  # Loại thông báo: success, danger, warning

    if request.method == "POST":
        if "file" not in request.files:
            message = "No file part found. Please upload a file."
            message_type = "danger"
            return render_template('upload.html', title="Upload CSV", message=message, message_type=message_type)
        
        file = request.files["file"]
        if file.filename == "":
            message = "No file selected. Please choose a file to upload."
            message_type = "danger"
            return render_template('upload.html', title="Upload CSV", message=message, message_type=message_type)
        
        if file and file.filename.endswith(".csv"):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                # Xử lý tệp CSV và cập nhật cơ sở dữ liệu
                update_database_with_csv(file_path)
                global components
                components = initialize()  # Tái khởi tạo dữ liệu sau khi cập nhật
                message = "File uploaded and database updated successfully."
                message_type = "success"
            except Exception as e:
                logging.error(f"Error processing CSV: {e}")
                message = f"Error processing CSV: {e}"
                message_type = "danger"
        else:
            message = "Invalid file format. Please upload a CSV file."
            message_type = "danger"

    return render_template('upload.html', title="Upload CSV", message=message, message_type=message_type)


@app.route("/navbar")
def navbar():
    return render_template('navbar.html')

def update_database_with_csv(file_path):
    """
    Đọc tệp CSV và cập nhật cơ sở dữ liệu PostgreSQL.
    """
    connection = connect_to_postgresql()
    if connection is None:
        raise Exception("PostgreSQL connection is not established.")

    with open(file_path, "r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Giả sử tệp CSV có các cột: title, embedding_title, content, source
            query = """
                INSERT INTO qas (title, embedding_title, content, source)
                VALUES (%s, %s, %s, %s)
            """
            execute_query_new(
                connection,
                query=query,
                params=(row["title"], row["embedding_title"], row["content"], row["source"]),
            )
    connection.close()

def execute_query_new(connection, query="", params=None):
    """
    Thực thi truy vấn SQL với tham số.
    """
    if connection is None:
        raise Exception("PostgreSQL connection is not established.")
    cursor = connection.cursor()
    cursor.execute(query, params)
    connection.commit()
    cursor.close()


def handle_sigterm(*args):
    logging.info("Received SIGTERM, shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_sigterm)
    from waitress import serve
    serve(app, host='127.0.0.1', port=5000)