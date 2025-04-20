import eventlet
eventlet.monkey_patch()
from flask import Flask, request, jsonify
import pickle
import re
import numpy as np
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from train_model import get_bert_embedding
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# --- Load dữ liệu ---
with open("model-state.bin", "rb") as f:
    data = pickle.load(f)
    faq_embeddings = data["faq_embeddings"]
    faq_data = data["faq_data"]
    product_embeddings = data["product_embeddings"]
    product_data = data["product_data"]
    print(product_data)

# --- Tiền xử lý dữ liệu ---
product_data["Giá"] = pd.to_numeric(product_data["Giá"], errors="coerce")
product_data = product_data.dropna(subset=["Giá"])

# --- Khởi tạo Gemini ---
genai.configure(api_key="AIzaSyAOZQcVWDnbnVU0tOtDAD5AvFhPR6UJZ_Y")
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# --- Phân loại câu hỏi đa nhiệm ---
def classify_question_types(user_question):
    type_list = [
        "Câu hỏi thường gặp",
        "Hỏi giá sản phẩm",
        "Tìm sản phẩm",
        "Hỏi về khuyến mãi",
        "Hỏi về vận chuyển",
        "Hỏi về thanh toán",
        "Hỏi về bảo hành",
    ]
    
    try:
        response = model.generate_content(
            f"Phân tích câu hỏi và liệt kê TẤT CẢ loại phù hợp (chỉ trả lời tên loại, cách nhau bằng dấu phẩy): '{user_question}'"
            f"Danh sách loại: {', '.join(type_list)}"
            "Ví dụ: 'Hỏi giá sản phẩm, Hỏi về khuyến mãi'"
        )
        if response and response.parts:
            return [t.strip() for t in response.text.split(",") if t.strip() in type_list]
    except Exception as e:
        print("Lỗi phân loại:", e)
    return []

# --- Xử lý từng loại câu hỏi ---
def handle_price_inquiry(question, product_name=None):
    if product_name:
        products = product_data[product_data["Tên sản phẩm"].str.contains(product_name, case=False)]
    else:
        products = find_relevant_products(question)
    
    if products.empty:
        return "Xin lỗi, tôi không tìm thấy thông tin giá."
    
    min_price, max_price = extract_price_range(question)
    if min_price is not None:
        products = products[(products["Giá"] >= min_price) & (products["Giá"] <= max_price)]
    
    if products.empty:
        return "Không có sản phẩm phù hợp với khoảng giá bạn yêu cầu."
    
    return f"{products.iloc[0]['Tên sản phẩm']} có giá {products.iloc[0]['Giá']:,.0f} VND."

def handle_promotion_inquiry(question):
    promotions = {
        "iphone": "Giảm 10% khi mua kèm Apple Care",
        "samsung": "Tặng phiếu mua hàng 1 triệu",
    }
    
    for keyword, promo in promotions.items():
        if keyword in question.lower():
            return f"Khuyến mãi hiện tại: {promo}"
    return "Hiện không có khuyến mãi đặc biệt."

def handle_product_search(question):
    products = find_relevant_products(question)
    if products.empty:
        return "Không tìm thấy sản phẩm phù hợp."
    
    response = "Có thể bạn quan tâm:"
    for _, row in products.head(3).iterrows():
        response += f"- {row['Tên sản phẩm']} ({row['Giá']:,.0f} VND) ({row['Hình']})"
    return response

def generate_combined_response(question, partial_responses):
    prompt = f"""
    Tổng hợp các thông tin sau thành câu trả lời tự nhiên:
    Câu hỏi: {question}
    Thông tin thành phần:
    {chr(10).join(partial_responses)}
    
    Yêu cầu:
    - Bạn là trợ lý ảo của cửa hàng RECO
    - Viết ngắn gọn, thân thiện
    - Giữ nguyên thông tin quan trọng
    - Không thêm thông tin không có trong dữ liệu
    """
    try:
        response = model.generate_content(prompt)
        return response.text if response else " ".join(partial_responses)
    except:
        return " ".join(partial_responses)

# --- Hàm chính ---
def find_best_match(user_question):
    question_types = classify_question_types(user_question)
    if not question_types:
        return ask_gemini_directly(user_question)
    
    partial_responses = []
    for q_type in question_types:
        if q_type == "Hỏi giá sản phẩm":
            partial_responses.append(handle_price_inquiry(user_question))
        elif q_type == "Hỏi về khuyến mãi":
            partial_responses.append(handle_promotion_inquiry(user_question))
        elif q_type == "Tìm sản phẩm":
            partial_responses.append(handle_product_search(user_question))
    
    return generate_combined_response(user_question, partial_responses)

# --- Các hàm phụ trợ ---
def extract_price_range(text):
    text = text.lower().replace(",", "").replace(".", "")
    pattern = r"(\d+)(?:\s*(k|nghìn|triệu|tr|vnd))?"
    matches = re.findall(pattern, text)
    prices = []
    for num, unit in matches:
        num = float(num)
        if unit in ["k", "nghìn"]:
            num *= 1000
        elif unit in ["triệu", "tr"]:
            num *= 1000000
        elif unit == "vnd":
            pass
        prices.append(num)
    
    if "dưới" in text and prices:
        return 0, min(prices)
    elif "trên" in text and prices:
        return max(prices), float('inf')
    elif "từ" in text and len(prices) >= 2:
        return min(prices), max(prices)
    elif prices:
        return min(prices), max(prices)
    
    return None, None

def find_relevant_products(user_question):
    try:
        user_embedding = get_bert_embedding(user_question)
        similarities = cosine_similarity(user_embedding, product_embeddings)
        best_idxs = np.argsort(similarities[0])[-5:][::-1]
        best_idxs = [idx for idx in best_idxs if idx < len(product_data)]
        
        if not best_idxs:
            return None
            
        return product_data.iloc[best_idxs]
    except Exception as e:
        print("Lỗi khi tìm sản phẩm:", str(e))
        return None

def ask_gemini_directly(user_question):
    try:
        context = ""
        if not faq_data.empty:
            context += "Thông tin Câu hỏi thường gặp:"
            for _, row in faq_data.iterrows():
                context += f"Q: {row['Câu hỏi']}A: {row['Câu trả lời']}"
        
        if not product_data.empty:
            context += "Thông tin Sản phẩm:"
            for _, row in product_data.head(5).iterrows():
                context += f"{row['Tên sản phẩm']} - {row['Giá']:,.0f} VND"
        
        prompt = f"""Bạn là trợ lý ảo của cửa hàng RECO. Hãy trả lời câu hỏi dựa trên thông tin sau:

{context}

Câu hỏi: {user_question}

Hướng dẫn:
1. Trả lời ngắn gọn, thân thiện
2. Nếu không có thông tin, nói "Xin lỗi, tôi chưa có thông tin về vấn đề này"
3. Không bịa thông tin"""
        
        response = model.generate_content(prompt)
        return response.text if response and response.parts else "Xin lỗi, tôi chưa thể trả lời câu hỏi này."
        
    except Exception as e:
        print("Lỗi khi hỏi Gemini:", str(e))
        return "Xin lỗi, có lỗi khi xử lý câu hỏi của bạn."


# --- Khởi tạo Flask ---
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000"], async_mode='eventlet')

# --- Route xử lý câu hỏi ---
@app.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({'error': 'Câu hỏi không hợp lệ'}), 400

    # Gọi hàm xử lý thực sự
    response = find_best_match(user_question)

    # Gửi kết quả về WebSocket
    socketio.emit('bot_response', {'response': response})

    return jsonify({'response': response})

# --- WebSocket events ---
@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

# --- Chạy server ---
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
