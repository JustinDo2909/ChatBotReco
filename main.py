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
from PIL import Image
import torch
import torchvision.transforms as transforms
import base64
from io import BytesIO
import torch.nn as nn
import torch.nn.functional as F

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
    - Không trả về hình ảnh, link hình ảnh và fileds hình ảnh
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
            return pd.DataFrame()
            
        return product_data.iloc[best_idxs]
    except Exception as e:
        print("Lỗi khi tìm sản phẩm:", str(e))
        return pd.DataFrame()

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
3. Không bịa thông tin
4. Không trả về hình ảnh, link hình ảnh và fileds hình ảnh
"""
        
        response = model.generate_content(prompt)
        return response.text if response and response.parts else "Xin lỗi, tôi chưa thể trả lời câu hỏi này."
        
    except Exception as e:
        print("Lỗi khi hỏi Gemini:", str(e))
        return "Xin lỗi, có lỗi khi xử lý câu hỏi của bạn."

# --- Khởi tạo Flask ---
app = Flask(__name__)
CORS(app, origins=["https://reco-fe.vercel.app"], supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins=["https://reco-fe.vercel.app"], async_mode='eventlet')

# --- Định nghĩa các phép biến đổi --- #
new_transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = NeuralNet()
net.load_state_dict(torch.load('trained_model.pth'))
net.eval()

class_name = ['Backpack', 'Crossbody Bag', 'Handbag', 'Shoulder Bag', 'Tote Bag', 'clutch bag']

# # Hàm load hình ảnh
# def load_image(image_data):
#     image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
#     image = new_transform(image)
#     image = image.unsqueeze(0)  # Thêm chiều batch vào (nếu cần)
#     return image

# --- Hàm xử lý ảnh ---
def process_image(image_data):
    try:
        # Xử lý base64 (bỏ phần header nếu có)
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
        image = new_transform(image).unsqueeze(0)
        return image
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

# --- Hàm dự đoán ---
def predict_image(image_tensor):
    with torch.no_grad():
        outputs = net(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return class_name[predicted.item()]

# --- Hàm gọi Gemini ---
def generate_product_description(product_class):
    similar_products = find_similar_products(product_class)
    prompt = f"""
    Bạn là trợ lý ảo cho cửa hàng thời trang RECO. Hãy tạo 1 dòng mô tả sản phẩm hấp dẫn với các thông tin sau:
    {similar_products}
    
    note : không cần trả về hình ảnh, chỉ ghi giới thiệu thôi
    """
    try:
        response = model.generate_content(prompt)
        return response.text if response else "Sản phẩm chất lượng cao, thiết kế thời thượng"
    except Exception as e:
        print(f"Gemini error: {str(e)}")
        return "Mô tả sản phẩm đang được cập nhật"

# --- Hàm tìm sản phẩm tương tự ---
def find_similar_products(product_class, top_n=3):
    try:
        similar = product_data[product_data['Loại túi'].str.contains(product_class, case=False, na=False)]
        if similar.empty:
            return []
            
        similar = similar.sort_values('Giá', ascending=False).head(top_n)
        return similar[['Tên sản phẩm', 'Giá', 'Hình']].to_dict('records')
    except Exception as e:
        print(f"Error finding similar products: {str(e)}")
        return []

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)
    emit('connection_response', {'status': 'connected', 'message': 'Kết nối thành công với RECO AI'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)

@socketio.on('ask_question')
def handle_image_upload(data):
    try:
        # Gửi trạng thái đang xử lý
        emit('processing_status', {
            'status': 'processing',
            'message': 'Đang phân tích hình ảnh...'
        })

        # 1. Nhận ảnh và xử lý thành tensor
        image_tensor = process_image(data['image'])

        # 2. Dự đoán loại sản phẩm
        product_class = predict_image(image_tensor)

        emit('processing_status', {
            'status': 'classified',
            'message': f'Đã nhận diện: {product_class}'
        })

        # 3. Gọi hàm sinh mô tả và lấy ảnh gợi ý
        description = generate_product_description(product_class)
        similar_products = find_similar_products(product_class, top_n=3)
        image_paths = [item['Hình'] for item in similar_products]

        # 4. Trả kết quả cuối cùng
        emit('prediction_result', {
            'product_class': product_class,
            'description': description,
            'image_path': image_paths  # mảng ảnh
        })

    except Exception as e:
        print(f"Error in image processing: {str(e)}")
        emit('prediction_error', {
            'error': str(e)
        })

   
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file).convert("RGB")
        image_tensor = new_transform(image).unsqueeze(0)
        prediction = predict_image(image_tensor)
        description = generate_product_description(prediction)
        similar_products = find_similar_products(prediction, top_n=3)
        image_paths = [item['Hình'] for item in similar_products]
        return jsonify({'prediction': prediction, 'description': description , 'image_path': image_paths}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
     
# --- Route xử lý câu hỏi --- #
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # --- 1. Lấy dữ liệu từ yêu cầu gửi lên ---
        user_question = request.form.get('question')
        image_file = request.files.get('image')

        prediction = None
        description = None
        similar_products = []
        image_paths = []

        # --- 2. Nếu có ảnh thì phân tích ---
        if image_file:
            image = Image.open(image_file).convert("RGB")
            image_tensor = new_transform(image).unsqueeze(0)
            prediction = predict_image(image_tensor)
            description = generate_product_description(prediction)
            similar_products = find_similar_products(prediction, top_n=3)
            image_paths = [item['Hình'] for item in similar_products]

        # --- 3. Nếu có câu hỏi thì xử lý tìm câu trả lời ---
        answer = None
        if user_question:
            answer = find_best_match(user_question)

        # --- 4. Trả về kết quả ---
        return jsonify({
            'question': user_question,
            'answer': answer,
            'prediction': prediction,
            'description': description,
            'similar_products': similar_products,
            'similar_image_paths': image_paths
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Run server --- #
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
