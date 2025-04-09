import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import pickle

# Load dữ liệu
faq_data = pd.read_excel("FAQ_Shop_Tui_Tai_Che.xlsx")
product_data = pd.read_excel("Product_List.xlsx")

# Load mô hình BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Hàm tạo embedding từ BERT
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Thêm giá tiền vào mô tả sản phẩm
product_data["Product_Description"] = (
    product_data["Tên sản phẩm"] + " " +
    product_data["Loại túi"].fillna("") + " " +
    product_data["Màu sắc"].fillna("") + " " +
    "Giá: " + product_data["Giá"].astype(str) + " VND"
)

# Tạo embeddings cho FAQ
faq_embeddings = np.vstack(faq_data["Câu hỏi"].apply(get_bert_embedding))

# Tạo embeddings cho sản phẩm
product_embeddings = np.vstack(product_data["Product_Description"].apply(get_bert_embedding))

# Lưu trạng thái mô hình vào file
with open("model-state.bin", "wb") as f:
    pickle.dump({
        "faq_embeddings": faq_embeddings,
        "faq_data": faq_data,
        "product_embeddings": product_embeddings,
        "product_data": product_data
    }, f)

print("Đã lưu trạng thái mô hình vào model-state.bin!")
