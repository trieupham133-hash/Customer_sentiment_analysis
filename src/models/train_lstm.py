import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size=64001, embedding_dim=128, hidden_dim=256, output_dim=1, n_layers=2):
        super().__init__()
        
        # 1. Tầng nhúng từ (Embedding Layer) - Nhận ma trận 2D [Batch, 128] -> Trả về 3D [Batch, 128, 128]
        # padding_idx=1 (Bỏ hết pad_idx = 1) vì PhoBERT Tokenizer của VinAI quy ước ID số 1 là ký tự đệm (Pad) để đáp ứng đủ độ dài chuỗi
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        
        # 2. Tầng mạng hồi quy LSTM 2 chiều (Bidirectional LSTM)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            batch_first=True, # Đảm bảo input: [Batch, Seq_len, Embed_dim]
                            bidirectional=True, # Học ngữ cảnh xuôi và ngược
                            dropout=0.3 if n_layers > 1 else 0.0)
        
        # 3. Tầng tuyến tính phân loại đầu ra (Fully Connected)
        # Nhân đôi hidden_dim vì cơ chế Bidirectional gộp kết quả chiều xuôi + chiều ngược lại với nhau
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text):
        # text shape đầu vào: [batch_size, sequence_length] (Ví dụ: [64, 128])
        embedded = self.embedding(text) # shape sau embedding: [batch_size, 128, 128]
        
        # Chạy qua mạng LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Trích xuất hidden state cuối cùng của chiều xuôi (forward) và chiều ngược (backward)
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        
        # Nối (Concatenate) 2 chiều lại thành một vector ngữ cảnh duy nhất đại diện cho cả câu
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1) # shape: [batch_size, hidden_dim * 2]
        
        # Đưa qua tầng tuyến tính và kích hoạt Sigmoid để tính xác suất (từ 0 đến 1)
        dense_out = self.fc(hidden_concat)
        return self.sigmoid(dense_out) # Trả về giá trị xác suất có là Positive hay không