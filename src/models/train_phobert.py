import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "df_lstm_segmented.csv"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "phobert"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# 1. Dataset class supporting both standard and hybrid training
class PhoBertDataset(Dataset):
    def __init__(self, encodings, labels, metadata=None):
        self.encodings = encodings
        self.labels = labels
        self.metadata = metadata
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.metadata is not None:
            item["metadata"] = torch.tensor(self.metadata[idx], dtype=torch.float32)
        return item
        
    def __len__(self):
        return len(self.labels)

# 2. Custom Hybrid Model: PhoBERT + Metadata
class PhoBertWithMetadata(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", num_labels=2, metadata_dim=2, hidden_dim=64):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        phobert_hidden_size = self.phobert.config.hidden_size # 768
        
        # Classifier head combining text features and metadata
        self.classifier = nn.Sequential(
            nn.Linear(phobert_hidden_size + metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )
        
    def forward(self, input_ids, attention_mask=None, metadata=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        # Use first token (CLS token representation)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        
        # Concatenate CLS features with metadata features
        combined = torch.cat((cls_rep, metadata), dim=1)
        
        # Output logits
        logits = self.classifier(combined)
        return logits

def load_and_preprocess_data():
    print(f"Reading preprocessed data from {DATA_PATH.name}...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess labels
    label_map = {"positive": 1, "negative": 0}
    df["label_code"] = df["label"].map(label_map)
    
    # Derive 'has_media' feature based on common Shopee review media keywords
    media_keywords = ["ảnh", "video", "hình", "clip", "media", "anh", "hinh"]
    df["has_media"] = df["review"].astype(str).apply(lambda x: 1 if any(kw in x.lower() for kw in media_keywords) else 0)
    
    # Normalize rating stars (1-5 to 0.2-1.0)
    df["rating_stars"] = df["rating"] / 5.0
    
    return df

def train_epoch(model, loader, optimizer, criterion, device, is_hybrid=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        if is_hybrid:
            metadata = batch["metadata"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, metadata=metadata)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(loader), correct / total

def evaluate_model(model, loader, device, is_hybrid=False):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if is_hybrid:
                metadata = batch["metadata"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask, metadata=metadata)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_labels, all_preds

def main():
    df = load_and_preprocess_data()
    
    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_code"])
    
    # Tokenization
    print("Tokenizing review text using PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    train_texts = train_df["segmented_review"].astype(str).tolist()
    test_texts = test_df["segmented_review"].astype(str).tolist()
    
    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=128)
    test_encodings = tokenizer(test_texts, padding="max_length", truncation=True, max_length=128)
    
    # Construct metadata matrices (rating_stars and has_media)
    train_metadata = np.stack([train_df["rating_stars"].values, train_df["has_media"].values], axis=1)
    test_metadata = np.stack([test_df["rating_stars"].values, test_df["has_media"].values], axis=1)
    
    # Create datasets
    train_dataset_std = PhoBertDataset(train_encodings, train_df["label_code"].values)
    test_dataset_std = PhoBertDataset(test_encodings, test_df["label_code"].values)
    
    train_dataset_hyb = PhoBertDataset(train_encodings, train_df["label_code"].values, train_metadata)
    test_dataset_hyb = PhoBertDataset(test_encodings, test_df["label_code"].values, test_metadata)
    
    # Dataloaders (using small batch sizes for stable fine-tuning)
    train_loader_std = DataLoader(train_dataset_std, batch_size=16, shuffle=True)
    test_loader_std = DataLoader(test_dataset_std, batch_size=16, shuffle=False)
    
    train_loader_hyb = DataLoader(train_dataset_hyb, batch_size=16, shuffle=True)
    test_loader_hyb = DataLoader(test_dataset_hyb, batch_size=16, shuffle=False)
    
    epochs = 3
    lr = 2e-5
    criterion = nn.CrossEntropyLoss()
    
    # ==========================================
    # 1. Train Standard PhoBERT model
    # ==========================================
    print("\n--- Training Standard PhoBERT Fine-Tuning ---")
    std_model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2).to(device)
    optimizer_std = optim.AdamW(std_model.parameters(), lr=lr)
    
    best_std_acc = 0.0
    for epoch in range(epochs):
        loss, train_acc = train_epoch(std_model, train_loader_std, optimizer_std, criterion, device, is_hybrid=False)
        val_acc, _, _ = evaluate_model(std_model, test_loader_std, device, is_hybrid=False)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_std_acc:
            best_std_acc = val_acc
            torch.save(std_model.state_dict(), MODEL_SAVE_DIR / "phobert_standard.pt")
            print("=> Saved best standard model checkpoint.")
            
    # ==========================================
    # 2. Train Hybrid PhoBERT + Metadata model
    # ==========================================
    print("\n--- Training Hybrid PhoBERT + Metadata Model ---")
    hyb_model = PhoBertWithMetadata().to(device)
    optimizer_hyb = optim.AdamW(hyb_model.parameters(), lr=lr)
    
    best_hyb_acc = 0.0
    for epoch in range(epochs):
        loss, train_acc = train_epoch(hyb_model, train_loader_hyb, optimizer_hyb, criterion, device, is_hybrid=True)
        val_acc, labels, preds = evaluate_model(hyb_model, test_loader_hyb, device, is_hybrid=True)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_hyb_acc:
            best_hyb_acc = val_acc
            torch.save(hyb_model.state_dict(), MODEL_SAVE_DIR / "phobert_hybrid.pt")
            print("=> Saved best hybrid model checkpoint.")
            
            # Print validation report for the final best model
            final_report = classification_report(labels, preds, target_names=["negative", "positive"])
            
    print("\nTraining complete!")
    print(f"Best Standard Val Accuracy: {best_std_acc*100:.2f}%")
    print(f"Best Hybrid Val Accuracy: {best_hyb_acc*100:.2f}%")
    print("\nClassification Report (Best Hybrid Model):")
    print(final_report)

if __name__ == "__main__":
    main()
