import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to compute embeddings
def compute_embeddings(texts, tokenizer, model, device, max_length=512):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling to get a fixed-size vector
    return embeddings

# Custom dataset to handle text data
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Main script
def main():
    try:
        print("Loading the IMDB dataset...")
        dataset = load_dataset('imdb')
        
        print("Initializing tokenizer and model...")
        model_name = 'roberta-large'  # Using RoBERTa-large model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Check if GPU is available and move model to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        print("Extracting texts and labels from the dataset...")
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']
        
        # Create custom datasets
        train_dataset = TextDataset(train_texts, train_labels)
        test_dataset = TextDataset(test_texts, test_labels)

        # Create DataLoaders
        batch_size = 64  # Adjust batch size according to your GPU memory
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Function to process batches and compute embeddings
        def process_batches(data_loader, tokenizer, model, device):
            all_embeddings = []
            all_texts = []
            all_labels = []
            for i, (batch_texts, batch_labels) in enumerate(data_loader):
                embeddings = compute_embeddings(batch_texts, tokenizer, model, device)
                all_embeddings.append(embeddings)
                all_texts.extend(batch_texts)
                all_labels.extend(batch_labels)
                print(f"Processed batch {i+1}/{len(data_loader)}")
            return np.concatenate(all_embeddings), all_texts, all_labels

        # Compute embeddings for training and test sets
        print("Computing embeddings for training set...")
        train_embeddings, train_texts_processed, train_labels_processed = process_batches(train_loader, tokenizer, model, device)

        print("Computing embeddings for test set...")
        test_embeddings, test_texts_processed, test_labels_processed = process_batches(test_loader, tokenizer, model, device)

        # Save embeddings to disk (optional)
        print("Saving embeddings to disk...")
        np.save('train_embeddings.npy', train_embeddings)
        np.save('test_embeddings.npy', test_embeddings)

        print("Embeddings have been saved to disk.")

        # Create DataFrames for CSV
        train_df = pd.DataFrame({
            'review': train_texts_processed,
            'label': train_labels_processed,
            'embedding': list(train_embeddings)
        })
        test_df = pd.DataFrame({
            'review': test_texts_processed,
            'label': test_labels_processed,
            'embedding': list(test_embeddings)
        })

        # Save DataFrames to CSV
        train_df.to_csv('train_embeddings.csv', index=False)
        test_df.to_csv('test_embeddings.csv', index=False)

        print("DataFrames have been saved to CSV.")

        # Train a simple classifier and check accuracy
        print("Training classifier on embeddings...")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_embeddings, train_labels_processed)

        print("Evaluating classifier on test set...")
        test_predictions = clf.predict(test_embeddings)
        accuracy = accuracy_score(test_labels_processed, test_predictions)
        print(f"Classifier accuracy: {accuracy * 100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

