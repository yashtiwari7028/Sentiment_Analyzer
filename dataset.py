import pandas as pd
import kagglehub
import re  # Ensure this import is present

# Download Sentiment140 dataset
print("Downloading Sentiment140 dataset...")
path = kagglehub.dataset_download("kazanova/sentiment140")
file_path = path + "/training.1600000.processed.noemoticon.csv"

# Load the dataset
print("Loading and preprocessing the dataset...")
data = pd.read_csv(file_path, encoding="latin-1", header=None)
data.columns = ["target", "id", "date", "query", "user", "text"]

# Preprocess Tweets (without the 'clean_tweet' function)
data["clean_text"] = data["text"].str.replace(r"http\S+|www\S+", "", regex=True)  # Remove URLs
data["clean_text"] = data["clean_text"].str.replace(r"@\w+", "", regex=True)  # Remove mentions
data["clean_text"] = data["clean_text"].str.replace(r"#", "", regex=True)  # Remove hashtags
data["clean_text"] = data["clean_text"].str.replace(r"[^a-zA-Z\s]", "", regex=True)  # Remove special characters
data["clean_text"] = data["clean_text"].str.lower()  # Convert to lowercase

# Sample a subset of the dataset
sample_size = 30000  # Adjust this size to ensure the file does not exceed 100MB
data = data.sample(n=sample_size, random_state=42)

# Check the size of the dataset to make sure it is under 100MB
print(f"Dataset size after reduction: {data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")

# Save the reduced dataset to a new CSV file
reduced_file_path = "reduced_sentiment140.csv"
data.to_csv(reduced_file_path, index=False)

print(f"Reduced dataset saved to: {reduced_file_path}")
