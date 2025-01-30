import tkinter as tk
from tkinter import messagebox
import kagglehub
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

# Download Sentiment140 dataset
print("Downloading Sentiment140 dataset...")
path = kagglehub.dataset_download("kazanova/sentiment140")
file_path = path + "/training.1600000.processed.noemoticon.csv"

# Load the dataset
print("Loading and preprocessing the dataset...")
data = pd.read_csv(file_path, encoding="latin-1", header=None)
data.columns = ["target", "id", "date", "query", "user", "text"]

# Preprocess Tweets
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#", "", tweet)
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
    tweet = tweet.lower()
    return tweet

data["clean_text"] = data["text"].apply(clean_tweet)

# Map target to classes
def map_target(value):
    if value == 0:
        return "Negative"
    elif value == 4:
        return "Positive"
    else:
        return "Neutral"

data["sentiment"] = data["target"].apply(map_target)

# Reduce dataset size for KNN
data = data.sample(50000, random_state=42)

# Train-Test Split
X = data["clean_text"]
y = data["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize Tweets
print("Training the model...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train KNN Classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_vect, y_train)

# Save the model and vectorizer
joblib.dump(model, "sentiment_model_knn.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Evaluate Model
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
cm = confusion_matrix(y_test, y_pred)

# Print model evaluation metrics
print(f"Model trained with accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("Confusion Matrix:")
print(cm)

# Define the class labels (can be dynamically adjusted to match actual classes present in the confusion matrix)
class_labels = ["Negative", "Positive", "Neutral"]

# Ensure that the confusion matrix is always (3, 3) by filling missing entries with zeros
cm_full = pd.DataFrame(cm, columns=class_labels[:cm.shape[1]], 
                       index=class_labels[:cm.shape[0]])

# Save the evaluation metrics to Excel
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall'],
    'Score': [accuracy, precision, recall]
})

# Save to Excel
with pd.ExcelWriter("model_evaluation_metrics.xlsx") as writer:
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    cm_full.to_excel(writer, sheet_name='Confusion Matrix')

# Tkinter GUI
app = tk.Tk()
app.title("Tweet Sentiment Analysis")
app.geometry("600x500")
is_dark_mode = False

# Initialize StringVar
filter_var = tk.StringVar(value="All")
sort_var = tk.StringVar(value="None")

# Functions
def predict_sentiment():
    tweet = tweet_entry.get()
    if not tweet.strip():
        messagebox.showwarning("Input Error", "Please enter a valid tweet!")
        return

    # Clean and vectorize the input tweet
    clean_tweet_input = clean_tweet(tweet)
    vectorized_input = vectorizer.transform([clean_tweet_input])
    prediction = model.predict(vectorized_input)[0]

    # Display the result
    messagebox.showinfo("Prediction", f"The sentiment is: {prediction}")

def toggle_theme():
    global is_dark_mode
    is_dark_mode = not is_dark_mode

    if is_dark_mode:
        app.config(bg="black")
        label_title.config(bg="black", fg="white")
        label_prompt.config(bg="black", fg="white")
        tweet_entry.config(bg="gray20", fg="white", insertbackground="white")
        button_analyze.config(bg="darkblue", fg="white")
        toggle_button.config(bg="black", fg="white", text="☀️")
    else:
        app.config(bg="white")
        label_title.config(bg="white", fg="black")
        label_prompt.config(bg="white", fg="black")
        tweet_entry.config(bg="white", fg="black", insertbackground="black")
        button_analyze.config(bg="blue", fg="white")
        toggle_button.config(bg="white", fg="black", text="🌙")

def filter_and_sort():
    selected_sentiment = filter_var.get()
    sort_by = sort_var.get()

    # Filter based on sentiment
    if selected_sentiment == "All":
        filtered_data = data
    else:
        filtered_data = data[data["sentiment"] == selected_sentiment]

    # Sort tweets
    if sort_by == "Alphabetical":
        filtered_data = filtered_data.sort_values(by="clean_text")
    elif sort_by == "Length":
        filtered_data = filtered_data.sort_values(by="clean_text", key=lambda col: col.str.len())
    
    # Debug: Print the first few entries of the filtered and sorted data
    print(filtered_data.head())

    # Ensure that filtered_data is not empty
    if filtered_data.empty:
        messagebox.showinfo("No Results", "No tweets found with the selected filter.")
        results_text.delete(1.0, tk.END)
        return

    # Clear the existing content in the text box
    results_text.delete(1.0, tk.END)

    # Insert sorted and filtered tweets into the text box
    for tweet in filtered_data["clean_text"]:
        results_text.insert(tk.END, f"{tweet}\n\n")

    # Scroll to the top to ensure the most recent results are visible
    results_text.yview(tk.END)


# GUI Widgets
label_title = tk.Label(app, text="Tweet Sentiment Analysis", font=("Arial", 16), bg="white", fg="black")
label_title.pack(pady=10)

label_prompt = tk.Label(app, text="Enter a tweet:", font=("Arial", 12), bg="white", fg="black")
label_prompt.pack(pady=5)

tweet_entry = tk.Entry(app, width=50, bg="white", fg="black")
tweet_entry.pack(pady=5)

button_analyze = tk.Button(app, text="Analyze Sentiment", command=predict_sentiment, font=("Arial", 12), bg="blue", fg="white")
button_analyze.pack(pady=20)

frame_filters = tk.Frame(app, bg="white")
frame_filters.pack(pady=10)

# Filtering and Sorting Widgets
tk.Label(frame_filters, text="Filter by Sentiment:", font=("Arial", 12), bg="white").grid(row=0, column=0, padx=5)
filter_menu = tk.OptionMenu(frame_filters, filter_var, "All", "Positive", "Negative", "Neutral")
filter_menu.grid(row=0, column=1, padx=5)

tk.Label(frame_filters, text="Sort by:", font=("Arial", 12), bg="white").grid(row=0, column=2, padx=5)
sort_menu = tk.OptionMenu(frame_filters, sort_var, "None", "Alphabetical", "Length")
sort_menu.grid(row=0, column=3, padx=5)

apply_button = tk.Button(frame_filters, text="Apply", command=filter_and_sort, font=("Arial", 12), bg="green", fg="white")
apply_button.grid(row=0, column=4, padx=5)

# Results Display
results_text = tk.Text(app, height=10, width=50, wrap=tk.WORD, bg="white", fg="black")
results_text.pack(pady=10)

# Compact toggle button for light/dark mode
toggle_button = tk.Button(app, text="🌙", command=toggle_theme, font=("Arial", 12), bg="white", fg="black")
toggle_button.place(x=560, y=10, width=30, height=30)

# Initial display of all tweets
for tweet in data["clean_text"]:
    results_text.insert(tk.END, f"{tweet}\n\n")

toggle_theme()

app.mainloop()