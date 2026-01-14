### 📩Text Classification System – Spam Detection <br>
## 📌 Project Overview <br>
This project is a text classification system that sorts input text messages into Spam or Not Spam (Ham) using traditional machine learning techniques. It shows the entire machine learning process, including data preprocessing, feature extraction, model training, evaluation, and deployment with Flask. <br>
## 🎯 Objective <br>
To build a machine learning model that: <br>
- Takes raw text as input
- Processes and converts text into numerical features
- Predicts the correct category (Spam / Not Spam)
- Provides an interactive interface for real-time predictions
## 📊 Dataset <br>
- Dataset: SMS Spam Collection Dataset
- Source: kaggle
- Classes: spam -> spam, ham -> not spam
The dataset includes thousands of labeled SMS messages that are suitable for binary text classification. <br>
## 🛠️ Technologies & Libraries Used <br>
- Python
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Flask
## ⚙️ Machine Learning Pipeline <br>
# 1️⃣ Text Preprocessing <br>
The following preprocessing steps were applied: <br>
- Lowercasing
- Removal of punctuation and special characters
- Tokenization
- Stopword removal using NLTK
# 2️⃣ Feature Extraction <br>
- TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert text into numerical vectors.
- This helps assign higher importance to meaningful and less frequent words.
# 3️⃣ Model Selection <br>
- Multinomial Naive Bayes was used as the classification model.
Reason for choice: <br>
- Performs well on high-dimensional and sparse text data
- Computationally efficient
- Strong baseline for NLP classification tasks
# 4️⃣ Model Evaluation <br>
The model was evaluated using: <br>
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix visualization
The model achieved high accuracy, demonstrating effective separation between spam and non-spam messages. <br>
## 🌐 Web Application (Flask) <br>
A simple Flask-based web interface was built to: <br>
- Accept user input text
- Apply the same preprocessing used during training
- Predict and display whether the message is spam or not
The UI is clean and user-friendly, with HTML templates and external CSS for styling. <br>
## 📁 Project Structure<br>
text-classifier/<br>
│<br>
├── app.py <br>
├── train.py<br>
├── model.pkl<br>
├── vectorizer.pkl<br>
├── requirements.txt<br>
│<br>
├── templates/<br>
│   └── index.html<br>
│<br>
├── static/<br>
│   └── style.css<br>
│<br>
└── data/<br>
    └── spam.csv<br>
## ▶️ How to Run the Project<br>
# 1️⃣ Clone the repository<br>
- git clone https://github.com/jathin-mygit/Spam-Detection.git <br>
# 2️⃣ Create and activate a virtual environment<br>
- python -m venv venv<br>
- venv\Scripts\activate      # Windows<br>
- source venv/bin/activate   # Linux/Mac<br>
# 3️⃣ Install dependencies<br>
- pip install -r requirements.txt<br>
# 4️⃣ Train the model<br>
- python train.py
This will: <br>
- Train the model
- Evaluate performance
- Save model.pkl and vectorizer.pkl
# 5️⃣ Run the Flask app<br>
- python app.py
Open your browser and go to: <br>
- http://127.0.0.1:5000/
## 📈 Results & Observations<br>
- The model achieved high accuracy and balanced precision and recall.
- Spam messages containing promotional or urgent language were classified correctly.
- Consistent preprocessing between training and inference ensured reliable predictions.
## Output<br>
- Home page of website
![home page](/images/home.png)
- When not spam message is given
![not spam](/images/not_spam.png)
- When spam message is given
![spam](/images/spam.png)
- Report of the model
![Report](/images/accuracy.png)