# Email/SMS Spam Classifier

A Streamlit web application that classifies emails and SMS messages as spam or not spam using machine learning.

## Features

- **Real-time Classification**: Enter any email or SMS message and get instant spam detection
- **Text Preprocessing**: Automatically cleans and processes text using:
  - Tokenization
  - Lowercasing
  - Stopword removal
  - Stemming (Porter Stemmer)
  - Punctuation removal
- **Trained Model**: Uses a pre-trained scikit-learn classifier

## Installation

1. Clone the repository
```bash
git clone https://github.com/ANSH-00p/ansh.git
cd ansh
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Deployment

### Deploy on Render

1. Push code to GitHub
2. Connect your repository to [Render](https://render.com)
3. Create a new **Web Service**
4. Select Python runtime
5. Build command: `pip install -r requirements.txt`
6. Start command: `streamlit run app.py --server.port=10000 --server.address=0.0.0.0`

## Project Structure

```
├── app.py                 # Main Streamlit application
├── trained_modle.pkl      # Pre-trained ML model
├── vectorizer.pkl         # TF-IDF vectorizer
├── requirements.txt       # Python dependencies
├── Procfile              # Deployment configuration
├── runtime.txt           # Python version specification
└── README.md             # This file
```

## Technologies Used

- **Streamlit**: Web app framework
- **scikit-learn**: Machine learning library
- **NLTK**: Natural language processing
- **Pickle**: Model serialization

## Author

ANSH-00p

## License

MIT
