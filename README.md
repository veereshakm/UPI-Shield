# UPI Fraud Detection System 🛡️💸

A comprehensive web-based application designed to detect fraudulent UPI (Unified Payments Interface) transactions in real-time using Deep Learning. This system empowers users and administrators to identify potential risks, analyze transaction patterns, and maintain a secure financial environment.

## 🚀 Features

*   **Real-time Fraud Prediction**: Instantly analyze transaction details (Amount, Category, Time, Location, Age) to predict the likelihood of fraud.
*   **Deep Learning Model**: Powered by a custom Convolutional Neural Network (CNN) trained on transaction data for high accuracy (~99%).
*   **Admin Dashboard**: Secure login for administrators to manage data and oversee the system.
*   **Data Management**: Upload new transaction datasets (CSV) and retrain the model directly from the web interface.
*   **Insightful Analysis**: Visual dashboard displaying algorithm performance comparisons and fraud statistics.
*   **Premium UI/UX**: Modern, responsive "Glassmorphism" design for a seamless user experience.

## 🛠️ Tech Stack

*   **Backend**: Python, Flask
*   **Machine Learning**: TensorFlow (Keras), Scikit-learn, Pandas, NumPy
*   **Frontend**: HTML5, CSS3 (Custom Glassmorphism), JavaScript
*   **Data Processing**: Pandas, Joblib (for scaler serialization)

## 📂 Project Structure

```
UPI_F/
├── app.py                 # Main Flask application entry point
├── requirements.txt       # Python dependencies
├── model/
│   ├── train_model.py     # Script to train the CNN model
│   ├── fraud_model.h5     # Trained model artifact
│   └── scaler.pkl         # Saved scaler for data preprocessing
├── static/
│   ├── css/
│   │   └── style.css      # Custom styling
│   └── images/            # UI assets (hero images, logos)
├── templates/             # HTML Templates
│   ├── base.html          # Base layout
│   ├── index.html         # Home page
│   ├── login.html         # Admin login
│   ├── predict.html       # Fraud prediction form
│   ├── upload.html        # Data upload & retraining
│   └── analysis.html      # Analysis dashboard
└── data/
    └── transactions.csv   # Dataset storage
```

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/veereshakm/UPI-Shield
    cd upi-fraud-detection
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    python app.py
    ```

5.  **Access the App**
    Open your browser and navigate to `http://127.0.0.1:5000`.

## ☁️ Streamlit Cloud Deployment

If you prefer to host on **Streamlit Cloud**, use the `streamlit_app.py` file instead of `app.py`.

1.  **Run Locally**:
    ```bash
    streamlit run streamlit_app.py
    ```
2.  **Deploy**:
    - Push your code to GitHub.
    - Go to https://upi-shield-x4epalaergpssfirhgzwsh.streamlit.app
    - Connect your repo and set the "Main file path" to `streamlit_app.py`.

## 📖 Usage Guide

1.  **Home**: Overview of the system features.
2.  **Check Transaction**: Go to the "Check Transaction" page to test specific scenarios. Enter details like UPI ID, Amount, Category, Time, etc., to get a "Fraud" or "Valid" prediction.
3.  **Admin Login**: Log in with default credentials (Username: `admin`, Password: `admin`) to access restricted features.
4.  **Data Management (Admin)**: Upload a new `csv` dataset to update the system's knowledge base. Click "Train Model" to retrain the CNN with the new data.
5.  **Analysis**: View charts and statistics about the model's performance and fraud trends.

## 📊 Dataset Schema

The system expects a CSV file with the following columns for training:
*   `trans_amount`: Amount of the transaction.
*   `category`: Transaction category (encoded).
*   `trans_hour`: Hour of the day (0-23).
*   `zip`: Zip code of the location.
*   `age`: Age of the account holder (derived from DOB).
*   `fraud_risk`: Target variable (0 for Valid, 1 for Fraud).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.
