🚢 Titanic - ML & DL Model Deployment using Streamlit
Welcome to the Titanic Survival Prediction Project!
This project combines the power of Machine Learning and Deep Learning to predict which passengers survived the Titanic disaster, and then deploys both models using a user-friendly Streamlit web app.

🎯 Project Goal
Build, compare, and deploy both ML and DL models to predict survival on the Titanic dataset. This project is designed to:

Demonstrate practical ML & DL implementation

Compare performance of traditional ML vs modern DL

Build a clean, interactive web interface using Streamlit

🧠 Models Used
✅ Machine Learning Model (ML)
Algorithm: Logistic Regression

Preprocessing: Label Encoding, Scaling

Evaluation: Accuracy, Precision, Recall, F1 Score

✅ Deep Learning Model (DL)
Framework: Keras (TensorFlow backend)

Layers: Dense, Dropout

Activation: ReLU, Sigmoid

Evaluation: Accuracy, Loss

📊 Results Comparison
Metric	ML Model	DL Model
Accuracy	0.82	0.87
Precision	0.80	0.85
Recall	0.78	0.83
F1 Score	0.79	0.84

👉 Deep Learning model slightly outperformed the ML model!

🖥️ Streamlit Deployment
The project includes a deployed Streamlit App where users can input passenger features and instantly see survival predictions from both models.

bash
Copy
Edit
streamlit run streamlit_app.py
📁 Project Structure
bash
Copy
Edit
.
├── app.py                # Streamlit App Launcher
├── streamlit_app.py      # Core app logic (UI + prediction)
├── ml_model.pkl          # Trained ML model
├── dl_model.h5           # Trained DL model
├── scaler.pkl            # Scaler used for preprocessing
├── requirements.txt      # Dependencies
└── README.md             # Project Overview
🧪 How to Run Locally
Clone the Repo

bash
Copy
Edit
git clone https://github.com/Mubashar228/Titanic-ML-and-DL-model-deployment-using-streamlit.git
cd Titanic-ML-and-DL-model-deployment-using-streamlit
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the App

bash
Copy
Edit
streamlit run streamlit_app.py
📌 Technologies Used
Python

Pandas, NumPy, Scikit-learn

TensorFlow / Keras

Streamlit

Matplotlib, Seaborn

Pickle & Joblib for model saving

💡 Future Improvements
Add more models (Random Forest, XGBoost)

Add performance visualization in the app

Deploy on cloud (e.g. Streamlit Cloud / HuggingFace / AWS)

🙋‍♂️ About the Author
Mubashar Ul Hassan
🎓 Data Scientist | 📊 Big Data Enthusiast | 🧠 AI Explorer
🔗 LinkedIn Profile
📧 Contact: your.email@example.com

⭐ If you like this project
Star 🌟 this repository

Share it with your friends
![ml_vs_dl_comparison](https://github.com/user-attachments/assets/194587a3-e7ce-4e90-99f4-363e0ae30264)

Feel free to fork and improve!

📌 License
This project is open-source and available under the MIT License.
