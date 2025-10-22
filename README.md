# PREDICT-AND-CLASSIFY-KNEE-OSTEOARTHRITIS-FROM-X-RAY-IMAGERY-USING-DEEP-LEARNING-
A deep learning-based web application using CNN and MobileNetV2 to predict and classify knee osteoarthritis severity from X-ray images. Built with Python and Flask, it uses image preprocessing and augmentation to enhance model accuracy, supporting early diagnosis and clinical decision-making.

---

# 🦵 Knee Osteoarthritis Severity Classification

A deep learning–based **web application** that predicts and classifies **knee osteoarthritis (KOA)** severity levels from **X-ray images** using **Convolutional Neural Networks (CNN)** and **MobileNetV2**.

---

## 📘 Project Overview

Knee Osteoarthritis (KOA) is a common degenerative joint disease often diagnosed using X-ray imagery. This project aims to assist in **early clinical diagnosis** by automatically classifying the severity of KOA into **Normal, Doubtful, Mild, Moderate, and Severe** stages using deep learning.

The system applies **image preprocessing, augmentation, and transfer learning** techniques to enhance dataset quality and model performance. The trained model is integrated into a **Flask-based web application** for easy image upload and prediction.

---

## 🚀 Features

* 🧠 Deep Learning model using **CNN and MobileNetV2**
* 🖼️ **Automatic KOA severity classification** from X-ray images
* ⚙️ **Image preprocessing and augmentation** for better accuracy
* 🌐 **Flask web interface** for uploading and predicting X-rays
* 📊 Achieved **85% model accuracy** supporting early diagnosis

---

## 🧩 Technologies Used

* **Python 3.8+**
* **Flask** – Web framework
* **CNN + MobileNetV2** – Deep learning architecture
* **NumPy, OpenCV, Matplotlib** – Image handling and visualization
* **scikit-learn** – Model evaluation metrics

---

## 🗂️ Project Structure

```
├── static/                # CSS, JS, images for the Flask app  
├── templates/             # HTML templates for the web interface  
├── model/                 # Trained model files (.h5 or .pkl)  
├── app.py                 # Main Flask application  
├── train_model.py         # Model training script  
├── requirements.txt       # Project dependencies  
└── README.md              # Project documentation  
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Knee-OA-Classification.git
   cd Knee-OA-Classification
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**

   ```bash
   python app.py
   ```

4. **Open your browser** and go to:

   ```
   http://127.0.0.1:5000/
   ```

## 📈 Model Performance

| Severity Level | Accuracy |
| -------------- | -------- |
| Severe         | 94%      |
| Moderate       | 95%      |
| Mild           | 69%      |
| Doubtful       | 65%      |
| Normal         | 83%      |

Overall average accuracy: **~85%**


## 📊 Dataset

* Combined four open-source datasets with **2,501 X-ray images**.
* Applied **data augmentation** to balance classes and improve generalization.
* Images were resized and preprocessed for optimal model training.


## 💡 Future Enhancements

* Integration of **Vision Transformers (ViT)** for higher accuracy.
* Building a **mobile-friendly interface** for doctors and patients.
* Expanding dataset with more diverse X-ray samples.


## 👩‍💻 Team

* **Essam Azeemuddin**
* **Mohammed Abdul Hakeem Siddiqui**
* **Mohammad Asadullah**
* Under the guidance of **Dr. Abdul Wajeed**

## 🏫 Institution

**LORDS Institute of Engineering and Technology**
Affiliated to **Osmania University**, Hyderabad

## 📜 License

This project is for educational and research purposes. All datasets used are publicly available.


