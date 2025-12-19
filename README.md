# 🚗 CarXplain – Intelligent Car Type Classification System

## 📌 Project Overview
**CarXplain** is an end-to-end Deep Learning project that focuses on **fine-grained car type classification** using **Convolutional Neural Networks (CNNs)** and **Transfer Learning**.  
The system is capable of recognizing **car makes and models** from images, comparing multiple CNN architectures, and providing **explainable AI (XAI)** visualizations through **Grad-CAM**.  
A fully interactive **Graphical User Interface (GUI)** allows users to upload images or use real-time webcam input to test the models.

---

## 🎯 Project Objective
- Design, train, and evaluate deep learning models for image classification.
- Apply **Transfer Learning** using multiple CNN architectures.
- Compare model performance using standard evaluation metrics.
- Provide **Explainable AI** visualizations.
- Deploy models through an intuitive GUI.
- Maintain a professional **GitHub repository** with collaborative version control.

---

## 🗂 Dataset
- **Name:** Stanford Cars Dataset  
- **Description:** A fine-grained image dataset containing car makes and models with high visual similarity.
- **Usage:**
  - Training
  - Validation
  - Evaluation (same dataset, architecture-specific preprocessing)

---

## 🧠 Models & Architectures
The project experiments with **three CNN architectures**:

| Architecture | Description |
|--------------|-------------|
| 🟢 EfficientNet-B4 | High accuracy with optimized parameter efficiency |
| 🔵 InceptionV3 | Multi-scale feature extraction |
| 🟠 ResNet50 | Deep residual learning |

All models were trained using **Transfer Learning** with custom classification heads.

---

## ⚙️ Data Preprocessing
- Image resizing to `224 × 224`
- Architecture-specific preprocessing functions:
  - EfficientNet preprocessing
  - Inception preprocessing
  - ResNet preprocessing
- Data augmentation (rotation, flipping, zoom)

> ⚠️ During evaluation, **the same images** are used with **different preprocessing pipelines** to ensure fair comparison.

---

## 📊 Model Evaluation
Each model was evaluated using the following metrics:

- ✅ **Accuracy**
- 🎯 **Precision**
- 🔁 **Recall**
- 🧩 **Confusion Matrix**
- 📈 **Comparison between 3 architectures**

Evaluation results are visualized and summarized in a comparative analysis.

---

## 🔍 Explainability (XAI)
To interpret model decisions, **Grad-CAM** is implemented to:
- Highlight important regions in the input image
- Show where the model focuses when making predictions
- Improve transparency and trust in predictions

---

## 🖥 Graphical User Interface (GUI)
The project includes a fully functional GUI built using **Streamlit**.

### GUI Features:
- 📤 Image upload
- 📷 Real-time webcam detection
- 🏆 Top-3 predictions with confidence scores
- 🔥 Grad-CAM heatmap visualization
- 📊 Model comparison dashboard
- 🧠 Architecture selection (EfficientNet / ResNet / Inception)

---

## 🧪 Bonus Features ⭐
- 🎥 **Real-time inference using webcam**
- 📊 Interactive visual analytics
- 🧾 Auto-generated PDF reports for predictions
- ⚡ Optimized inference using cached models

---

## 🗃 Repository Structure
CarXplain/
│
├── data/ # Dataset structure (not uploaded)
├── models/ # Trained CNN models (.keras)
├── gui/ # Streamlit GUI applications
│ ├── Image_Analysis.py
│ ├── Real_Time.py
│ └── Model_Comparison.py
├── utils/ # Helper functions (preprocessing, Grad-CAM, loaders)
├── assets/ # CSS & UI assets
├── docs/ # Reports and documentation
├── README.md # Project documentation
└── requirements.txt # Dependencies


---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
streamlit run gui/Image_Analysis.py
```
---

## 🧑‍🤝‍🧑Roles
| Role                             | Responsibility                            |
| -------------------------------- | ----------------------------------------- |
| Data Acquisition & Preprocessing | Dataset handling and augmentation         |
| Model Building & Training        | CNN design and transfer learning          |
| Evaluation & Visualization       | Metrics, confusion matrices, comparison   |
| GUI Development                  | Streamlit interface & real-time inference |
| Documentation & Reporting        | README, reports, analysis                 |
| GitHub Management                | Version control and collaboration         |

---
## 📌 Learning Outcomes

By completing this project, the team gained experience in:

CNN-based image classification

Transfer learning techniques

Model evaluation and benchmarking

Explainable AI (Grad-CAM)

GUI-based ML deployment

Collaborative development using GitHub

---

## 🏁 Conclusion

CarXplain demonstrates a complete deep learning pipeline from data preparation to deployment.
By combining multiple CNN architectures, explainable AI, and an interactive GUI, the project delivers a robust and professional AI solution for car type classification.

---

## ⭐ Acknowledgments

Stanford Cars Dataset

TensorFlow & Keras

Streamlit Community

---

## Course: AI Skills
