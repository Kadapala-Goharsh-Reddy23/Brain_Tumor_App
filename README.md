Brain Tumor MRI Classification Project
--------------------------------------

Overview
--------
This project uses Deep Learning to classify Brain MRI images into four tumor categories:
1. Glioma
2. Meningioma
3. Pituitary
4. No Tumor

The system is built using Python, TensorFlow/Keras, and Streamlit. It provides a complete pipeline including data preprocessing, model training, evaluation, and a web application for real-time predictions.

Dataset
-------
The dataset contains MRI images organized into:
- train/
- test/
- val/

Each folder contains four subfolders:
- glioma
- meningioma
- pituitary
- no_tumor

Model Features
--------------
- Image preprocessing with resizing and normalization
- Data augmentation for improving generalization
- CNN and Transfer Learning (MobileNetV2)
- Softmax output with 4 classes
- Real-time prediction using Streamlit web app

Technologies Used
-----------------
Programming: Python  
Deep Learning: TensorFlow, Keras  
App UI: Streamlit  
Utilities: NumPy, Pandas, PIL

Project Structure
-----------------
Brain-Tumor-Classification/
│
├── dataset/
│   ├── train/
│   ├── test/
│   ├── val/
│
├── models/
|   |── BrainTumor.ipynb     (Building Model)
│   ├── best_model.h5        (Output model)
│
├── Main_app.py              (Streamlit application)
|── styles.css               (Styles for UI)
└── README.txt

Installation and setup
----------------------
1. Install pip, pandas, numpy, tensorflow, keras, matplotlib and streamlit.
   
2. Run the Streamlit web application:
   streamlit run Main_app.py

3. Upload an MRI image to view:
   - Predicted tumor category.
   - Confidence score.
   - Probability bar for visualization.

Model Training
--------------
No need to train model explicitly, it's already in codebase (BrainTumor.ipynb).

Results
-------
It generalizes on unseen MRI images from the test set.
