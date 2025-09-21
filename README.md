# 🧠 Brain Tumor Detection using Deep Learning  

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org)  [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io)  [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)  [![](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)  [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)  [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=plotly&logoColor=white)](https://matplotlib.org)  

---

## 🩺 Introduction  

Brain tumors are among the most severe neurological disorders, requiring **early detection** for effective treatment. Manual diagnosis from MRI scans is time-consuming and subject to human error. This project leverages **Deep Learning (CNNs)** to automatically classify brain MRI images into **tumor vs. no-tumor** categories with high accuracy.  

By integrating **convolutional neural networks, image preprocessing, and augmentation techniques**, the system aids radiologists and healthcare professionals in making faster, more reliable decisions.  

---

## 🔍 Problem Statement  

- Detecting tumors from MRI scans requires **expert-level precision**.  
- Manual interpretation is **time-intensive** and prone to bias.  
- There is a need for an **automated, scalable, and accurate deep learning model** to support radiologists in decision-making.  

This project focuses on **binary classification (tumor vs. non-tumor)** using CNN architectures.  

---

## 📊 Dataset  

The dataset consists of **MRI brain scan images** stored in an organized folder structure:  

- `yes/` → Images with brain tumors.  
- `no/` → Images without brain tumors.  

**Preprocessing steps include:**  

- Resizing images to a uniform shape (e.g., 128x128).  
- Normalizing pixel values.  
- Data augmentation (rotation, flipping, zooming) to reduce overfitting.  

---

## ⚙️ Deep Learning Approach  

1. **Image Preprocessing** → Normalize, resize, and augment images.  
2. **CNN Model Architecture** → Stacked convolutional layers, pooling layers, and fully connected layers.  
3. **Activation Functions** → ReLU, Softmax for final classification.  
4. **Regularization** → Dropout layers to prevent overfitting.  
5. **Optimization** → Adam optimizer with learning rate scheduling.  

### 🔧 Tools & Libraries  

- TensorFlow / Keras → CNN model building.  
- OpenCV → Image preprocessing.  
- Scikit-learn → Model evaluation metrics.  
- Matplotlib → Visualization of training results.  

---

## 📈 Evaluation  

The model is evaluated on test MRI scans using classification metrics:  

- [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)  
- [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)  
- [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)  
- [F1 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)  
- Confusion Matrix  

**Visualization of Results:**  

- Training vs. Validation Accuracy curve.  
- Training vs. Validation Loss curve.  
- Sample predictions with heatmaps (Grad-CAM).  

---

## 🚀 Results  

- Achieved **high accuracy** in classifying tumor vs. non-tumor MRI scans.  
- Model generalizes well due to **data augmentation & dropout layers**.  
- Potential to integrate into **clinical decision support systems**.  

---

## 🛠 Future Scope  

- Expand to **multi-class tumor classification** (e.g., meningioma, glioma, pituitary).  
- Deploy as a **Flask/Django web app** for hospital usage.  
- Integrate **Grad-CAM visualizations** for explainability.  
- Explore **transfer learning (ResNet, EfficientNet, VGG16)** for improved accuracy.  

---

## ✅ Outcomes 

- Automates brain tumor detection using deep learning.
- Provides faster diagnosis support for healthcare professionals.
- Reduces human error and improves early treatment chances.
- Acts as a baseline for AI-driven medical imaging solutions. 

---

## 💻 Directions to Download & Run  

1. You'll have to download and install Git which could be used for cloning the repositories that are present. The link to download Git is https://git-scm.com/downloads.
 
&emsp;&emsp; <img src = "https://github.com/Sankethprasad09/Images/blob/68ec804a501be2c8861643e707166d33a5d0ccd2/washington_bike_prediction_images/Screenshot%20(14).png" width = "600"/>
 
2. Once "Git" is downloaded and installed, you'll have to right-click on the location where you would like to download this repository. I would like to store it in the "Git Folder" location. 

&emsp;&emsp; <img src = "https://github.com/Sankethprasad09/Images/blob/68ec804a501be2c8861643e707166d33a5d0ccd2/washington_bike_prediction_images/Screenshot%20(15).png" width = "600" />

3. If you have successfully installed Git, you'll get an option called "Gitbash Here" when you right-click on a particular location. 

&emsp;&emsp; <img src = "https://github.com/Sankethprasad09/Images/blob/68ec804a501be2c8861643e707166d33a5d0ccd2/washington_bike_prediction_images/Screenshot%20(16).png" width = "600" />


4. Once the Gitbash terminal opens, you'll need to write "Git clone" and then paste the link to the repository.

5. The link to the repository can be found when you click on "Code" (Green button) and then, there would be an HTML link just below. Therefore, the command to download a particular repository should be "Git clone HTML" where the HTML is replaced by the link to this repository. 

&emsp;&emsp; <img src = "https://github.com/Sankethprasad09/Images/blob/68ec804a501be2c8861643e707166d33a5d0ccd2/washington_bike_prediction_images/Screenshot(17).png" width = "600" />

6. After successfully downloading the repository, there should be a folder with the name of the repository as can be seen below.

&emsp;&emsp; <img src = "https://github.com/Sankethprasad09/Images/blob/60b3eb7b600fef4bc619331ba5caf21e93b149ef/washington_bike_prediction_images/Screenshot%20(19).png" width = "600" />

7. Once the repository is downloaded, go to the start button and search for "Anaconda Prompt" if you have anaconda installed. 

&emsp;&emsp; <img src = "https://github.com/Sankethprasad09/Images/blob/60b3eb7b600fef4bc619331ba5caf21e93b149ef/washington_bike_prediction_images/Screenshot%20(20).png" width = "600" />

8. Later, open the Jupyter notebook by writing "Jupyter notebook" in the Anaconda prompt. 

&emsp;&emsp; <img src = "https://github.com/Sankethprasad09/Images/blob/60b3eb7b600fef4bc619331ba5caf21e93b149ef/washington_bike_prediction_images/Screenshot(21).png" width = "600" />

9. Now the following would open with a list of directories. 

&emsp;&emsp; <img src = "https://github.com/Sankethprasad09/Images/blob/60b3eb7b600fef4bc619331ba5caf21e93b149ef/washington_bike_prediction_images/Screenshot(22).png" width = "600" />

10. Search for the location where you have downloaded the repository. Be sure to open that folder. 

&emsp;&emsp; <img src = "https://github.com/Sankethprasad09/Images/blob/60b3eb7b600fef4bc619331ba5caf21e93b149ef/washington_bike_prediction_images/Screenshot%20(12).png" width = "600" />

11. You might now run the .ipynb files present in the repository to open the notebook and the python code present in it. 

&emsp;&emsp; <img src = "https://github.com/Sankethprasad09/Images/blob/60b3eb7b600fef4bc619331ba5caf21e93b149ef/washington_bike_prediction_images/Screenshot(13).png" width = "600" />

That's it, you should be able to read the code now. Thanks. 
