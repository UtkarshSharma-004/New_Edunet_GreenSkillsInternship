
# ♻️ E-Waste Generation Classification  
### 🚀 AICTE Internship – Shell.ai & Edunet Foundation  
*Intern: Utkarsh Sharma | B.Tech CSE (AI), UIET CSJMU Kanpur*

---

## 📌 Project Overview

This project aims to classify different categories of e-waste images using **MobileNetV3Large** (transfer learning) and deploy the model via an intuitive **Gradio web interface**. It was developed under the **AICTE Virtual Internship Program** in collaboration with **Shell.ai and Edunet Foundation**.

---

## 📁 Dataset (https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset)

The dataset consists of labeled images across **10 categories of e-waste**:
- Battery
- Keyboard
- Microwave
- Mobile
- Mouse
- PCB
- Player
- Printer
- Television
- Washing Machine

> 🗂️ *The dataset was organized into `train`, `validation`, and `test` folders using `image_dataset_from_directory`.*

---

## 🧠 Model Architecture

- **Base Model**: MobileNetV3Large (pretrained on ImageNet, frozen)
- **Custom Layers**:  
  - GlobalAveragePooling2D  
  - Dense(256, ReLU)  
  - Dropout(0.3)  
  - Dense(10, Softmax)

### ✅ Accuracy: **99.91%** (achieved in just 10 epochs)

---

## 🔄 Data Augmentation

To improve generalization and robustness, the following augmentations were applied:

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(factor=0.2),
    tf.keras.layers.RandomCrop(224, 224),
    tf.keras.layers.Rescaling(1./255),
])
```

---

## 📈 Model Evaluation

- Plotted training vs validation accuracy/loss curves  
- Generated **confusion matrix** and **classification report**  
- Visualized predictions on test images  

---

## 🌐 Gradio Interface

A user-friendly Gradio interface was created for real-time predictions.

```python
gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="E-Waste Image Classifier",
    description="Upload an image of e-waste to classify it into one of 10 categories."
).launch()
```

---


## 📎 Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib / Seaborn
- scikit-learn
- Gradio
- PIL

> Install using: `pip install tensorflow gradio matplotlib seaborn scikit-learn pillow`

---

## 📬 Connect with Me

- 📧 Email: your_email@example.com  
- 🔗 [LinkedIn](https://www.linkedin.com/in/utkarsh-sharma-16126a2a0/)  
- 💻 [GitHub](https://github.com/UtkarshSharma-004)

---

## 🙏 Acknowledgements

This project is a part of **AICTE Virtual Internship Program** organized by **Shell.ai and Edunet Foundation**.  
Special thanks to the mentors and coordinators for their continuous support.
