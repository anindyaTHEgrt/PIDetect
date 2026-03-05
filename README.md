# 🏭 PIDetect - Intelligent P&ID Analysis Tool

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF.svg?style=for-the-badge&logo=ai&logoColor=black)](https://ultralytics.com/)
[![Gemini](https://img.shields.io/badge/Gemini_2.0_Flash-Google_AI-4285F4.svg?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)

> **Automate Piping and Instrumentation Diagram (P&ID) reading with AI.**
> PIDetect is an interactive web application that uses custom-trained Computer Vision models to automatically detect, classify, and highlight industrial components in P&ID schematics. It is paired with an integrated AI assistant ("PIDgpt") to help engineers and students analyze diagrams faster and more accurately.

---

## ✨ Key Features

* **🔍 Automated Object Detection:** Upload any P&ID image (or select from a gallery of examples). The app uses a fine-tuned YOLO model to instantly locate key components.
* **🎛️ Interactive Class Filtering:** A dynamic sidebar allows users to toggle specific components on and off, making complex, cluttered diagrams easier to read.
* **🤖 PIDgpt AI Assistant:** A built-in chatbot powered by Google's **Gemini 2.0 Flash**. Users can ask questions, get explanations of standard symbols, or request help directly within the application.
* **🎨 Custom Bounding Boxes:** Detected components are highlighted with color-coded bounding boxes and clear text labels using OpenCV.

### Detected Components:
Currently, the CV model (`finaltrain_best.pt`) is trained to recognize:
1. `Ball Valve 1` 
2. `Ball Valve 2`
3. `Ball Valve 3`
4. `Onsheet Connector`
5. `Centrifugal Fan`
6. `IHTL`
7. `Pneumatic Signal`
8. `NP`

---

## 🛠️ How it's Built (Tech Stack)

PIDetect is built using a modern Python data science and machine learning stack:

* **Frontend Interface:** [Streamlit](https://streamlit.io/) provides the responsive web UI, handling file uploads, image selection (`streamlit_image_select`), and the chat interface.
* **Computer Vision Engine:** [Ultralytics YOLO](https://ultralytics.com/) is used for high-speed, high-accuracy object detection.
* **Image Processing:** [OpenCV (`cv2`)](https://opencv.org/) and [Pillow (`PIL`)](https://pillow.readthedocs.io/) handle image matrix conversions and the drawing of bounding boxes/labels.
* **Large Language Model:** The new [Google GenAI SDK](https://ai.google.dev/) connects to the Gemini 2.0 Flash model to power the conversational assistant.

---

## 🚀 Getting Started

Follow these steps to run the application on your local machine.

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/PIDetect.git](https://github.com/yourusername/PIDetect.git)
cd PIDetect
