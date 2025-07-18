# Gastric Cancer Diagnosis with patch images from Whole Slide Images

## Features

This project provides a web application for diagnosing gastric cancer using patch images extracted from whole slide images. The application includes the following features:

- **Image Upload**: Users can upload images for analysis.
- **Image Classification**: The application classifies the uploaded images as either abnormal or normal.
- **Anomaly Detection**: The application detects and highlights abnormal regions in the uploaded images.

![Homepage](public/home.png)

![Image Classification](public/classification.png)

![Anomaly Detection](public/detection.png)

## Installation and Usage Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/sunahiri25/gastric-cancer-diagnosis.git
cd gastric-cancer-diagnosis
```

### Step 2: Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
```

### Step 3: Run the Web Application

1. **Start the Web Server**:

   ```bash
   python app.py
   ```
2. **Access the Web Application**:

   Open your browser and navigate to `http://127.0.0.1:5000`
