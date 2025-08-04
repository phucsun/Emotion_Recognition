# 🎥 Emotion Recognition from Video using Deep Learning

This project focuses on analyzing facial expressions from videos using advanced deep learning and computer vision techniques. It detects faces, tracks individuals, identifies peak emotional moments (apex frames), and predicts the emotion being expressed.

## 🚀 Features

- 🎯 **Face Detection**: Detects multiple faces in each frame using YOLOv8 or RetinaFace.
- 🧠 **Face Recognition**: Maintains consistent identity tracking using ArcFace embedding.
- 🧭 **Multi-person Tracking**: Uses DeepSORT or ByteTrack to follow each person across frames.
- 🔍 **Apex Frame Detection**: Combines Optical Flow and Eulerian Video Magnification to detect peak emotional expressions.
- 🎞️ **Segment Emotion Analysis**: Identifies the onset-apex-offset segments for emotion analysis.
- 📊 **Emotion Classification**: Uses a ViT (Vision Transformer) model trained for emotion classification.
- 💾 **Data Saving**: Saves apex frame images, emotion predictions, and logs for further analysis.

## 🧠 Techniques & Technologies

- Python, OpenCV, NumPy, Pandas
- TensorFlow / PyTorch
- YOLOv8, RetinaFace, DeepSORT, ByteTrack
- ArcFace (Face embedding)
- Eulerian Video Magnification
- Optical Flow (Farneback / Lucas-Kanade)
- Vision Transformer (ViT) for emotion classification

## 📁 Project Structure

Emotion-Video-Recognition/
├── Code/ # Main source code
│ ├── face_detection.py
│ ├── tracking.py
│ ├── face_recognition.py
│ ├── emotion_classifier.py
│ ├── segment_extraction.py
│ └── main_pipeline.py
├── apex_frames/ # Saved apex frame images
├── data/ # Input videos or test datasets
├── models/ # Pretrained or fine-tuned models
├── output/ # Results, logs, prediction CSVs
└── README.md # Project overview and documentation

pgsql
Sao chép
Chỉnh sửa

## 📌 How It Works

1. **Input**: A video file with human faces.
2. **Detection**: Each frame is processed to detect faces using YOLOv8/RetinaFace.
3. **Tracking & Recognition**: DeepSORT + ArcFace embedding ensures consistent ID for each person across frames.
4. **Segment Extraction**: Using optical flow to determine expression segments (onset to offset).
5. **Apex Frame**: Identified as the frame with the strongest facial motion.
6. **Emotion Prediction**: The apex frame is passed through a deep learning model (ViT) to classify the emotion.
7. **Output**: Final results are saved as images, logs, and prediction files.

## 🛠️ Setup & Usage

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Emotion-Video-Recognition.git
cd Emotion-Video-Recognition

# 2. Create and activate a virtual environment
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the main program
python Code/main_pipeline.py --video ./data/sample_video.mp4
📚 Dataset & Training
Pre-trained on public emotion datasets (e.g. FER+, AffectNet, RAF-DB)

Fine-tuned on expert-labeled apex frames from in-house datasets

📈 Results
Emotion	Accuracy
Happy	91.5%
Sad	88.3%
Angry	90.1%
Surprise	89.7%
Fear	86.0%

Evaluation based on apex frame prediction using ViT.

🙋‍♂️ Author
Nguyen Tien Phuc
📧 Email: ngtphuc0@gmail.com

📄 License
This project is licensed under the MIT License. See LICENSE for details.

Feel free to star ⭐ this repo if you found it helpful!
