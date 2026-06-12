# Enhanced-Facial-Emotion-Recognition-Using-Transfer-Learning-with-ResNet152
📌 Overview
This repository presents a Facial Emotion Recognition (FER) system built using ResNet152, leveraging transfer learning, extensive preprocessing, and data augmentation to achieve high accuracy in real-time emotion classification. The model identifies six emotions: Happy, Angry, Sad, Neutral, Surprise, and Ahegao.

Deep learning—especially CNN-based architectures—has transformed FER by automatically learning hierarchical features, outperforming traditional hand-engineered approaches. ResNet152's deep residual architecture allows the model to capture subtle facial variations that are often missed by conventional methods.

🔥 Our final model achieved an accuracy of 84%, demonstrating strong generalization across images sourced from diverse real-world environments.

🔍 Abstract
Facial expression-based emotion recognition has gained significant traction across domains such as camera systems, mental health assessment, and human–computer interaction. This project introduces a FER framework built using the ResNet152 architecture, enhanced by preprocessing, data augmentation, and transfer learning.

While classical machine learning approaches struggle to jointly optimize feature extraction and classification, modern CNN models excel at learning complex visual patterns. By fine-tuning ResNet152—pre-trained on ImageNet—we efficiently adapt its learned representations to the task of emotion classification.

Extensive testing on a custom dataset confirms the effectiveness of our method, with ResNet152 achieving 84% accuracy. Real-world experiments with randomly sampled images further validate its robustness and precision. This work presents a reliable FER pipeline that advances beyond traditional techniques through the power of deep residual learning.

This research was presented at the 2024 International Conference on Intelligent Computing and Sustainable Innovations in Technology (IC-SIT).

🧠 Introduction
Facial expressions play a crucial role in human communication, conveying emotional and psychological states in day-to-day interactions. Recognizing emotions from facial cues has major applications in:

Human–Computer Interaction
Affective Computing
Behavioral and Mental Health Analysis
Intelligent camera systems
Although deep learning has significantly advanced FER, challenges such as subtle emotion differences, varying lighting, pose variations, and natural expression diversity remain. Traditional feature-engineered models often fail due to limited generalization.

ResNet152, known for its deep architecture and skip connections, offers powerful representational capacity ideal for complex visual tasks. Transfer learning further enhances performance by allowing us to reuse learned features from large-scale datasets.

This work was presented at the 2024 International Conference on Intelligent Computing and Sustainable Innovations in Technology (IC-SIT), highlighting the latest advancements in applying deep learning techniques to real-world problems.

🧪 Methodology
1. Data Collection
The dataset includes six emotion categories:

Images were collected from:

YouTube videos
Social media platforms (Facebook, Instagram)
Public datasets (AffectNet, IMDB)
All images were cropped to contain only faces.

Note: Due to privacy and licensing restrictions, the raw dataset is not publicly available. However, the preprocessing steps and augmentation techniques are fully documented to enable replication using publicly available datasets.

2. Data Preprocessing
Several preprocessing steps ensured the dataset was model-ready:

Dataset Organization: Images were structured into separate directories for each emotion category.

Normalization: Pixel values were scaled to [0, 1] for stable and faster convergence.

Data Augmentation: Using ImageDataGenerator, we applied:

±30° rotation
Zoom variations (0.2)
Width/height shifts
Horizontal flips
Brightness adjustments
These transformations increase dataset diversity and reduce overfitting by simulating real-world conditions such as lighting changes, pose variations, and facial symmetry differences.

🏗 Model Architecture
Why ResNet152?
ResNet152 is one of the deepest models in the ResNet family and excels at extracting fine-grained features due to:

Residual Blocks: Allow gradients to flow efficiently using skip connections
Bottleneck Layers: Reduce computational cost while improving depth
Deep Feature Extraction: Captures subtle muscle movements and micro-expressions
Transfer Learning Support: Pretrained weights on ImageNet enable fast adaptation
Residual Learning Concept
Instead of learning direct mappings, ResNet learns residual functions, solving the vanishing gradient problem and enabling networks with over 150 layers to train effectively.

ResNet152 Architecture

📈 Training Process
Loss Curve
Training Loss Curve

Accuracy Curve
Training Accuracy Curve

🎯 Results
Overall Performance
Accuracy Achieved: 84% on the custom dataset
Model Performance: Outperformed baseline CNN architectures
Generalization: Successfully recognized emotions in randomly sourced images
Robustness: Stable predictions across lighting and pose differences
Classification Report
Class	Precision	Recall	F1-Score	Support
Ahegao	0.99	0.94	0.97	240
Angry	0.81	0.79	0.80	253
Happy	0.94	0.95	0.95	725
Neutral	0.77	0.77	0.77	811
Sad	0.77	0.80	0.79	817
Surprise	0.91	0.84	0.87	245
Accuracy			0.84	3091
Macro Avg	0.87	0.85	0.86	3091
Wgt Avg	0.84	0.84	0.84	3091
Confusion Matrix
Confusion Matrix

🧾 Conclusion
This project delivers a robust FER system that integrates:

Advanced preprocessing
Data augmentation
Transfer learning
Deep residual learning with ResNet152
The approach significantly improves accuracy and real-world performance compared to traditional CNN-based emotion recognition systems. The model's success highlights the strength of deep learning for understanding human emotions in practical applications.

📁 Directory Structure
Enhanced-Facial-Emotion-Recognition-Using-Transfer-Learning-with-ResNet152/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── run_app.bat             # Windows batch file to run the app
├── README.md               # This file
├── LICENSE                 # License file
├── .gitignore              # Git ignore file
├── Model/
│   └── emotion_model.keras     # Trained model file (ignored in Git)
├── Notebook/
│   └── Enhanced Facial Emotion Recognition using Transfer Learning with Resnet152.ipynb
├── images/
│   ├── Accuracy Curve.png
│   ├── Loss Curve.png
│   ├── Confusion Matrix.png
│   ├── Model Architecture.png
│   └── demo.gif
🚀 Getting Started
Prerequisites
Python 3.7+
TensorFlow 2.15.0
Streamlit 1.28.0
OpenCV 4.8.1.78
NumPy 1.24.3
Pillow 10.0.1
Installation
# Clone the repository
git clone https://github.com/pSahoo-456/Enhanced-Facial-Emotion-Recognition-Using-Transfer-Learning-with-ResNet152.git
cd Enhanced-Facial-Emotion-Recognition-Using-Transfer-Learning-with-ResNet152

# Install dependencies
pip install -r requirements.txt
Model File
The trained model file (emotion_model.keras) is intentionally excluded from this repository due to its large size (approximately 690MB). To run the application:

Navigate to the project root directory
Download the model file from the release section or retrain it using the provided Jupyter notebook
Place the model file in the Model/ directory
Alternatively, you can retrain the model by running the Jupyter notebook provided in the Notebook/ directory.

Running the Application
# Run the Streamlit app
streamlit run app.py

📚 References
[1] Phutela, Deepika. "The importance of non-verbal communication." IUP Journal of Soft Skills 9.4 (2015): 43.

[2] Ioannou, Spiros V., et al. "Emotion recognition through facial expression analysis based on a neurofuzzy network." Neural Networks 18.4 (2005): 423-435.

[3] Baecker, Ronald, and William Buxton. "Readings in human-computer interaction: A multidisciplinary approach." (1987).

[4] Canedo, Daniel, and Antonio JR Neves. "Facial expression recognition using computer vision: A systematic review." Applied Sciences 9.21 (2019): 4678.

[5] Pantic, Maja, and Leon J. M. Rothkrantz. "Automatic analysis of facial expressions: The state of the art." IEEE Transactions on pattern analysis and machine intelligence 22.12 (2000): 1424-1445.

[6] Fei, Zixiang, et al. "Deep convolution network based emotion analysis towards mental health care." Neurocomputing 388 (2020): 212-227.

[7] Mollahosseini, Ali, David Chan, and Mohammad H. Mahoor. "Going deeper in facial expression recognition using deep neural networks." 2016 IEEE Winter conference on applications of computer vision (WACV). IEEE, 2016.

[8] Dhall, Abhinav, et al. "Video and image based emotion recognition challenges in the wild: Emotiw 2015." Proceedings of the 2015 ACM on international conference on multimodal interaction. 2015.

[9] Khan, Amjad Rehman. "Facial emotion recognition using conventional machine learning and deep learning methods: current achievements, analysis and remaining challenges." Information 13.6 (2022): 268.

[10] Giannopoulos, Panagiotis, Isidoros Perikos, and Ioannis Hatzilygeroudis. "Deep learning approaches for facial emotion recognition: A case study on FER-2013." Advances in hybridization of intelligent methods: Models, systems and applications (2018): 1-16.

📚 Citation
If you use this work in your research, please cite our paper:

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Special thanks to all the researchers whose work is cited in our references section and my mentor for their valuable guidance and support.
