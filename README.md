# Lane Detection using CNN Algorithm  

## 📌 Description  

Self-initiated Project | Python, OpenCV, TensorFlow

This project implements a **Convolutional Neural Network (CNN)** for lane detection in driving scenarios, basically It processes a given video input, detects the lane, and outputs a processed video with marked lane lines. This technology is useful for autonomous vehicles and driver-assistance systems.
through this project,one can learn how the autonomous cars detects the lane and avoids collision with other vehicles

- Achieved **>90% training accuracy** with steadily decreasing loss over epochs.

- Implemented lane segmentation pipeline using **custom CNN architecture**, trained on annotated driving footage.

- Detected lane lines with high **temporal consistency** and **pixel-level accuracy** in diverse scenarios.

- Delivered real-time performance with smooth boundary overlays at **20+ FPS** on test video.

- Evaluated using MSE, pixel accuracy, model loss and visual validation for real-world usability.



![Image](https://github.com/user-attachments/assets/fbc86c4e-b727-4b34-9ece-a4f0bd8c7bd4)

## Features ✨  
- Uses **CNN-based lane detection** for improved accuracy  
- Supports **video input processing** with real-time lane detection  
- Outputs a processed video with detected lanes overlayed
- **Pre-trained model included** for quick inference  

## Tech Stack 💻
- **Programming Language:** Python  
- **Machine Learning Framework:** TensorFlow/Keras  
- **Other Technologies:** OpenCV, NumPy, Matplotlib  

## Installation  
Follow these steps to set up and run the project:  

1. **Clone the Repository**  
   ```bash
   git clone 'url of the repository'
   cd lane-detection-cnn
   cd src
   ```  

2. **Create a Virtual Environment (Optional but Recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # For MacOS/Linux
   venv\Scripts\activate      # For Windows
   ```  

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```  

## Usage  
1. **Provide a Video Input**  
   - Place your video file in the project directory (e.g., `input_video.mp4`).  

2. **Run the Lane Detection Script**  
   ```bash
   python draw_detected_lanes.py
   ```  

3. **View the Output**  
   - The processed video with detected lanes will be saved as `output_video.mp4`.  


## ⚙️ Configuration  
- Modify **fully_conv_NN.py** to fine-tune the CNN model.  
- Adjust **draw_detected_lanes.py** to change visualization settings.  

## Contribution  
Contributions are welcome! To contribute:  
1. Fork the repository  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m "Add feature"`)  
4. Push to the branch (`git push origin feature-name`)  
5. Create a Pull Request  

## License  
This project is licensed under the **MIT License**. Feel free to use and modify it.  

## Screenshots & Demos  
- Training history ![Image](https://github.com/user-attachments/assets/9f6077f6-985f-4df4-8414-1065403e4edc)
---  

Developed with ❤️ by [Ashiq Muhammed M]  
