# Lane Detection using CNN Algorithm  

## 📌 Description  
This project implements a **Convolutional Neural Network (CNN)** for lane detection in driving scenarios, basically It processes a given video input, detects the lane, and outputs a processed video with marked lane lines. This technology is useful for autonomous vehicles and driver-assistance systems.
through this project,one can learn how the autonomous cars detects the lane and avoids collision with other vehicles

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
- Training history ![Image](https://github.com/user-attachments/assets/f79d83e9-e4ba-4359-bef7-219e7712e5cd)   

---  

Developed with ❤️ by [Ashiq Muhammed M]  
