# 🚨👷 Real-Time PPE Detection and Tracking for Manufacturing Safety

## 📝 Project Overview

This repository presents a solution for **Real-Time Personal Protective Equipment (PPE) Detection and Tracking** in manufacturing environments. The project leverages **CCTV cameras** 📹 to monitor workers and ensure safety compliance through the detection of essential PPE like helmets 👷, vests 🦺, and head🧑‍🏭. By utilizing cutting-edge technologies like **YOLOv8** 🔥 and **BotSORT** 🧠, along with **custom-developed algorithms** 💡, this system is capable of accurately detecting and **tracking multiple workers' PPE statuses** in dynamic environments with **high speed** ⚡ and **accuracy** 🎯.

## 📚 Research Significance

This project contributes to the **Information Technology and Quality Engineering (ITQE)** field by offering a scalable, real-time solution for industrial safety monitoring. With an extensive dataset of **54,325 annotated images** 📸 of helmets, vests, and heads, the model is designed  by authors to achieve **high accuracy** while ensuring that it works efficiently in real-time environments.

## ⚙️ Technologies Used

- **YOLOv8**: A state-of-the-art object detection model that has been customized for PPE detection tasks 🤖
- **BotSORT**: A robust tracking algorithm used to track multiple objects in real-time, even in crowded environments in workplace🏭🚶‍♂️🚶‍♀️
- **Custom Algorithms**: Proprietary algorithms have been developed to enhance the accuracy and performance of the system, ensuring that it runs **fast** and **efficiently** in real-time 🧠💨
- **Python**: The core programming language used for the implementation and training model🐍.
- **OpenCV**: Used for image processing and video streaming from CCTV cameras 🖥️.
- **PyTorch**: Deep learning framework for model training and inference 🔥.

## 📦 Dataset ( by authors )

The project utilizes a substantial dataset with a total of **54,325 images** (JPG/PNG) containing various instances of:
- **Helmets 👷**
- **Vests 🦺**
- **Heads 🧑‍🏭**

This dataset has been annotated to train and test the model’s ability to detect and classify PPE in a variety of industrial settings, ensuring that the system can generalize well across different scenarios.
### Asking access to the DATASET :  azimjaan21@gmail.com 📧

-----

## 📌 Parameters:

**input_video_path:** Path to the input video (CCTV footage) 🎥

**output_video_path:** Path to save the output video with tracked objects 🎬

-----

## 📑 Academic Contributions
This repository also serves as the basis for an academic research paper 📄 in the field of Information Technology and Quality Engineering (ITQE), addressing critical issues related to worker safety in manufacturing. The paper explores:

The application of real-time computer vision for safety monitoring 🔍.
The challenges and solutions related to tracking multiple objects (workers) across different environments ⚙️.
The potential of AI-driven systems 🤖 to improve workplace safety and compliance ⚖️.

-----

## 🌟 Key Features

- Real-Time Detection: Instantly identifies and tracks PPE usage in live CCTV streams ⏱️.

- Multiple Object Tracking: Handles tracking of multiple workers simultaneously using BotSORT 👷‍♂️👷‍♀️.

- Custom Model & Algorithms: A tailored version of YOLOv8, optimized specifically for PPE detection and enhanced with proprietary algorithms for fast and high-accuracy real-time performance 🧠.

- Scalability: Suitable for large-scale deployment in industrial environments with multiple CCTV cameras 🏗️.

----

## 🏆 Qualitative Results

The system demonstrates strong performance in various industrial settings, including:

Accurate PPE Detection: Precision in identifying helmets, vests, and heads, even in crowded or low-light conditions 🔍.
Tracking Multiple Workers: Seamlessly tracks multiple workers without losing accuracy or performance 🏃‍♂️🏃‍♀️.
Real-Time Monitoring: Provides continuous, real-time monitoring of worker safety ⚠️.
High-Speed Processing: Optimized algorithms allow the system to run in real-time without lag or delay ⚡.

----

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details 📃.

-----

## 🤝 Contributing
We welcome contributions from the community 🌍. To contribute, please follow these steps:

Fork the repository 🍴
Create a new branch for your feature or bug fix 🧑‍💻
Submit a pull request with a clear description of your changes 📝

-----

## 📬 Contact
For any inquiries related to this project or collaboration, feel free to reach out:

Email: azimjaan21@gmail.com 📧

GitHub: azimjaan21 🧑‍💻
