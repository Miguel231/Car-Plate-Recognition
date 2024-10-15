# Car Plate Recognition

[Character Dataset](https://drive.google.com/drive/folders/1qMIVP5-665d0stgs0FKR6Z5DrRBZyl1J?usp=sharing)

## 1. Introduction  

In recent decades, detection and recognition systems have been widely applied across diverse fields, from medical diagnostics to security and surveillance. Among these applications, machine learning (ML)[1] and deep learning (DL)[2] algorithms have become essential tools for object detection in images and videos. One specific area of interest is the automatic detection and recognition of vehicle license plates. 

This project aims to develop a system that can accurately detect, segment, and identify license plates from images in a dataset. The primary motivation is the increasing demand for automated systems in traffic management, security, and law enforcement, where manual license plate identification is time-consuming and prone to errors. 

The main challenges addressed by this project include the variability in plate appearances due to factors like lighting conditions, camera angles, and image quality. To face these issues some methods that can be used include traditional ML techniques like Support Vector Machines (SVMs) for feature-based recognition, as well as DL approaches, such as Convolutional Neural Networks (CNNs). 

The aim of this project is to create an efficient system capable of handling diverse real-world scenarios for license plate recognition, from video frames to images. 
 
GitHub has been used as a platform to store and manage the code and other files of the project. The reason has been because it allows collaboration between multiple people in an efficient way. And it is a place where progress is stored online without fear of losing it. 

## Goals   

The primary goal of this project is to accurately detect car license plates and identify the characters (numbers and letters) on them. To achieve this, several specific objectives and tasks must be completed, such as: 

- Car License Plate Detection and Character Identification: The core aim is to develop a system that can reliably detect the presence of license plates in images or videos and accurately extract the letters and numbers from them. 

- To enhance the effectiveness of this goal, the following supporting tasks are necessary: 

- Data Augmentation: Expanding the dataset through techniques such as image rotation, scaling, and brightness adjustments to improve the system’s ability to generalize across various conditions. 

- Creation of Dataset for Letters and Numbers: Curating a labelled dataset specifically focused on the different letters and numbers that appear on license plates, which will be essential for training the model. 

- In addition to the main goal, there are additional aspirations to further improve the system: 

- Video-Based Data Analysis: Incorporating video input to complement static images, allowing the system to process dynamic scenarios and detect license plates in motion. 

- Country Detection: Enhancing the model to identify the country of origin based on license plate formats and styles. 

- Car Model Identification: Adding functionality to recognize the car’s make or model using information inferred from the license plate and additional visual clues. 

- Dataset Query Functionality: Developing a feature that allows users to input a license plate number and check whether it already exists in the dataset. If it does, the model will retrieve relevant information. 
