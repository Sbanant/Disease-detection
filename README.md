# Disease-detection 
This project is able to do two things, first of all, it consists of Disease Detection by entering symptoms and also skin disease detection via images.

# Disease detection via symptoms 

This part of the project will take input in the form of symptoms the user is experiencing. when the code is run the user will be prompted 
"Enter your symptoms (or 'exit' to quit)", after this when the user enters the system he/she is experiencing the model will then compare the entered 
symptoms with the data providing the input as the disease the user has.

# Skin disease detection 

This part of the project will take input in the form of images through a camera, when the code runs the camera will switch on and capture the image in
front of it, this process will take 5 seconds which can be changed in the code. The trained model will then process the captured image using the TensorFlow 
library, according to this an output will be provided along with a confidence score telling which skin disease the user has. 

# Optimizing Skin Disease Prediction Model - Achieving Higher Accuracy
Introduction 
This section outlines the various methodologies employed to enhance the accuracy of the skin disease prediction model, which utilizes a Multinomial Naive Bayes algorithm. The model identifies and classifies skin conditions such as Acne and Rosacea, Eczema, Cellulitis, Melanoma, and others, based on user-uploaded images.

Feature Engineering
One of the critical aspects of improving accuracy was extensive feature engineering. Since skin diseases present with diverse patterns, textures, and colors, effective image preprocessing was crucial:

Image Scaling and Normalization: Images were resized to 224x224 pixels for consistent input to the model. Additionally, each pixel value was normalized to the range [0,1] to reduce the variance between images.

Data Augmentation: The dataset was augmented to increase variability and prevent overfitting. Techniques such as:

Rotation,
Zooming,
Horizontal flipping, and
Brightness adjustments
were employed to simulate real-world variations in the appearance of skin diseases.


Color Histogram Analysis: Color histograms were used to extract relevant color features. Diseases like rosacea and melanoma often have distinctive color patterns that were leveraged for better classification.
Model Selection
Although Multinomial Naive Bayes was initially chosen for its simplicity and interpretability, it was essential to optimize its usage for image data. Several approaches were taken to improve classification:

Multimodal Feature Representation: The Naive Bayes model was enhanced by combining pixel-level features with domain-specific knowledge, such as texture patterns often seen in diseases like eczema and herpes. This combination of feature vectors yielded better disease discrimination.

Grid Search for Hyperparameter Tuning: A thorough grid search was conducted to identify optimal hyperparameters such as the smoothing parameter (alpha). Fine-tuning this parameter significantly increased the model's generalization capability across different skin disease categories.


Ensemble Learning: To further increase the robustness of the predictions, we tested ensemble approaches. By integrating multiple models, including Convolutional Neural Networks (CNNs) with Naive Bayes, we were able to create a hybrid model that balanced the strengths of both methodologies, leading to improved overall performance, especially in challenging categories like Melanoma and Systematic diseases.
Evaluation Metrics
To evaluate the model’s performance, precision, recall, and F1-scores were calculated for each skin disease class. Focus was put on imbalanced classes like "Light Disease," where specialized sampling techniques, such as SMOTE (Synthetic Minority Over-sampling Technique), were used to address bias.


Conclusion
Through advanced image preprocessing, data augmentation, model tuning, and multimodal feature representation, we were able to significantly improve the accuracy of the model. The combination of these techniques allowed the classifier to excel in both common and rare skin conditions. Future improvements may involve leveraging more advanced deep-learning models and exploring fine-grained disease subtypes.



# Note

All the databases attached here are picked up from websites online and haven't been formed personally by me.

Also for disease detection via symptoms the database attached needs to be downloaded and then the path and column headings for the same replaced in the code
as mentioned in the code itself.

