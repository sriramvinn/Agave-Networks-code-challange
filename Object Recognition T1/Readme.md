# Code Challenge: Object Recognition
## **Instructions**
*1. Use an existing Object Detection model architecture of your choice (e.g., ResNet, MobileNet, etc.) available in a deep learning library (e.g., TensorFlow, PyTorch).*

I have used the PyTorch FasterRCNN with resnet50 as a backbone

*2. Implement a function to perform data augmentation on the given dataset. The augmentation techniques may include rotation, scaling, flipping, or any other relevant techniques.*

Done✅ with albumentations library

*3. Implement a function to load the dataset and split it into training and testing sets. The dataset should contain images and their corresponding labels.*

Spitted with 70, 15, and 15 ratios for training, validation an testing.

*4. Implement a function to train the chosen model on the augmented training dataset. Train for at least 1 epoch. We do not expect any specific accuracy scores on the model, so do not spend too much time in this step.*

Implemented both train and validation functions and ran for 3 epochs

*5. Implement a function to evaluate the trained model's performance on the testing dataset and calculate accuracy scores.*

accuracy is not considered a good metric, when it comes to object detection, but implemented based and how much data it can able to predict bounding boxes, also calculated precision and recall with F1 score, which looks promising.

*6. Consider any potential biases in the dataset (e.g., class imbalance) and handle them appropriately during training and evaluation.*

Considered class imbalance, and explained in the code.

*7. Write clear comments and explanations in your code to demonstrate your understanding of the steps and decisions made.*

✅

*8. Please provide your code along with explanations for each function you implement. Additionally, include a brief explanation of the techniques and strategies you used to handle bias and improve accuracy scores.*

Explained Inside the code
