# **Cloud Inference Endpoint**

For this task, I have used a previously trained model which includes the first two steps of this task, and deployed it in Amazon sagemaker to create an API endpoint that is configured to take an image in JSON format and return bounding boxes with labels in JSON format.

I have also used the Flask web app to deploy it locally, we can use this to create a docker container image and use the EC2 instance to pull the docker image from ECR to the EC2 instance. since this task mentions not to consider scalability or high loads, I have utilized sagemaker to deploy an already trained model.

## **Instructions**
*-Model Training: Utilize cloud-based machine learning platforms like TensorFlow or PyTorch in conjunction with pre-trained models, such as YOLO (You Only Look Once) or SSD (Single Shot MultiBox Detector), to train an object detection model. Fine-tune the pre-trained model on the provided labeled dataset to adapt it to the specific objects of interest.*

In the Object Detection task, I have trained the model by fine tuning PyTorch pretrained model FasterRCNN ResNet50 for object detection, for this task I am utilising that model for deployment.

*- Evaluation: Assess the performance of the trained object detection model using evaluation metrics such as mean Average Precision (mAP) or Intersection over Union (IoU). Analyze the results to gain insights into the model's accuracy, precision, and recall for different object classes.*

While evaluating the model on the test dataset, I calculated the IOU of the bounding boxes and utilized it to calculate precision, recall, and f1 score for each class

*- Deployment: Deploy the trained object detection model as an inference endpoint that can be accessed through our API. Configure the endpoint to handle image input and provide bounding box predictions for detected objects. Consider the endpoint to be part of an MVP, so do not consider scalability or high loads.*

I have attached the Sagemaker jupyter notebook instance to deploy the model to get the API. Furthermore, we can implement the AWS lambda function or serverless framework which can be triggered when input is posted to API.

For the purpose of local deployment I have Implimented Flask, UI and results are attched in the Flask implimentation folder
