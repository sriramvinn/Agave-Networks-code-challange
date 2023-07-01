# **Cloud Inference Endpoint**

For this task I have used previously trained model which include first two steps of this task, and deployed it in Amazon saze maker to create an api endpoit which is configured to take imahe in json format and returns bounding boxes with labels in json format.

I have also used Flask webapp to deploy it locally, we can use this to create docker container image, and use the EC2 instance to pull the docker image from ecr to ec2 instance. since this task menctions not to consider scalability or high loads, I have utilised sagemaker to deploy already trained model.

## **Instructions**
*-Model Training: Utilize cloud-based machine learning platforms like TensorFlow or PyTorch in conjunction with pre-trained models, such as YOLO (You Only Look Once) or SSD (Single Shot MultiBox Detector), to train an object detection model. Fine-tune the pre-trained model on the provided labeled dataset to adapt it to the specific objects of interest.*
I would reqire furtur details to redo this step, I am confused with the statement "cloud-based machine learning platforms like TensorFlow or PyTorch", plese let me know if this task meant to train the model in cloud environment such as AWS. Since the task 1 done it in same way, using pytorch pertraoined model and finetuning it, I would like to to continue with final step(Deployment).

*- Evaluation: Assess the performance of the trained object detection model using evaluation metrics such as mean Average Precision (mAP) or Intersection over Union (IoU). Analyze the results to gain insights into the model's accuracy, precision, and recall for different object classes.*
While evaluating the model on test dataset, I have calculated the IOU of the bounding boxes, and utilised it to calculate precision, recall and f1 score for each class

*- Deployment: Deploy the trained object detection model as an inference endpoint that can be accessed through our API. Configure the endpoint to handle image input and provide bounding box predictions for detected objects. Consider the endpoint is to be part of an MVP, so do not consider scalability nor high loads.*
I haves attched the sagmaker jupyeter notbook instance to deploy the model to get the api, further more we can implement the aws lambda function which can be triggred by when input is posted to api.
