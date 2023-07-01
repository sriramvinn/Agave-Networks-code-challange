from flask import Flask, request, jsonify, Response, render_template
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import io
import base64
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torchvision



app = Flask(__name__)

# Load the saved Faster R-CNN model
classes = ['bg', 'dog', 'cat']

def create_model(num_classes):

    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = create_model(len(classes))
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

def draw_bounding_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
      
        draw.rectangle(box.tolist(), outline='red', width=2)
        draw.text((box[0], box[1]), classes[label], fill='red')
    return image

# Define the API endpoint
@app.route('/', methods=['POST', 'GET'])
def index():
    error = None
    if request.method == 'POST':
        # Get the uploaded file from the POST request
        image_file = request.files['image']
        image = Image.open(image_file)
        
        # Preprocess the image
        image = image.convert("RGB")
        image = image.resize((200, 200))
        image_tensor = transforms.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            predictions = model(image_tensor)

        # Extract the bounding box coordinates from the predictions
        predicted_boxes = predictions[0]['boxes'].data
        predicted_labels = predictions[0]['labels'].data
        scores = predictions[0]['scores'].data
        predicted_labels = predicted_labels[scores >= 0.5]
        predicted_boxes = predicted_boxes[scores >= 0.5]

        # Draw bounding boxes on the image

        if len(predicted_labels)!= 0:
            # Visualize the image with predicted bounding boxes
            annotated_image = draw_bounding_boxes(image.copy(), predicted_boxes,predicted_labels)

            # Return the annotated image as a responsebuffered = io.BytesIO()
            buffered = io.BytesIO()
            annotated_image.save(buffered, format="PNG")
            annotated_image_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Return the annotated image as a response
            return render_template('predict.html', image_data=annotated_image_str)
        else:
            error = 'Face verification failed.'

    return render_template('index.html', error = error)


# Run the Flask app
if __name__ == '__main__':
    app.run()
