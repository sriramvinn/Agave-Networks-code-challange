import json
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import io
import base64
from PIL import Image

# Load the model
def model_fn(model_dir):
    num_classes = 3
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(os.path.join(model_dir,'model.pth'), map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the input image
def transform_image(image_data):
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = image.convert("RGB")
    image = image.resize((200, 200))
    image_tensor = transforms.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

# Handle the prediction request
def predict_fn(input_data, model):
    image = transform_image(input_data)
    with torch.no_grad():
        predictions = model(image)
    # Perform post-processing if required
    return predictions

# Define the input and output formats for the SageMaker endpoint
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        request = json.loads(request_body)
        return request['input']
    raise ValueError(f"Unsupported content type: {request_content_type}")

def output_fn(predictions, response_content_type):
    threshold = 0.7
    if response_content_type == 'application/json':
        predicted_boxes = predictions[0]['boxes'].data
        predicted_labels = predictions[0]['labels'].data
        scores = predictions[0]['scores'].data
        predicted_labels = predicted_labels[scores >= threshold]
        predicted_boxes = predicted_boxes[scores >= threshold]
        response = {'labels': predicted_labels.tolist(), 'boxes':predicted_boxes.tolist() }
        return json.dumps(response)
    raise ValueError(f"Unsupported content type: {response_content_type}")

