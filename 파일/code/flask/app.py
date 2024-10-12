from flask import Flask, render_template, request, jsonify
import torch
from efficientnet import model  # efficientnet.py에서 정의한 model을 가져옴
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# 이미지 전처리 함수
def transform_image(image):
    # Convert 4-channel RGBA to 3-channel RGB if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        # Convert any other mode (e.g., grayscale) to RGB
        image = image.convert('RGB')

    # Apply the necessary preprocessing transformations
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_test(image).unsqueeze(0)  # Add batch dimension

# 예측 함수
def predict(image_tensor):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        print(f"Prediction: {preds.item()}")  # Print the prediction to terminal
    return preds.item()  # Return the predicted class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream)  # Convert uploaded image to PIL Image
    image_tensor = transform_image(image)  # Preprocess the image
    prediction = predict(image_tensor)  # Perform the prediction

    # Return the result based on prediction
    result = '결함 있음' if prediction == 0 else '결함 없음'
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
