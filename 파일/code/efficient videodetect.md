### 기존 모델에 동영상 분석을 위한 전처리 코드입니다
```python
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image

# 1. EfficientNet 모델 로드
model = EfficientNet.from_name('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 2)  # 이진 분류에 맞게 수정
model.load_state_dict(torch.load("efficientnet_model_binary_class.pt", map_location=torch.device('cpu')))
model.eval()

# 2. 전처리 함수 정의
def preprocess_frame(frame):
    # Resize the frame to 224x224
    frame = cv2.resize(frame, (224, 224))

    # Apply Gaussian Blur to reduce noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert to grayscale and apply histogram equalization to reduce lighting variations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)

    # Convert back to BGR format for consistency
    frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)

    # Convert frame to PIL image for further processing
    pil_img = Image.fromarray(frame)

    # Define the same transforms as used in EfficientNet training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # EfficientNet's normalization
    ])

    # Apply the transformations
    input_tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension
    return input_tensor

# 3. 동영상 분석 및 결과 화면 표시
video_path = "C:/Users/tihd1/OneDrive/바탕 화면/새 폴더 (4)/KakaoTalk_20240930_171403572.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
    exit()

print("동영상 파일이 성공적으로 열렸습니다.")

frame_number = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 전처리
    input_tensor = preprocess_frame(frame)

    # 모델에 프레임 입력
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        prob_damaged = prob[0][0].item()  # 결함 있음 확률
        prob_undamaged = prob[0][1].item()  # 결함 없음 확률

    # 결과 출력 텍스트 설정
    if prob_damaged > prob_undamaged:
        label = f"Damaged ({prob_damaged:.4f})"
        color = (0, 0, 0)
    else:
        label = f"Undamaged ({prob_undamaged:.4f})"
        color = (173, 216, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2  # 글자 크기 확대
    thickness = 4  # 글자 두께
    # 글자 크기 측정
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

    # 텍스트 위치 계산
    text_x = (frame.shape[1] - text_size[0]) // 2  # 중앙 정렬
    text_y = (frame.shape[0] + text_size[1]) // 2

    # 프레임에 텍스트 추가
    cv2.putText(frame, label, (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 프레임을 창에 표시
    cv2.imshow("Defect Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()
print("동영상 분석 완료.")

```
