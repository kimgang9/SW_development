### Efficientnet모델

```python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, f1_score

# 데이터셋 경로
train_data_dir = 'C:/Users/tihd1/Downloads/car-part-dataset/car-part-dataset/train'
test_data_dir = 'C:/Users/tihd1/Downloads/car-part-dataset/car-part-dataset/test'


# 데이터 전처리
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 학습 데이터 로더
train_data = datasets.ImageFolder(root=train_data_dir, transform=transform_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 테스트 데이터 로더
test_data = datasets.ImageFolder(root=test_data_dir, transform=transform_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# EfficientNet 모델 로드
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(train_data.classes))

# 손실 함수 및 최적화
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# CPU 사용
device = torch.device('cpu')
model = model.to(device)

# 학습 과정
for epoch in range(10):  # 10 에포크만 예시로 설정
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader)}')

# 테스트 평가
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 정확도 및 F1 Score 계산
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f'Test Accuracy: {acc}, Test F1 Score: {f1}')

```
