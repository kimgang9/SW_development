```python
import os
import zipfile
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# 업로드된 ZIP 파일 경로
zip_file_path = r'C:\Users\tihd1\OneDrive\문서\카카오톡 받은 파일\car-damage-dataset.zip'
extracted_dir_path = r'C:\Users\tihd1\OneDrive\문서\카카오톡 받은 파일\car-damage-dataset'

# ZIP 파일 경로 확인
if not os.path.exists(zip_file_path):
    print(f"Error: The ZIP file {zip_file_path} does not exist.")
else:
    print(f"The ZIP file {zip_file_path} exists.")

    # ZIP 파일 압축 해제
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir_path)
        print(f"ZIP file extracted to {extracted_dir_path}")
    except zipfile.BadZipFile:
        print(f"Error: The file {zip_file_path} is not a valid ZIP file.")
    except FileNotFoundError as e:
        print(f"Error: {e}")

    # 추출된 데이터 경로 설정
    data_dir = os.path.join(extracted_dir_path, 'car-damage-dataset/data1a/training')

    # 경로 디버깅을 위한 코드
    print(f"Checking if the directory exists: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"Error: The directory {data_dir} does not exist.")
    else:
        print(f"The directory {data_dir} exists.")

        # 데이터 로딩 및 전처리
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )

        print("Train and validation data generators are created.")

        # 모델 구축
        model = Sequential([
            Input(shape=(224, 224, 3)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])

        # 모델 컴파일
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # 모델 요약
        model.summary()

        # 모델 훈련
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=10
        )

        # 모델 평가
        loss, accuracy = model.evaluate(validation_generator)
        print(f"Validation Loss: {loss}")
        print(f"Validation Accuracy: {accuracy}")


```
