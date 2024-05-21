import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# 데이터셋 경로
train_dir = 'C:/Users/LEE/PycharmProjects/sw_dev/dataset/car-damage-dataset/data1a/training'
validation_dir = 'C:/Users/LEE/PycharmProjects/sw_dev/dataset/car-damage-dataset/data1a/validation'

# 이미지 데이터 제너레이터 생성
# 데이터 증강을 사용하여 모델의 일반화 성능을 향상시킵니다.
train_datagen = ImageDataGenerator(
    rescale=1./255, # 픽셀 값을 0~1사이로 정규화
    rotation_range=20, # 20도 범위에서 임의로 원본 이미지 회전
    width_shift_range=0.2, # 20% 범위에서 임의로 원본 이미지를 수평으로 이동
    height_shift_range=0.2, # 20% 범위에서 임의로 원본 이미지를 수직으로 이동
    horizontal_flip=True) # 임의로 원본 이미지를 수평으로 뒤집기

test_datagen = ImageDataGenerator(rescale=1./255)

# ImageDataGenerator 의 flow_from_directory 메서드 호출
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='binary')

# ResNet50 모델 사용
base_model = tf.keras.applications.ResNet50(
    weights='imagenet', # ImageNet 데이터셋으로 훈련된 가중치를 사용
    include_top=False, # 네트워크의 최상단에 완전 연결 레이어(즉, 분류를 담당하는 레이어)를 포함하지 않도록 설정
    input_shape=(224, 224, 3))

# ResNet50 모델의 일부 레이어의 가중치를 재학습하도록 설정
# 이를 통해 모델이 특정 데이터셋에 더 잘 적응하도록 합니다.
for layer in base_model.layers[:143]:
    layer.trainable = False
for layer in base_model.layers[143:]:
    layer.trainable = True

model = tf.keras.models.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  Dropout(0.5), # 드롭아웃을 추가하여 과적합을 방지합니다.
  BatchNormalization(), # 배치 정규화를 추가하여 학습을 안정화하고 속도를 높입니다.
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 조기 종료 콜백 추가
# 검증 세트에 대한 성능이 더 이상 향상되지 않을 때 학습을 중단하여 과적합을 방지합니다.
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
      train_generator,
      epochs=10,
      validation_data=validation_generator,
      callbacks=[early_stopping]) # 조기 종료 콜백 사용

loss, accuracy = model.evaluate(validation_generator)

print('Test accuracy :', accuracy)
print('Test loss :', loss)

#Test accuracy : 0.676086962223053
#Test loss : 0.8829212188720703
