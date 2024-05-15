import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터셋 경로
train_dir = 'C:/Users/lhw0841/PycharmProjects/pythonProject/sw_dev/dataset/car-damage-dataset/data1a/training'
validation_dir = 'C:/Users/lhw0841/PycharmProjects/pythonProject/sw_dev/dataset/car-damage-dataset/data1a/validation'

# 이미지 데이터 제너레이터 생성
train_datagen = ImageDataGenerator(rescale=1./255) # ImageDataGenerator 클래스의 인스턴스 생성.
test_datagen = ImageDataGenerator(rescale=1./255) # rescale=1./255 는 이미지의 픽셀 값을 0~1사이로 정규화.

# ImageDataGenerator 의 flow_from_directory 메서드 호출
# 지정된 디렉토리에서 이미지를 로드하고, 이미지 데이터를 전처리하며, 배치 단위로 이미지와 레이블을 제공
train_generator = train_datagen.flow_from_directory(
        train_dir, # 경로 설정
        target_size=(224, 224), # 이미지 크기 지정
        batch_size=20, # 한번에 학습하는 개수
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='binary')

print(len(train_generator),len(validation_generator)) ## 데이터 증강 적용중 , 184,46 나옴.

# ResNet50 모델 사용
base_model = tf.keras.applications.ResNet50(weights='imagenet', # ImageNet 데이터셋으로 훈련된 가중치를 사용
                                            include_top=False, # 네트워크의 최상단에 완전 연결 레이어(즉, 분류를 담당하는 레이어)를 포함하지 않도록 설정
                                            input_shape=(224, 224, 3)) # 네트워크에 입력되는 이미지 텐서의 크기를 지정

# base_model.trainable = False: 이 줄은 ResNet50 모델의 가중치를 고정합니다. 즉, 훈련하는 동안 이 가중치가 업데이트되지 않도록 설정
base_model.trainable = False

model = tf.keras.models.Sequential([ # Sequential : 순서대로 레이어를 쌓는 모델
  base_model, # ResNet50
  tf.keras.layers.GlobalAveragePooling2D(), # average
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# model 구조 출력
model.summary()

history = model.fit(
      train_generator,
      steps_per_epoch=184, # 100 으로 설정시, 한 epoch에 100개의 batch를 시도. batch = 20 일시 총 2000개 필요. train 에는 920개가 있으므로 총 1840개 처리후 입력이 부족해짐.
      epochs=10,
      validation_data=validation_generator,
      validation_steps=46)
# 데이터 증강을 사용시에는 len(train_generator), len(validation_generator) 를 사용해야함. 기존의 데이터셋 입력수와 다를것이기 때문
