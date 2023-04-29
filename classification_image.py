import tensorflow as tf
import numpy as np
from PIL import Image

# 모델 불러오기
model = tf.keras.models.load_model('model.h5')

# 입력 이미지 전처리 함수
def preprocess_image(image_path):
    # 이미지 불러오기
    image = Image.open(image_path)

    # 이미지 크기 조정
    image = image.resize((500, 500))

    # 이미지 배열로 변환
    image_array = np.array(image)

    # 이미지 배열 형태 변환 (1, width, height, channels)
    image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1], image_array.shape[2])

    # 픽셀 값 범위 조정 (0~1)
    image_array = image_array / 255.

    return image_array

# 이미지 분류 함수
def classify_image(image_path):
    # 이미지 전처리
    image_array = preprocess_image(image_path)

    # 모델에 입력하여 결과 예측
    result = model.predict(image_array)

    # 분류 결과 해석 및 반환
    if result[0][0] > result[0][1]:
        return 'Plastic'
    else:
        return 'Non-Plastic'

