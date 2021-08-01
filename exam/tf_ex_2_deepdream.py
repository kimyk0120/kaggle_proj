"""
    Deep dream
    https://www.tensorflow.org/tutorials/generative/deepdream?hl=ko
"""

import tensorflow as tf
import numpy as np
import PIL.Image
import time
import functools
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl

import tensorflow_hub as hub

from tensorflow.keras.preprocessing import image

print("prcs start")


# 이미지를 내려받아 넘파이 배열로 변환합니다.
def download(url, max_dim=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


# 이미지를 정규화합니다.
def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


# 이미지를 출력합니다.
def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))


url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

# 이미지의 크기를 줄여 작업이 더 용이하도록 만듭니다.
original_img = download(url, max_dim=500)
plt.imshow(original_img)
plt.show()

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

base_model.summary()

# 선택한 층들의 활성화값을 최대화합니다.
names = ['mixed3', 'mixed5']
# names = ['mixed1', ]
layers = [base_model.get_layer(name).output for name in names]

# 특성 추출 모델을 만듭니다.
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
print('&'*1000+'\n')
dream_model.summary()


def calc_loss(img, model):
    # 이미지를 순전파시켜 모델의 활성화값을 얻습니다.
    # 이미지의 배치(batch) 크기를 1로 만듭니다.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # `img`에 대한 그래디언트가 필요합니다.
                # `GradientTape`은 기본적으로 오직 `tf.Variable`만 주시합니다.
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # 입력 이미지의 각 픽셀에 대한 손실 함수의 그래디언트를 계산합니다.
            gradients = tape.gradient(loss, img)

            # 그래디언트를 정규화합니다.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # 경사상승법을 이용해 "손실" 최대화함으로써 입력 이미지가 선택한 층들을 보다 더 "흥분" 시킬 수 있도록 합니다.
            # (그래디언트와 이미지의 차원이 동일하므로) 그래디언트를 이미지에 직접 더함으로써 이미지를 업데이트할 수 있습니다.
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


deepdream = DeepDream(dream_model)

# mauin custom loop
def run_deep_dream_simple(img, steps=100, step_size=0.01):
    # 이미지를 모델에 순전파하기 위해 uint8 형식으로 변환합니다.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size))

        # display.clear_output(wait=True)
        # show(deprocess(img))
        print("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    plt.imshow(result)
    plt.show()
    # display.clear_output(wait=True)
    return result


dream_img = run_deep_dream_simple(img=original_img,
                                  steps=100, step_size=0.01)


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


file_name = 'deepdreamed-image.png'
tensor_to_image(dream_img).save(file_name)


print("prcs fin")

if __name__ == "__main__":
    pass
