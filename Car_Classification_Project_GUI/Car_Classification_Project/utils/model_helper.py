import tensorflow as tf
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet


@st.cache_resource
def load_custom_model(model_path):
    """..."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None


def smart_preprocess(image, model_name):
    """
    ...
    """
    if image.mode != "RGB":
        image = image.convert("RGB")


    if "InceptionV3" in model_name:
        target_size = (299, 299)
        preprocessor = preprocess_inception
    elif "ResNet50" in model_name:
        target_size = (224, 224)
        preprocessor = preprocess_resnet
    elif "EfficientNetB4" in model_name:
        target_size = (384, 384)
        preprocessor = preprocess_efficientnet
    else:
        target_size = (224, 224)
        preprocessor = preprocess_resnet

    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocessor(img_array)

    return img_array


def get_last_conv_layer(model):
    """Convolution"""
    for layer in reversed(model.layers):
        try:

            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
            elif hasattr(layer, 'output'):
                output_shape = layer.output.shape
            else:
                continue


            if isinstance(output_shape, tuple) and len(output_shape) == 4:
                return layer.name

        except (AttributeError, ValueError):
            continue

    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """(Heatmap)"""


    inputs = model.inputs

    grad_model = tf.keras.models.Model(
        inputs=inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)


        if isinstance(preds, list):
            preds = preds[0]

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)


    if grads is None:
        raise ValueError("Gradient calculation failed. The layer might be disconnected.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]


    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """..."""
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    original_img = np.array(original_img)
    jet = cv2.resize(jet, (original_img.shape[1], original_img.shape[0]))

    superimposed_img = jet * alpha + original_img * (1 - alpha)
    return np.uint8(superimposed_img)