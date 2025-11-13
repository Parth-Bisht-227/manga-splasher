import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# -------------------------------
# Page Configuration & Styles
# -------------------------------
st.set_page_config(
    page_title = "Manga Colorization Demo",
    page_icon = "🎨",
    layout = "centered"
)
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .css-18e3th9 {background-color: #18182F;}
        .stButton>button {
            background-color: #F63366;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Generator Architecture (U-Net)
# -------------------------------
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def build_generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])  # L channel only

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),                        # (bs, 64, 64, 128)
        downsample(256, 4),                        # (bs, 32, 32, 256)
        downsample(512, 4),                        # (bs, 16, 16, 512)
        downsample(512, 4),                        # (bs, 8, 8, 512)
        downsample(512, 4),                        # (bs, 4, 4, 512)
        downsample(512, 4),                        # (bs, 2, 2, 512)
        downsample(512, 4),                        # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),      # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),      # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),      # (bs, 8, 8, 1024)
        upsample(512, 4),                          # (bs, 16, 16, 1024)
        upsample(256, 4),                          # (bs, 32, 32, 512)
        upsample(128, 4),                          # (bs, 64, 64, 256)
        upsample(64, 4),                           # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(2, 4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')  # (bs, 256, 256, 2)

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


# -------------------------------
# Download Model Checkpoints
# -------------------------------

checkpoint_files = [
    {
        "url": "https://drive.google.com/uc?id=1bW6DrmRR1eKliq3_0Zdz1ujft2gj5_2E",
        "output": "./checkpoints/ckpt_epoch30-3.data-00000-of-00001"
    },
    # Add similar dicts here for .index and checkpoint files if needed
]

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

def download_checkpoints():
    for entry in checkpoint_files:
        if not os.path.exists(entry["output"]):
            gdown.download(entry["url"], entry["output"], quiet=False)

download_checkpoints()

# -------------------------------
# Load Generator from Checkpoint
# -------------------------------

generator = build_generator()
checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint.restore(tf.train.latest_checkpoint('./checkpoints')).expect_partial()

# -------------------------------
# Streamlit UI
# -------------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #F63366;'>🎨 Manga Colorization Demo</h1>
    <p style='text-align: center; font-size:20px'>
      Upload your black-and-white manga panel below to see it magically colored!
    </p>
    """, unsafe_allow_html=True
)
st.markdown("---")

with st.expander("About This App"):
    st.info("This demo uses a U-Net GAN trained for manga panel colorization. Powered by TensorFlow, OpenCV, and Streamlit.")

st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a black-and-white manga panel", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image.resize((256, 256)))

    # Convert RGB to LAB
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    L_channel = lab_image[:, :, 0]
    L_norm = (L_channel / 127.5) - 1.0  # Normalize to [-1, 1]
    L_input = tf.expand_dims(tf.expand_dims(L_norm, axis=0), axis=-1)  # Shape: [1, H, W, 1]

    
    with st.spinner('Colorizing...'):
        # Run inference
        ab_output = generator(L_input, training=False)
        ab_output = ab_output[0].numpy() * 128.0 + 128.0  # Denormalize

        # Reconstruct LAB image
        L_denorm = (L_input[0, :, :, 0] + 1.0) * 127.5
        L_denorm = np.expand_dims(L_denorm, axis=-1)
        lab_combined = np.concatenate([L_denorm, ab_output], axis=-1).astype(np.uint8)

        # Convert LAB to RGB
        rgb_output = cv2.cvtColor(lab_combined, cv2.COLOR_LAB2RGB)

        #size: 256X256 after model inference
        # get original size from PIL Image Object and resize it back to input size
        original_size = image.size
        rgb_resized = cv2.resize(rgb_output, original_size, interpolation=cv2.INTER_LANCZOS4)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption = "Original (B&W)", use_container_width=True)
    
    with col2:
        st.image(rgb_resized, caption="Colorized Output", use_container_width=True)
    
    st.success("Colorization complete!")