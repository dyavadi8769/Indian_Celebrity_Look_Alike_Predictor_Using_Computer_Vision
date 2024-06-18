'''
Author: Sai Kiran Reddy Dyavadi
Email: dyavadi324@gmail.com
Date:18-June-2024
'''


import os
import cv2
import numpy as np
import pickle
import logging
import streamlit as st
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.all_utils import read_yaml, create_directory

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load configuration
config = read_yaml('config/config.yaml')
params = read_yaml('params.yaml')

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

# Upload directory
upload_image_dir = artifacts['upload_image_dir']
upload_path = os.path.join(artifacts_dir, upload_image_dir)

# Pickle file paths
pickle_format_data_dir = artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)

# Feature extraction path
feature_extraction_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)
features_name = os.path.join(feature_extraction_path, extracted_features_name)

# Model parameters
model_name = params['base']['BASE_MODEL']
include_tops = params['base']['include_top']
input_shapes = tuple(params['base']['input_shape'])
poolings = params['base']['pooling']

# Initialize MTCNN detector and VGGFace model
detector = MTCNN()
model = VGGFace(model=model_name, include_top=include_tops, input_shape=input_shapes, pooling=poolings)
feature_list = pickle.load(open(features_name, 'rb'))
filenames = pickle.load(open(pickle_file, 'rb'))

# Save uploaded image function
def save_uploaded_image(uploaded_image, save_path):
    try:
        create_directory(dirs=[upload_path])
        with open(save_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())
        logging.info(f"Image saved successfully to {save_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving image: {e}")
        return False

# Extract features function
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    
    if results:
        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]

        image = Image.fromarray(face)
        image = image.resize((224, 224))

        face_array = np.asarray(image)
        face_array = face_array.astype('float32')

        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        result = model.predict(preprocessed_img).flatten()
        return result
    else:
        logging.error("No face detected")
        return None

# Recommend image function
def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), feature.reshape(1, -1))[0][0] for feature in feature_list]
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

# Streamlit UI
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('To whom does your face match?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # Manually assign a file name to the uploaded image
    file_name = "uploaded_image.jpg"
    save_path = os.path.join(upload_path, file_name)
    
    if save_uploaded_image(uploaded_image, save_path):
        display_image = Image.open(uploaded_image)

        features = extract_features(save_path, model, detector)
        if features is not None:
            index_pos = recommend(feature_list, features)

            file_path = filenames[index_pos]
            predicted_actor = " ".join(os.path.basename(file_path).split('.')[0].rsplit(' ', 2)[:2])

            st.header('Your uploaded image')
            st.image(display_image)

            st.header("You look like " + predicted_actor)
            st.image(filenames[index_pos], width=300)
        else:
            st.error("No face detected in the uploaded image.")
    else:
        st.error("Error in saving the uploaded image.")
