from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from src.utils.all_utils import read_yaml,create_directory
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np


config=read_yaml('config/config.yaml')
params=read_yaml('params.yaml')

artifacts=config['artifacts']
artifacts_dir=artifacts['artifacts_dir']

#upload
upload_image_dir=artifacts['upload_image_dir']
upload_path=os.path.join(artifacts_dir,upload_image_dir)

#pickle format data dir
pickle_format_data_dir =artifacts['pickle_format_data_dir']
img_pickle_file_name=artifacts['img_pickle_file_name']

raw_local_dir_path=os.path.join(artifacts_dir,pickle_format_data_dir )

pickle_file = os.path.join(raw_local_dir_path,img_pickle_file_name)


#feature path

feature_extractor_dir = artifacts['feature_extraction_dir']
extracted_features_name= artifacts['extracted_features_name']
feature_extraction_path=os.path.join(artifacts_dir, feature_extractor_dir)
feature_name = os.path.join(feature_extraction_path, extracted_features_name)

model_name = params['base']['BASE_MODEL']
include_tops= params['base']['include_top']
poolings= params['base']['pooling']

