from setuptools import setup
setup(
    name="src",
    version="0.0.6",
    author="Sai Kiran Dyavadi",
    description= "A small application showing the face classification",
    author_email="dyavadi324@gmail.com",
    packages=["src"],
    python_requires=">3.7.0",
    install_requires=[
        "mtcnn==0.1.0",
        "tensorflow==2.3.1",
        "keras==2.4.3",
        "keras-vggface==0.6",
        "keras_applications==1.0.8",
        "PyYAML==5.3.1",
        "tqdm==4.46.1",
        "scikit-learn==0.23.2",
        "streamlit==0.66.0",
        "bing-image-downloader==1.1.2"

    ]
)

