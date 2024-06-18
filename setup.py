from setuptools import setup
setup(
    name="src",
    version="0.0.1",
    author="Sai Kiran Dyavadi",
    description= "A small application showing the face classification",
    author_email="dyavadi324@gmail.com",
    packages=["src"],
    python_requires=">=3.11.7",
    install_requires=[
        "mtcnn",
        "tensorflow",
        "keras",
        "keras-vggface",
        "keras_applications",
        "PyYAML",
        "tqdm",
        "scikit-learn",
        "streamlit",
        "bing-image-downloader"

    ]
)