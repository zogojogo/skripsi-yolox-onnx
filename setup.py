from setuptools import setup, find_packages

setup(
    name='skripsi',
    version='0.1.0',
    description='Implementation of Indoor object detection using ONNX and OpenCV',
    author='Kristian Wilianto',
    author_email='kriswiliant0@gmail.com',
    url='https://github.com/zogojogo/yolox-onnx.git',
    install_requires=[
        'matplotlib',
        'numpy',
        'onnxruntime',
        'opencv_contrib_python'],
    packages={
        'skripsi': "skripsi",
    },
    python_requires=">=3.8",
    classifiers=[
        'Development Status :: 3 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
    ],
)