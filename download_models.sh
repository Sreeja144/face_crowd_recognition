#!/bin/bash

# Create the models directory
mkdir -p models/buffalo_l

# Download each ONNX model from InsightFace HuggingFace repo
curl -L -o models/buffalo_l/2d106det.onnx https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/2d106det.onnx
curl -L -o models/buffalo_l/det_10g.onnx https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/det_10g.onnx
curl -L -o models/buffalo_l/genderage.onnx https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/genderage.onnx
curl -L -o models/buffalo_l/w600k_r50.onnx https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx

# Then start Streamlit
#streamlit run app.py
