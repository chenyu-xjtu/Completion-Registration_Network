# CRNet: Completion-Registration Network for Partial Point Cloud Registration

## 📖 Introduction
CRNet is a deep learning framework for partial point cloud registration by first completing missing regions and then performing registration. This repository contains the implementation of our paper *"Research and implementation of point cloud registration method based on partial point cloud completion"*.

**Key Features**:
- 🧩 Two-stage architecture: Point cloud completion followed by registration
- 🔍 DGCNN-based feature extraction with local geometric awareness
- ⚡ Hard point elimination & hybrid elimination for efficient matching
- 🔄 Iterative Distance-Aware Similarity Matrix Convolution (IDAM)
- 🏆 State-of-the-art performance on MVP dataset

## 🚀 Installation
```bash
# Clone repository
git clone https://github.com/yourusername/CRNet.git
cd CRNet

# Install dependencies
conda create -n crnet python=3.8
conda activate crnet
pip install -r requirements.txt

# Install customized DGCNN components
cd models/DGCNN_Pytorch
python setup.py install
