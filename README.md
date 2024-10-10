This repository includes the implementations of AlexNet, ResNet-18, and Faster R-CNN using PyTorch. These models are commonly used for image classification and object detection tasks on datasets like CIFAR-10, ImageNet, and COCO.

Architectures Implemented
1. AlexNet
AlexNet is a pioneering deep learning model introduced in 2012, consisting of:
Five convolutional layers with max-pooling for feature extraction.
Three fully connected layers for classification.
Dropout layers for regularization.
Designed for large-scale image classification.
2. ResNet-18
ResNet-18 is part of the Residual Network family, designed to overcome training difficulties in deep networks:
Uses residual blocks with identity mappings to preserve gradients during backpropagation.
Consists of 18 layers, making it a lightweight version suitable for various tasks.
Ideal for image classification with deeper and more stable training.
3. Faster R-CNN (Object Detection)
Faster R-CNN is one of the most popular deep learning models for object detection:
Combines Region Proposal Networks (RPN) with Fast R-CNN, making object detection much faster and efficient.
The network identifies regions of interest (ROIs) in images and classifies objects within them.
Can be trained on large object detection datasets like COCO, PASCAL VOC, etc.
Features
AlexNet and ResNet-18 for Image Classification: Pre-built CNN architectures that can be modified for classification tasks with a flexible number of output classes.
Faster R-CNN for Object Detection: Pre-trained Faster R-CNN that can be fine-tuned for detecting objects in custom datasets.
Modular PyTorch Implementations: Each architecture is designed with reusability and customization in mind.
Efficient Training and Evaluation: Models include support for training, evaluation, and testing pipelines.
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/deep-learning-architectures.git
Install dependencies:
bash
Copy code
pip install torch torchvision
Usage
AlexNet Example:
python
Copy code
from alexnet import AlexNet
model = AlexNet(num_classes=2)  # Modify num_classes based on your dataset
ResNet-18 Example:
python
Copy code
from resnet import ResNet18
model = ResNet18(num_classes=2)  # Modify num_classes based on your dataset
Faster R-CNN Example:
To use Faster R-CNN for object detection, you can load a pre-trained model from the PyTorch torchvision library and fine-tune it for custom object detection tasks.

python
Copy code
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load Faster R-CNN pre-trained on COCO dataset
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the model for a custom number of object classes (num_classes includes background)
num_classes = 3  # Example: 2 object classes + 1 background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

# Load images and perform object detection
images, _ = dataset[0]  # Load an example image
output = model([images])  # Perform object detection
Training Faster R-CNN
For training Faster R-CNN on your own dataset, prepare a dataset in COCO or PASCAL VOC format and use the standard PyTorch DataLoader:

python
Copy code
from torch.utils.data import DataLoader

# Prepare your dataset and DataLoader
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
for images, targets in train_loader:
    optimizer.zero_grad()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    losses.backward()
    optimizer.step()
Contributing
Contributions, issues, and feature requests are welcome! Feel free to open a pull request or an issue for any improvements or questions.

License
This repository is licensed under the MIT License. See the LICENSE file for more details.

