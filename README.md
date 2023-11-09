# CIFAR-10 Image Classification - Project1

This project is designed to explore and evaluate CNNs and Visual Transformer for the classification of the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. These classes include common objects like airplanes, automobiles, birds, cats, and more.

In this project, we provide a Python script, `cifar10.py`, that allows you to train and evaluate CNN models on the CIFAR-10 dataset. You can specify hyperparameters and choose from different model architectures for classification. The `visual_transformer.ipynb` notebook implements a vision transformer.

This README provides an overview of the project and instructions on how to use the script and notebook.

## Dataset

The CIFAR-10 dataset is automatically downloaded and used for training and testing. You don't need to download the dataset separately.

## CNN
Before using the script, you need to set up your environment and install the required dependencies. We recommend using a virtual environment to isolate your project dependencies. You can create a virtual environment using `venv` or `conda`. To install the required packages, use the following command:

```bash
pip install torch torchvision scikit-learn tqdm
```

### Usage

You can run the script with the following command:

```bash
python cifar10.py [options]
```

### Options

- `-h`, `--help`: Show the help message and exit.

#### Model Configuration

- `--model_name {CNN,CustomCNN,TransferCNN}`: Choose the model architecture for classification. You can select from the following models:
  - `CNN`: A simple convolutional neural network (CNN).
  - `CustomCNN`: A custom CNN architecture.
  - `TransferCNN`: A transfer learning model (e.g., using pre-trained models like ResNet, VGG, etc.).

#### Hyperparameters

- `--lr LR`: Set the learning rate for training the model. The default value is 0.001.
- `--epochs EPOCHS`: Specify the number of training epochs. The default value is 20.
- `--batch_size BATCH_SIZE`: Set the batch size for training. The default value is 32.
- `--num_workers NUM_WORKERS`: Specify the number of workers for data loading during training. The default value is 2.


### Results

The script will train the selected model on the CIFAR-10 dataset and display the training progress, including the loss and accuracy. After training, the script will also evaluate the model's performance on the test dataset and save the test accuracy, along with plots on ROC and Confusion Matrices.

## Vision Transformer

We have included a Jupyter notebook, `visual_transformer.ipynb`,  that demonstrates the implementation of Vision Transformers (ViTs) using PyTorch Lightning. ViTs are a powerful deep learning architecture that has gained popularity in computer vision tasks. The notebook provides a step-by-step guide on how to build and train Vision Transformers for image classification tasks, offering insights into their architecture, self-attention mechanisms, and pre-processing techniques. 

