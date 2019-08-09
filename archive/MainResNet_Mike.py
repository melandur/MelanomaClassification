import os, sys, time
import numpy as np
import shutil


# Torch modules
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable

# Own modules
from dataloader import DataManager
import ResNet as rs

"""
labels:

1: Melanoma
2: Nevus
3: Seborrheic Keratosis
"""

def reduce_img_dimensions(self, images):
    numOfImages, numOfChannels, width, height = images.shape
    reduced_images = torch.zeros(numOfImages, width, height).type(torch.FloatTensor)

    for i in range(numOfImages):
        reduced_images[i, :, :] = normalize(
            torch.sqrt(
                torch.pow(images[i, 0, :, :], 2) +
                torch.pow(images[i, 1, :, :], 2) +
                torch.pow(images[i, 2, :, :], 2)
            )
        )
    return reduced_images

def normalize(image):
    return torch.div(image, image.sum())

def load_data():
    # load images and labels (actually, it is get_sebor because this set is smaller and therefore faster loaded, no time to load the whole set)
    images, labels = mymanager.reduce_data(mymanager.get_images(as_tensor=True), mymanager.get_labels(as_tensor=True),
                                           indexFrom=0, indexTo=data_size)
    labels = labels.type(torch.FloatTensor)

    # check if images and labels have similar size
    if len(images) != len(labels):
        raise Exception('Error: Images and labels does not have equal length')

    # shuffle images and labels
    images, labels = mymanager.shuffle(images, labels)

    # split data in train, test and validation subset
    X_train, y_train, X_test, y_test, X_val, y_val = mymanager.datasplit(images, labels, train_size=train_size,
                                                                         validation_size=0.1)
    # print('X_train: {} with {}'.format(X_train.type(), X_train.shape) + "\n" +
    #       'X_test:  {} with {}'.format(X_test.type(), X_test.shape) + "\n" +
    #       'X_val:   {} with {}'.format(X_val.type(), X_val.shape) + "\n" +
    #       'y_train: {} with {}'.format(y_train.type(), y_train.shape) + "\n" +
    #       'y_test:  {} with {}'.format(y_test.type(), y_test.shape) + "\n" +
    #       'y_val:   {} with {}\n'.format(y_val.type(), y_val.shape))

    return X_train, y_train, X_test, y_test, X_val, y_val

def train():
    if torch.cuda.is_available():
        model.cuda()
    print('Model generated!')

    # Initialize crierion
    criterion = torch.nn.MSELoss(size_average=True)
    # criterion = torch.nn.MultiLabelMarginLoss()
    print('Criterion generated!')

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print('Optimizer generated!')

    for epoch in range(num_epochs):

        losses = []

        for i, (images, labels,) in enumerate(train_dataloader):

            # Convert the images to a torch vector of appropriate size
            X = Variable(images.view(-1, 3, 256, 256))
            y = Variable(labels, requires_grad=False)

            # Compute the prediction
            y_pred = model(X)

            # Compute the loss
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check the loss
            if (i + 1) % 1 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.6f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss))

def test():
    # Test the net on the test set
    correct = 0
    total = 0
    lab = torch.zeros(batch_size, 1).type(torch.IntTensor)

    for data in test_dataloader:
        images, labels = data
        outputs = model(Variable(images))

        # Make prediciton with trained model
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)

        # Reshape prediction from 50 to 50x1
        pred = pred.view(-1,1).type(torch.IntTensor)

        # Rearange labels from 50x3 to 50x1, from [1 0 0] -> 0, [0 1 0] -> 1, [0 0 0] -> 2
        i = 0
        for ind in labels:
            _, indices = ind.max(0)
            lab[i] = indices
            i += 1

        # Convert tensors to numpy arrays
        numpy_labels = lab.numpy()
        numpy_pred = pred.numpy()

        correct += (numpy_pred == numpy_labels[0]).sum()

    print('Accuracy of the network on the ' + str(batch_size) + ' test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    # Enable this if cuda is available

    # Epochs
    num_epochs = 50

    # data size
    data_size = 300

    # Batch size
    batch_size = 50

    # Load / Safe
    load = True
    safe = True

    # Parameter
    train_size = 0.7
    validation_size = 0.1

    # image_path = 'data/ISIC-2017_Training_Data'
    image_path = 'data/processed'
    mask_path = 'data/ISIC-2017_Training_Part1_GroundTruth'
    label_file = 'data/ISIC-2017_Training_Part3_GroundTruth.csv'

    # call class DataManger
    mymanager = DataManager(image_path, mask_path, label_file)

    X_train, y_train, X_test, y_test, X_val, y_val = load_data()

    # Initialize the variables
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        y_train = mymanager.convert_labels(y_train).cuda()
        X_test = X_test.cuda()
        y_test = mymanager.convert_labels(y_test).cuda()
    else:
        X_train = X_train
        y_train = mymanager.convert_labels(y_train) #.type(torch.LongTensor)
        X_test = X_test
        y_test = mymanager.convert_labels(y_test) #.type(torch.LongTensor)

    # Generate mini-batch for learning
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = rs.resnet18(pretrained=False)

    if load:
        model = torch.load('resNet1.pt')
        # model = torch.load('resNet1.pt',  map_location='cpu')
        # model = torch.load('resNet1.pt', map_location=lambda storage, loc: storage.cuda(0))

        print('Model loaded!')

    # Train model
    train()


    if safe:
        torch.save(model, 'resNet1.pt')

    test()