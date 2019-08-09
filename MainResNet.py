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

class MainResNet():

    def load_data(self):
        # load images and labels (actually, it is get_sebor because this set is smaller and therefore faster loaded, no time to load the whole set)
        images, labels = mymanager.reduce_data(mymanager.get_images(as_tensor=True), mymanager.get_labels(as_tensor=True),
                                               indexFrom=start_index, indexTo=start_index + data_size)
        labels = labels.type(torch.FloatTensor)

        # check if images and labels have similar size
        if len(images) != len(labels):
            raise Exception('Error: Images and labels does not have equal length')

        # shuffle images and labels
        images, labels = mymanager.shuffle(images, labels)

        # split data in train, test and validation subset
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val \
            = mymanager.datasplit(images, labels, train_size=train_size, validation_size=0.1)



    def dataloader(self):
        # Initialize the variables cpu or gpu based
        if use_cuda:
            X_train = self.X_train.cuda()
            y_train = mymanager.convert_labels(self.y_train).cuda()
            X_test = self.X_test.type(torch.FloatTensor).cuda()
            y_test = mymanager.convert_labels(self.y_test).type(torch.FloatTensor).cuda()
            X_val = self.X_val.type(torch.FloatTensor).cuda()
            y_val = mymanager.convert_labels(self.y_val).type(torch.FloatTensor).cuda()

        else:
            X_train = self.X_train
            y_train = mymanager.convert_labels(self.y_train)
            X_test = self.X_test
            y_test = mymanager.convert_labels(self.y_test)
            X_val = self.X_val
            y_val = mymanager.convert_labels(self.y_val)

        # Generate mini-batch for learning
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    def train(self):
        if use_cuda:
            model.cuda()

        # Min accuracy value for saving the model the first time
        val_best_acc = 10

        # Initialize crierion
        criterion = torch.nn.MSELoss(size_average=True)
        print('Criterion generated!')

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        print('Optimizer generated!')

        for epoch in range(num_epochs):
			start_time = time.time()
            train_correct = 0
            train_total = 0
            model.train()
            for i, (images, labels,) in enumerate(self.train_dataloader):
                # Convert the images to a torch vector of appropriate size
                X_train = Variable(images.view(-1, 3, 256, 256))
                y_train = Variable(labels, requires_grad=False)

                # Compute the prediction
                y_train_pred = model(X_train)

                # Compute the loss
                train_loss = criterion(y_train_pred, y_train)

                # Adjust the weights
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # Calculate accuracy of train set
                train_images = Variable(images.view(-1, num_channels, height, width))
                train_output = model(train_images)

                _, train_indices = train_output.max(1)
                train_labels = labels
                _, train_condensed_labels = train_labels.max(1)

                train_correct += torch.eq(train_indices.data, train_condensed_labels).sum()
                train_total += len(train_condensed_labels)

            train_accuracy = 100 * train_correct / train_total


            # Validation after each epoch, save the model with the lowest loss
            val_correct = 0
            val_total = 0
            model.eval()

            for i, (images, labels,) in enumerate(self.val_dataloader):
                # Convert the images to a torch vector of appropriate size
                X_val = Variable(images.view(-1, num_channels, height, width))
                y_val = Variable(labels, requires_grad=False)

                # Compute the validation prediction
                y_val_pred = model(X_val)

                # Compute the validation loss
                val_loss = criterion(y_val_pred, y_val)

                # Calculate accuracy of validation set
                val_images = Variable(images.view(-1, num_channels, height, width))
                val_output = model(val_images)

                _, val_indices = val_output.max(1)
                val_labels = labels
                _, val_condensed_labels = val_labels.max(1)

                val_correct += torch.eq(val_indices.data, val_condensed_labels).sum()
                val_total += len(val_condensed_labels)

            val_accuracy = 100 * val_correct / val_total

            # Save automatically the best model with the best accuracy
            if val_accuracy > val_best_acc:
                val_best_acc = val_accuracy
                best_model = model

            # Print the train and validation results for each epoch
            print('************************************')
            print('Epoch [%d/%d]' %(epoch + 1, num_epochs))
            print('train Loss: %f  Acc: %d%%' % (train_loss / batch_size, train_accuracy))
            print('valid Loss: %f  Acc: %d%%' % (val_loss / batch_size, val_accuracy))
			print('Needed Time: %.2f h \n' % ((time.time()-start_time)*(num_epochs-epoch)/3600))
			
		# Safe the best model and load the best model for the testing
		torch.save(best_model, save_name + str(format(val_best_acc, '.2f')))
		self.model = best_model


    def test(self):
        # Test the model on the test set
        correct = 0
        total = 0
        model.eval()

        for images, labels in self.test_dataloader:
            images = Variable(images.view(-1, num_channels, height, width))
            output = model(images)

            _, indices = output.max(1)
            _, condensed_labels = labels.max(1)

            correct += torch.eq(indices.data, condensed_labels).sum()
            total += len(condensed_labels)

        accuracy = 100 * correct / total
        print('Accuracy of the network on the test images: %.4f %%' % accuracy)


if __name__ == '__main__':
    # Enable this if cuda is available
    use_cuda = True

    # Epochs
    num_epochs = 300

    # data size
    data_size = 2000
    start_index = 0

    # Batch size
    batch_size = 10

    # Image properties
    num_channels = 3
    width = 256
    height = 256

    # Load / Safe
    load = False
    train_model = True
    save = False
    test_model = True

    # Parameter
    train_size = 0.9
    validation_size = 0.1

    # image_path = 'data/ISIC-2017_Training_Data'
    image_path = 'data/processed'
    mask_path = 'data/ISIC-2017_Training_Part1_GroundTruth'
    label_file = 'data/ISIC-2017_Training_Part3_GroundTruth.csv'

    # call class DataManger
    mymanager = DataManager(image_path, mask_path, label_file)

    # call class MainResNEt
    r = MainResNet()

    # Load images and labels
    r.load_data()

    # Generate dataloaders for train, test, validation
    r.dataloader()

    # Initialize the model
    model = rs.resnet101(pretrained=False)
    save_name = 'resNet101_1.pt'

    # Load model
	if load:
		if use_cuda:
			model = torch.load(save_name, map_location=lambda storage, loc: storage.cuda(0))
			print('Model-gpu loaded!')
		else:
			model = torch.load(save_name,  map_location='cpu')
			print('Model-cpu loaded!')

    # Train model
    if train_model:
        print('Start training...')
        r.train()
        print('Training finished!')

    # Save the model
    if save:
        torch.save(model, save_name)
        print('Model saved!')

    # Test the model
    if test_model:
        print('Start testing...')
        r.test()
        print('Testing finished!')

