import os
import glob
from PIL import Image
import numpy as np
import torch

"""
labels:

1: Melanoma
2: Nevus
3: Seborrheic Keratosis
"""


class DataManager:
    def __init__(self, image_path, mask_path, label_file):
        self.imagepath = image_path
        self.maskpath = mask_path
        self.labelfile = label_file

    # private function to get images
    def _get_images(self, path):
        image_list = []
        for filename in glob.glob(path + '/ISIC_00[0-9][0-9][0-9][0-9][0-9].jpg'):
            m = pattern.search(filename)
            filename2 = m.group(1)
            temp = Image.open(filename)
            keep = temp.copy()
            image_list.append(keep)
            temp.close()
        return image_list

    # private function to get masks (segmentation)
    def _get_masks(self):
        mask_list = []
        for filename in glob.glob(self.maskpath + '/ISIC_00[0-9][0-9][0-9][0-9][0-9]_segmentation.png'):
            im = Image.open(filename)
            mask_list.append(im)
        return mask_list

    def _generate_tensor(self, imagelist):
        np_list = []
        for image in imagelist:
            array=np.asarray(image).reshape(3, image.size[0], image.size[1])
            np_list.append(array)
        np_ndarray = np.stack(np_list, axis=0)
        img_tensor = torch.FloatTensor(np_ndarray)
        return img_tensor

    # Function to generate segmented, cropped and resized images
    def generate_preprocessed_images(self, res=256, save_images=False):
        imagelist = self._get_images(self.imagepath)
        masklist = self._get_masks()

        for i in range(len(imagelist)):
            mask = masklist[i]
            image = imagelist[i]
            background = mask.convert(mode='RGB')

            # apply mask to image
            image = Image.composite(image, background, mask)

            # get box of nonzero values, then crop
            box = image.getbbox()
            image = image.crop(box)
            w = box[2] - box[0]
            h = box[3] - box[1]

            if w > h:
                if (w - h) % 2 != 0: w += 1
                new = Image.new('RGB', (w, w), (0, 0, 0))
                new.paste(image, (0, int((w - h) / 2)))
            else:
                if (h - w) % 2 != 0: h += 1
                new = Image.new('RGB', (h, h), (0, 0, 0))
                new.paste(image, (int((h - w) / 2), 0))

            # Resize image with LANCZOS algorithm to res x res
            image = new.resize((res, res), resample=Image.LANCZOS)

            if save_images:
                if not os.path.isdir('data/processed'):
                    os.mkdir('data/processed')
                image.save('data/processed/' + imagelist[i].filename[29:])
            imagelist[i] = image

            i += 1
            if i % 100 == 0: print('Preprocessing image {}/{}'.format(i, len(imagelist)))
        return imagelist

    # function to get all labels
    def get_labels(self, as_tensor=False):
        tags = np.genfromtxt(self.labelfile, delimiter=',')[1:, 1:3]
        labels = np.ones((tags.shape[0])) * 2
        labels[tags[:, 0] == 1] = 1
        labels[tags[:, 1] == 1] = 3
        if as_tensor:
            return torch.IntTensor(labels)
        else:
            return labels

    # Function to get tensor labels of size 1x3
    def convert_labels(self, labels):
        tensor_labels = torch.zeros(len(labels), 3).type(torch.FloatTensor)
        for i in range(len(labels)):
            if labels[i] == 1: # Melanoma
                tensor_labels[i, 0] = 1
            elif labels[i] == 2: # Nevus
                tensor_labels[i, 1] = 1
            elif labels[i] == 3: # Seborrheic Keratosis
                tensor_labels[i, 2] = 1
        return tensor_labels

    # public function to retrieve all images
    def get_images(self, as_tensor=False):
        if as_tensor:
            return self._generate_tensor(self._get_images(self.imagepath))
        else:
            return self._get_images(self.imagepath)

    # reduces the images and the labels from index i to index j
    def reduce_data(self, images, labels, as_tensor=False, indexFrom=0, indexTo=1):
        if as_tensor:
            return self._generate_tensor(images[indexFrom:indexTo, :, :, :]), torch.IntTensor(labels[indexFrom:indexTo])
        else:
            return images[indexFrom:indexTo, :, :, :], labels[indexFrom:indexTo]

    # public function to get all melanoma
    def get_melanoma(self, as_tensor=False):
        images = self._get_images(self.imagepath)
        labels = self.get_labels()
        melanomes = []
        for i, label in enumerate(labels):
            if label == 1:
                melanomes.append(images[i])
        if as_tensor:
            return self._generate_tensor(melanomes)
        else:
            return melanomes

    # public function to get all nevi
    def get_nevi(self, as_tensor=False):
        images = self._get_images(self.imagepath)
        labels = self.get_labels()
        nevi = []
        for i, label in enumerate(labels):
            if label == 2:
                nevi.append(images[i])
        if as_tensor:
            return self._generate_tensor(nevi)
        else:
            return nevi

    # public function to get all Seborrheic Keratosis
    def get_sebor(self, as_tensor=False):
        images = self._get_images(self.imagepath)
        labels = self.get_labels()
        sebor = []
        for i, label in enumerate(labels):
            if label == 3:
                sebor.append(images[i])
        if as_tensor:
            return self._generate_tensor(sebor)
        else:
            return sebor

    # public function to shuffle images and labels
    def shuffle(self, images, labels, replace=False):
        indexes = np.arange(len(images))
        indexes = np.random.choice(indexes, size=len(images), replace=replace)
        np.random.shuffle(indexes)
        sh_images = images
        sh_labels = labels
        i = 0
        for ind in indexes:
            sh_images[i] = images[ind]
            sh_labels[i] = labels[ind]
            i += 1
        return sh_images, sh_labels

    # public function to split data in train, test and validation set
    def datasplit(self, images, labels, train_size=0.7, validation_size=0.1):
        if 1.0 < train_size > 0.0:
            print('Error datasplit, chose train_size in range 0.0 - 1.0')

        split, X_train, X_test, X_val, y_train, y_test, y_val = 0,0,0,0,0,0,0
        if validation_size == 0.0:
            split = int(len(images) * train_size)
            X_train = images[:split]
            X_test = images[split:]
            y_train = labels[:split]
            y_test = labels[split:]
        elif validation_size > 0.0:
            split = int((len(images) - len(images) * validation_size) * train_size)
            val_split = int(len(images) * validation_size)
            X_train = images[0:split]
            X_test = images[split:int(len(images)-val_split)]
            X_val = images[int(len(images)-val_split):len(images)]
            y_train = labels[0:split]
            y_test = labels[split:int(len(images)-val_split)]
            y_val = labels[int(len(images)-val_split):len(images)]
        else:
            print('Error datasplit, validation_size negative or to big')
        return X_train, y_train, X_test, y_test, X_val, y_val
