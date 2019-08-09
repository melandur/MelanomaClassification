import os
import errno
import csv
import shutil


class ImageSorter:
    """
    Copies the images in a new structure based on a separate folder for each class

    Args:
        origin_path (str):      The path of the images
        label_file_path (str):  The CSV file, which contains the labels
        target_path (str):      The target path for the images
        labels (str list):      A list of the different class labels

    """

    def __init__(self, origin_path, label_file_path, target_path, labels):

        # Sets the origin path of the images
        self.originPath = origin_path

        # Sets the label file path
        self.labelFilePath = label_file_path

        # Sets the target path of the images
        self.targetPath = target_path

        # Sets the labels as a list of strings
        self.labels = labels

    def generate_target_structure(self):
        """
        Function that generates the folder structure in the target directory

        Returns:
            targetPaths (str list): A list of the target paths
            result (bool):          True if the operation was successful, otherwise False

        """

        # Set the result to False for default
        result = False

        # Initialize an empty list for the targetPaths
        target_paths = []

        # Get the target directory
        target_directory = os.path.dirname(self.targetPath)

        # Check if the directory is existing
        try:

            # Generate the main target directory
            os.makedirs(target_directory)

            for cls in self.labels:

                # Generate the new directory object
                temp_path = os.path.join(target_directory, str(cls))

                # Generate the new directory
                os.makedirs(temp_path)

                # Append the tempPath to the targetPaths
                target_paths.append(temp_path)

        # Catch for the exception
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        return target_paths, result

    def copy_images(self):
        """
        Function that copies the images in the origin path to the target path and sort them in the correct subfolder

        """

        # Instantiate a dictionary
        example_association = {}

        # Open the csv label file
        with open(self.labelFilePath, 'r') as label_file:

            # Instantiate a csv reader
            reader = csv.reader(label_file, delimiter=',')

            # Omit the header of the file
            next(reader)

            # Iterate through the rows of the file
            for row in reader:

                # Change the definition of the labels
                if row[1] == '1.0':
                    lbl = self.labels[0]
                elif row[2] == '1.0':
                    lbl = self.labels[1]
                else:
                    lbl = self.labels[2]

                # Append the filename and the label to the dictionary
                example_association.update({row[0]:lbl})

            del lbl

        # Iterate through the dictionary
        for filename, label in example_association.items():

            # Copy the files depending on their labels
            if label == self.labels[0]:
                shutil.copy2(
                    os.path.join(self.originPath, 'processed/', str(filename + ".jpg")),
                    os.path.join(self.targetPath, str(self.labels[0] + '/' + filename + '.jpg'))
                )

            if label == self.labels[1]:
                shutil.copy2(
                    os.path.join(self.originPath, 'processed/', str(filename + ".jpg")),
                    os.path.join(self.targetPath, str(self.labels[1] + '/' + filename + '.jpg'))
                )

            if label == self.labels[2]:
                shutil.copy2(
                    os.path.join(self.originPath, 'processed/', str(filename + ".jpg")),
                    os.path.join(self.targetPath, str(self.labels[2] + '/' + filename + '.jpg'))
                )