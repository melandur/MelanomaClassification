from ImageSorter import ImageSorter
import os, copy
import numpy as np
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage import transform
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity

"""
This modules can be used to copy, generate and delete augmented images.


Author:     Elias Ruefenacht, Student, University of Bern, Switzerland
Version:    V01 (27.04.2018)

"""


class DataAugmentor:
    """
    Class for the execution of augmentation operations

    Args:
        target_path (str):      The path where the images are stored
        labels ([str]):         A list of labels
    """

    def __init__(self, target_path, labels, copy=False, origin_path=None, label_file_path=None):

        # Target directory for the images
        self.target_path = target_path

        # Labels
        self.labels = labels

        # Dict for the original images
        self.filenames_all = {}

        # Copy images if appropriate
        if copy:
            img_sort = ImageSorter(origin_path, label_file_path, target_path, labels)
            img_sort.generate_target_structure()
            img_sort.copy_images()

        # Check if the target directory contains the appropriate structure
        self.correct_target_structure = self._check_dir_on_structure_(self.target_path, self.labels)
        if not self.correct_target_structure:
            print('The target directory of the data does not contain the correct structure!')

        else:
            # Get filenames of original images
            for label in self.labels:
                self.filenames_all[label] = self._get_filenames_in_directory_(target_path, label)




    def all(self, origin_path, target_path, oper):
        """
        This function performs an operation on all images in the original path and save them in the target path

        Args:
            origin_path (str):      The path, where the images to process are stored
            target_path (str):      The path, where the processed images needs to be stored
            oper (ImageOperation):  The operation which should be performed on the images

        """

        # Get all the files in the directory
        files = [file for file in os.listdir(origin_path) if os.path.isfile(os.path.join(origin_path, file))]

        # Compute the operation
        for file in tqdm(files, desc='Processing of images'):

            # Get the image
            image = imread(os.path.join(origin_path, file))

            # Perform the operation
            proc_image, oper_desc = oper.proc(image)

            # Get the augmented filename
            filename_aug = self._build_augmented_filename_(file, oper_desc, None)

            # Save the processed image
            imsave(os.path.join(target_path, filename_aug), proc_image)

    def all2(self, oper):
        """
        This function performs an operation on all images in the target path of the class

        Args:
            oper (ImageOperation):  The operation which should be performed on the images

        """

        # Loop through all labels
        for label in list(self.filenames_all.keys()):

            # Get the label specific path
            origin_path = self._get_directory_path_(self.target_path, label)

            # Loop through the images
            for file in tqdm(self.filenames_all[label], desc=str('Processing of ' + label + ' data')):

                # Get the image
                image = imread(os.path.join(origin_path, file))

                # Perform the operation
                proc_image, oper_desc = oper.proc(image)

                # Get the augmented filename
                filename_aug = self._build_augmented_filename_(file, oper_desc)

                # Save the processed image
                imsave(os.path.join(origin_path, filename_aug), proc_image)

    def all_sequential(self, origin_path, target_path, opers, oper_desc):
        """
        This function performs multiple operations on all images in the original path and save them in the target path

        Args:
            origin_path (str):          The path, where the images to process are stored
            target_path (str):          The path, where the processed images needs to be stored
            oper ([ImageOperation]):    The list of operations which should be performed on the images

        """

        # Get all the files in the directory
        files = [file for file in os.listdir(origin_path) if os.path.isfile(os.path.join(origin_path, file))]

        # Compute the operation
        for file in tqdm(files, desc='Processing of images'):
            # Get the image
            proc_image = imread(os.path.join(origin_path, file))

            # Perform the operation
            for oper in opers:
                proc_image, _ = oper.proc(proc_image)

            # Get the augmented filename
            filename_aug = self._build_augmented_filename_(file, str(oper_desc), None)

            # Save the processed image
            imsave(os.path.join(target_path, filename_aug), proc_image)

    def all_sequential2(self, opers, oper_desc='misc'):
        """
        This function performs multiple operations on all images in the target path of the class

        Args:
            opers ([ImageOperation]):    The list of operations which should be performed on the images
            oper_desc (str):            A descriptor for the operations

        """
        # Loop through all labels
        for label in list(self.filenames_all.keys()):

            # Get the label specific path
            origin_path = self._get_directory_path_(self.target_path, label)

            # Loop through the images
            for file in tqdm(self.filenames_all[label], desc=str('Processing of ' + label + ' data')):

                # Get the image
                proc_image = imread(os.path.join(origin_path, file))

                # Perform the operation
                for oper in opers:
                    proc_image, _ = oper.proc(proc_image)

                # Get the augmented filename
                filename_aug = self._build_augmented_filename_(file, oper_desc)

                # Save the processed image
                imsave(os.path.join(origin_path, filename_aug), proc_image)

    def generate_new_data_structure(self, origin_path, label_file_path):
        """
        Generates a new folder system, copies and sort the images

        Args:
            origin_path (str):      The path, where the original images lies
            label_file_path (str):  The path to the CSV-file with the labels
        """

        try:
            image_sorter = ImageSorter.ImageSorter(origin_path, label_file_path, self.target_path, self.labels)
            image_sorter.generate_target_structure()
            image_sorter.copy_images()

        except Exception as e:
            print(type(e))
            print(e.args)
            print(e)

    @staticmethod
    def _check_dir_on_structure_(path, labels):
        """
        This function checks if the path contains the necessary folder structure or not

        Args:
            path (str):     The path, which should be checked

        Returns:
            contains(bool): True if the directory contains the necessary folder structure
        """

        # Instantiate a new list for the subfolders
        subfolders = []

        # Define the result variable
        contains = False

        # Iterate through the subfolders
        for folder_name in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder_name)) and labels.__contains__(folder_name):
                subfolders.append(folder_name)

        # Check if the subdirectories are set correct
        if frozenset(subfolders).intersection(labels):
            contains = True

        return contains

    @staticmethod
    def _get_directory_path_(path, label):
        """
        This function join the path of the subdirectory correctly

        Args:
            path (str):         The top directory path of the data
            label (str):        The label for which the path should be generated

        Returns:
            subdir_path (str):  The path of the appropriate subdirectory
        """

        return os.path.join(path, label)

    @staticmethod
    def _get_filenames_in_directory_(path, label=None):
        """
        This function gets the filenames in the appropriate directory

        Args:
            path (str):     The path which should be examined
            label (str):    If a label is provided, the path is extended with the labels directory

        Returns:
            value ([str]):  A list of the filenames in the appropriate directory
        """

        # Get the directory path
        if label is None:
            origin_path = path
        else:
            origin_path = os.path.join(path, label)

        # Return the filenames
        return [file for file in os.listdir(origin_path) if os.path.isfile(os.path.join(origin_path, file))]

    @staticmethod
    def _build_augmented_filename_(original_filename, operation_desc, operation_value=None):
        """
        This function concatenates the filename of a processed image

        Args:
            original_filename (str):    The filename of the unprocessed image
            operation_desc (str):       A descriptor for the operation
            operation_value:            An iterator for the separation of the images

        Returns:
            value (str):                The augmented filename
        """

        # Split the filename and the extension
        filename_raw = os.path.splitext(os.path.basename(original_filename))[0]
        ext = '.'.join(original_filename.split('.')[1:])

        # Build the new filename
        if operation_value is None:
            return str(filename_raw + '_' + str(operation_desc) + '.' + ext)
        else:
            return str(filename_raw + '_' + str(operation_desc) + '_' + str(operation_value) + '.' + ext)

    def delete_augmented_files(self, target_path=None):
        """
        This function deletes the augmented files

        Args:
            target_path (str):  The path in which the augmented files should be deleted

        """

        # If no target path is provided process all generated files
        if target_path is None:

            # Loop through all labels
            for label in list(self.filenames_all.keys()):

                # Get the label specific path
                origin_path = self._get_directory_path_(self.target_path, label)

                # Get all the filenames within this directory
                filenames = self._get_filenames_in_directory_(self.target_path, label)

                # Delete the augmented files
                for filename in filenames:
                    if len(filename.split(sep='_'))>2:
                        os.remove(os.path.join(origin_path, filename))

        else:

            # Get all the filenames within this directory
            filenames = self._get_filenames_in_directory_(target_path, None)

            # Delete the augmented files
            for filename in filenames:
                if len(filename.split(sep='_')) > 2:
                    os.remove(os.path.join(origin_path, filename))


class ImageOperation(metaclass=ABCMeta):
    """
    The base class for the operations
    """

    @abstractmethod
    def proc(self, image):
        return image


class RotateRandom(ImageOperation):
    """
    This class rotates an input image by a random angle (angle range: 1 - angle_max)

    Args:
        angle_max (dbl):  The maximal rotation angle (angle = 1 - angle_max)
    """
    def __init__(self, angle_max=359):

        # Sets the maximal rotation angle
        self.angle_max = angle_max

    def proc(self, image):
        """
        This function performs a rotation of an image with a random angle (angle range: 1 - angle_max)
            Args:
                image (ndarray):        The image to rotate

            Returns:
                proc_image (ndarray):   The rotated image
                operation_desc (str):   A characteristic descriptor of this operation
        """

        # Create a random angle in the appropriate range
        angle = np.random.random_sample(1)[0] * (self.angle_max - 1) + 1

        # Return the rotated image
        return transform.rotate(image, -angle), 'rot'


class ChromaticAberrationRandom(ImageOperation):
    """
    This class generates a random chromatic aberration on an image

    Args:
        shift_max (int):        The maximal shift of the color channel
    """

    def __init__(self, shift_max=2):

        # Set the max shift
        self.shift_max = shift_max

    def proc(self, image):
        """
        This function performs a random chromatic aberration on an image

        Args:
            image (ndarray):        The image to process

        Returns:
            proc_image (ndarray):   The processed image
            operation_desc (str):   A characteristic descriptor of this operation
        """

        # Choose randomly the changeable channels
        variable_chs = np.random.randint(0, 3, 2)

        # Choose random shifts
        shift_ch1_x = np.random.randint(0, self.shift_max + 1)
        shift_ch1_y = np.random.randint(0, self.shift_max + 1)
        shift_ch2_x = np.random.randint(0, self.shift_max + 1)
        shift_ch2_y = np.random.randint(0, self.shift_max + 1)

        # Copy the image
        nimage = np.array(copy.deepcopy(image))

        # Swap the rows and columns
        nimage[:, :, variable_chs[0]] = np.roll(nimage[:, :, variable_chs[0]], shift_ch1_x, axis=1)
        nimage[:, :, variable_chs[0]] = np.roll(nimage[:, :, variable_chs[0]], shift_ch1_y, axis=0)
        nimage[:, :, variable_chs[1]] = np.roll(nimage[:, :, variable_chs[1]], shift_ch2_x, axis=1)
        nimage[:, :, variable_chs[1]] = np.roll(nimage[:, :, variable_chs[1]], shift_ch2_y, axis=0)

        return nimage, 'aber'


class GaussianNoise(ImageOperation):
    """
    This class adds Gaussian noise to an image

    Args:
        random_sigma (boolean):     True, if the sigma should be chosen randomly
        sigma (float):              If random_sigma is false, then take this sigma instead
        max_sigma (float):          If random_sigma is false, then this variable limits the randomly generated sigma
    """
    def __init__(self, random_sigma=True, sigma=0.01, max_sigma=0.01):

        # Assign the values
        self.sigma = sigma
        self.random_sigma = random_sigma
        self.max_sigma = max_sigma

    def proc(self, image):
        """
        This function performs the addition of Gaussian noise to an image

        Args:
            image (ndarray):            The image to process

        Returns:
            proc_image (ndarray):       The processed image
            operation_desc (str):       A characteristic descriptor of this operation

        """

        # Generate the random sigma if necessary
        if self.random_sigma:
            sigma = np.random.random_sample(1)[0] * self.max_sigma
        else:
            sigma = self.sigma

        return random_noise(image, mode='gaussian', var=sigma), 'gno'


class Blur(ImageOperation):
    """
    This class blurs an image

        Args:
            random_sigma (boolean):     True, if the sigma should be chosen randomly
            sigma (float):              If random_sigma is false, then take this sigma instead
            max_sigma (float):          If random_sigma is false, then this variable limits the randomly generated sigma
        """

    def __init__(self, random_sigma=True, sigma=0.01, max_sigma=0.2):
        # Assign the values
        self.sigma = sigma
        self.random_sigma = random_sigma
        self.max_sigma = max_sigma

    def proc(self, image):
        """
        This function performs the blurring on image

        Args:
            image (ndarray):            The image to process

        Returns:
            proc_image (ndarray):       The processed image
            operation_desc (str):       A characteristic descriptor of this operation
        """

        # Generate the random sigma if necessary
        if self.random_sigma:
            sigma = np.random.random_sample(1)[0] * self.max_sigma
        else:
            sigma = self.sigma

        # Check if the image contains multiple channels
        is_color = len(image.shape) == 3

        return rescale_intensity(gaussian(image, sigma=sigma, multichannel=is_color)), 'blur'


class ZoomRandom(ImageOperation):
    """
    This class zooms and crops an image randomly

    Args:
        max_zoom_factor (float):    The maximal zoom factor
        min_zoom_factor (float):    The minimal zoom factor
    """

    def __init__(self, max_zoom_factor=2, min_zoom_factor=1.1):

        self.max_zoom_factor = max_zoom_factor
        self.min_zoom_factor = min_zoom_factor

    def proc(self, image):
        """
        This function performs the random zooming of an image

        Args:
            image (ndarray):            The image to process

        Returns:
            proc_image (ndarray):       The processed image
            operation_desc (str):       A characteristic descriptor of this operation
        """

        # Get the image dimensions
        height = len(image)
        width = len(image[0])

        # Get the random zoom factor
        zoom_factor = np.random.random_sample(1)[0]*(self.max_zoom_factor-self.min_zoom_factor)+self.min_zoom_factor

        # Get the cropping size w.r.t. the image
        crop_height = int(np.round(height/zoom_factor))
        crop_width = int(np.round(width/zoom_factor))

        # Get random points for the croping
        crop_point1_x = int(np.random.randint(0, width - crop_width, 1)[0])
        crop_point1_y = int(np.random.randint(0, height - crop_height, 1)[0])
        crop_point2_x = crop_point1_x + crop_width
        crop_point2_y = crop_point1_y + crop_height

        # Crop the image
        cropped_image = image[crop_point1_y:crop_point2_y, crop_point1_x:crop_point2_x]

        return transform.resize(cropped_image, (height, width)), 'zoom'


class HFlip(ImageOperation):
    """
        This class flips an input image horizontically
    """

    def proc(self, image):
        """
        This function flips the input image horizontically

        Args:
            image (ndarray):    The input image to flip horizontically


        Returns:
            value (ndarray):    The processed image

        """

        return np.fliplr(image)


class VFlip(ImageOperation):
    """
        This class flips an input image vertically
    """

    def proc(self, image):
        """
        This function flips the input image vertically

        Args:
            image (ndarray):    The input image to flip vertically


        Returns:
            value (ndarray):    The processed image

        """

        return np.flipud(image)