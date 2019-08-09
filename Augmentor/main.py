import DataAugmentor as da

""" 
This module can be used to perform a data augmentation.
You can compose different operations with it.

    

    Help:
        DataAugmentor.all:                  Base for the augmentation of all images
                                            with ONE operation (extended options) 

        DataAugmentor.all2:                 Base for the augmentation of all images
                                            with ONE operation (less options)

        DataAugmentor.all_sequential:       Base for the augmentation of all images
                                            with MULTIPLE operations (extended options)

        DataAugmentor.all_sequential2:      Base for the augmentation of all images
                                            with MULTIPLE operations (less options)

        DataAugmentor.RotateRandom:         Performs a random rotation

        DataAugmentor.ChromaticAberration:  Performs a random chromatic abberation

        DataAugmentor.GaussianNoise:        Performs the addition of Gaussian noise

        DataAugmentor.Blur:                 Performs a blurring

        DataAugmentor.ZoomRandom:           Performs a zooming and croping to original shape
"""


if __name__ == '__main__':

    # Augmentation of the provided data
    # You can compose your own augmentation
    dataAug = da.DataAugmentor(
        target_path='C:/Users/CorePy/Desktop/GrouPro/Augmentor/test1/',
        labels=['melanoma', 'seborrheic_keratosis', 'nevus'],
        copy=True,
        origin_path='C:/Users/CorePy/Desktop/GrouPro/data/',
        label_file_path='C:/Users/CorePy/Desktop/GrouPro/data/ISIC-2017_Training_Part3_GroundTruth.csv'
    # origin_path = 'C:/Users/eluru/Dropbox/Group_Project/data/',
    # label_file_path = 'C:/Users/eluru/Dropbox/Group_Project/data/ISIC-2017_Training_Part3_GroundTruth.csv'
    )
    dataAug.all_sequential2([da.ChromaticAberrationRandom(), da.GaussianNoise()], oper_desc='cg')
    dataAug.all_sequential2([da.ZoomRandom(), da.GaussianNoise()], oper_desc='zg')
    dataAug.all_sequential2([da.HFlip(), da.Blur()], oper_desc='hfb')


    # label_file_path = 'C:/Users/CorePy/Desktop/GrouPro/data/ISIC-2017_Training_Part3_GroundTruth.csv'


    # dataAug.all(origin_path, target, da.Blur())
    # dataAug.all2('mmm')
    # dataAug.all_sequential(origin_path, target, [da.Blur(), da.RotateRandom()], oper_desc='aaa')
    #


    # If you want to delete the generated images uncomment the following line
    # dataAug.delete_augmented_files()