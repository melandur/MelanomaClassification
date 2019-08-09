import os

# Set the paths to the folders
path1 = 'C:/Users/CorePy/Desktop/train1/'
path2 = 'C:/Users/CorePy/Desktop/test1/'

try:
     # Check Melanoma
     a = os.listdir(str(path1) + 'melanoma/')
     b = os.listdir(str(path2) + 'melanoma/')
     x = set(a) & set(b)
     if len(x) > 0:
          print("Folder melanoma contains images with the same name!")
          print(x)

     # Check Nevus
     a = os.listdir(str(path1) + 'nevus/')
     b = os.listdir(str(path2) + 'nevus/')
     x = set(a) & set(b)
     if len(x) > 0:
          print("Folder nevus contains images with the same name!")
          print(x)

     # Check Seborrheic_Keratosis
     a = os.listdir(str(path1) + 'seborrheic_keratosis/')
     b = os.listdir(str(path2) + 'seborrheic_keratosis/')
     x = set(a) & set(b)
     if len(x) > 0:
          print("Folder seborrheic_keratosis contains images with the same name!")
          print(x)

except:
     print("No contamination of train/test-set, go ahead")

