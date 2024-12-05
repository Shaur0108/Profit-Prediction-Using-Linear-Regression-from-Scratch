import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.linalg import eigh, norm
import os

img = Image.open('Python/Assets/Linear_Algebra.jpg')
img_grey = img.convert('LA')

matrix = np.array(list(img_grey.getdata(band=0)),float)
matrix = matrix.reshape(img_grey.size[1],img_grey.size[0])
img_matrix = np.matrix(matrix)
   
U,S,V = np.linalg.svd(img_matrix)

term_number = int(input("Enter number: "))

reconstruct_image = np.matmul(np.matrix(U[:,:term_number]),np.matmul(np.diag(S[:term_number]),np.matrix(V[:term_number,:])))
reconstructed_save = reconstruct_image.astype(np.uint8)
output_directory = 'Python/Assets'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_path = os.path.join(output_directory, "Reconstructed_Image.jpg")
reconstructed_image_pil = Image.fromarray(reconstructed_save)
reconstructed_image_pil.save(output_path)
