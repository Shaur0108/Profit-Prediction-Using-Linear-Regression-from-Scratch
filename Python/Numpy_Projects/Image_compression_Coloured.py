import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.linalg import svd


img = Image.open('Python/Assets/Linear_Algebra.jpg')
img_rgb = img.convert('RGB')
img_array = np.array(img_rgb)
R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]


U_R, S_R, V_R = svd(R, full_matrices=False)
U_G, S_G, V_G = svd(G, full_matrices=False)
U_B, S_B, V_B = svd(B, full_matrices=False)

term_number = int(input("Enter number: "))

R_reconstructed = np.matmul(U_R[:, :term_number], np.matmul(np.diag(S_R[:term_number]), V_R[:term_number, :]))
G_reconstructed = np.matmul(U_G[:, :term_number], np.matmul(np.diag(S_G[:term_number]), V_G[:term_number, :]))
B_reconstructed = np.matmul(U_B[:, :term_number], np.matmul(np.diag(S_B[:term_number]), V_B[:term_number, :]))


img_reconstructed = np.stack((R_reconstructed, G_reconstructed, B_reconstructed), axis=-1)
img_reconstructed = np.clip(img_reconstructed, 0, 255).astype(np.uint8)
img_reconstructed_pil = Image.fromarray(img_reconstructed)


output_directory = 'Python/Assets/Reconstructed_Colored_Image.jpg'
img_reconstructed_pil.save(output_directory)

print(f"Reconstructed image saved to {output_directory}")
