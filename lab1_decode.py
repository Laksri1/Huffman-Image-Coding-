
import numpy as np


# Decoding -----------------------------------------------------------------

# def decode(c_img,cBook):



    # return ans

# --------------------------------------------------------------------------




file = open("cropped.txt", "r")

for i in file.read() :
    print(i)

codedImg = np.array(file.read())

# decodedImg = decode(codedImg,codeBook)

print(file.read())