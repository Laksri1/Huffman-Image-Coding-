# 
# EE596 Lab 1
# E/18/023
# 

import numpy as np
import cv2
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
from scipy.stats import entropy
from math import log10, sqrt




# Quantizer --------------------------------------------------------
def quantiser(array):

    # max=np.max(red_array)
    # min=np.min(red_array)
    max=255
    min=0

    # q = (max-min)/7

    q=36.4285714

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j]>=min and array[i][j]<min+q/2 :
                    array[i][j]= min
            elif array[i][j]>=min+q/2 and array[i][j]<min+3*q/2 :
                    array[i][j]= min + q
            elif array[i][j]>=min+3*q/2 and array[i][j]<min+5*q/2 :
                    array[i][j]= min + 2*q
            elif array[i][j]>=min+5*q/2 and array[i][j]<min+7*q/2 :
                    array[i][j]= min + 3*q
            elif array[i][j]>=min+7*q/2 and array[i][j]<min+9*q/2 :
                    array[i][j]= min + 4*q
            elif array[i][j]>=min+9*q/2 and array[i][j]<min+11*q/2 :
                    array[i][j]= min + 5*q
            elif array[i][j]>=min+11*q/2 and array[i][j]<min+13*q/2 :
                    array[i][j]= min + 6*q
            elif array[i][j]>=min+13*q/2 and array[i][j]<min+15*q/2 :
                    array[i][j]= min + 7*q

    return np.array(array)

# ------------------------------------------------------------------

# Probability ------------------------------------------------------

def probability(q_vals):
  
    unique_values, counts = np.unique(q_vals, return_counts=True)
    
    value_counts = dict(zip(unique_values, counts))

    for key in value_counts:
        value_counts[key]=value_counts[key]/(16*16)

    return value_counts

# ------------------------------------------------------------------

# Sort data --------------------------------------------------------

def sort(input_dict):
    items = list(input_dict.items())
    n = len(items)

    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if items[j][1] > items[j + 1][1]:
                items[j], items[j + 1] = items[j + 1], items[j]

    sorted_img = dict(items)
    
    return sorted_img

# ---------------------------------------------------------------------

# Huffman ----------------------------------------------------------

def huffman(sorted_dict,input_dict):
    for key in input_dict:
        input_dict[key] = ""

    while (len(sorted_dict)>1):
        new_value = list(sorted_dict.values())[0] + list(sorted_dict.values())[1]
        key = str(list(sorted_dict.keys())[0])+"_"+str(list(sorted_dict.keys())[1])

        sorted_dict[key] = new_value

        result = str(list(sorted_dict.keys())[0]).split('_')

        for i in result:
            input_dict[int(i)] = "0" + input_dict[int(i)]

        sorted_dict.pop(list(sorted_dict.keys())[0])

        result = str(list(sorted_dict.keys())[0]).split('_')

        for i in result:
            input_dict[int(i)] = "1"+input_dict[int(i)]

        sorted_dict.pop(list(sorted_dict.keys())[0])

        # Step 4: Sort the dictionary again using bubblesort
        sorted_dict = sort(sorted_dict)

    return input_dict

# ------------------------------------------------------------------

# compress ---------------------------------------------------------

def compress(arr,arr2,name):
    arr1=np.array(arr[:,:])

    q=36.4285714

    hight=arr1.shape[0]
    width=arr1.shape[1]

    file_path = str(name)+".txt"
    with open(file_path, "w") as file:
        count=0
        
        for i in range(hight):
            for j in range(width):
                for l in list(arr2.keys()):
                    if arr1[i][j]>=(l-q/2) and arr1[i][j]<(l+q/2):
                        
                        # Write data to the file
                        file.write(arr2[l])
                        count+=1
                        break

#-------------------------------------------------------------------

# Num concat ------------------------------------------------------------

def numConcat(num1, num2):
 
     # find number of digits in num2
     digits = len(str(num2))
 
     # add zeroes to the end of num1
     num1 = num1 * (10**digits)
 
     # add num2 to num1
     num1 += num2
 
     return num1

# --------------------------------------------------------------------------

# Decoding -----------------------------------------------------------------

def decode(c_img,cBook,hight,width,name):

    # decompresed the rgb img
    
    code=list(cBook.values())
    key=list(cBook.keys())

   
    f1=c_img.readline()
 
    arr19=np.zeros((hight,width, 3),dtype=np.uint8)

    count =0

    for i in range(hight):
            for j in range(width):
                count += 1
                print(count)
                s=""
                for x in f1:
                    s+=x
                    
                    if s in code:
                        index_of_element = code.index(s)
                        arr19[i][j]=key[index_of_element]
                        
                        if len(f1)>len(s):
                            f1=f1[len(s):]

                        s=""
                        break

    display = str(name)+" decoded image (E/18/023)"
    cv2.imshow(display, np.array(arr19))
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    print(arr19)
    return np.array(arr19)

# --------------------------------------------------------------------------

# entropy ------------------------------------------------------------------

def find_entropy(gray_image,name):
    

    _bins = 128

    hist, _ = np.histogram(gray_image.ravel(), bins=_bins, range=(0, _bins))

    prob_dist = hist / hist.sum()
    image_entropy = entropy(prob_dist, base=2)
    print(f"{name} Entropy {image_entropy}")


# --------------------------------------------------------------------------

# PSNR----------------------------------------------------------------------

def PSNR(original, compressed):
	mse = np.mean((original - compressed) ** 2)
	if(mse == 0): # MSE is zero means no noise is present in the signal .
				# Therefore PSNR have no importance.
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr

# --------------------------------------------------------------------------

original_img = cv2.imread("pattern.jpg")
original_img_array = np.array(original_img)
cv2.imshow('origional IMAGE (E/18/023)',original_img_array)
cv2.waitKey(0) 
cv2.destroyAllWindows()

x = 0*60
y = 23*4

cropped_img = original_img[y:y+16, x:x+16]

cv2.imshow('cropped image (E/18/023)',cropped_img)
cv2.waitKey(0) 
cv2.destroyAllWindows()


red_img = original_img[:,:,2]

cv2.imshow('origional red image (E/18/023)',red_img)
cv2.waitKey(0) 
cv2.destroyAllWindows()
cv2.imwrite('original_img.jpg',red_img) 

red_cropped_img = red_img[y:y+16, x:x+16]

cv2.imshow('cropped decoded image (E/18/023)',red_cropped_img)
cv2.waitKey(0) 
cv2.destroyAllWindows()
cv2.imwrite('cropped_img.jpg', red_cropped_img)

quantised_img =quantiser(red_cropped_img)

cv2.imshow('quantized red image (E/18/023)',quantised_img)
cv2.waitKey(0) 
cv2.destroyAllWindows()

img_probability = probability(quantised_img)
print('image probabilities :')
print(img_probability)

codeBook = img_probability.copy()

sorted_img_prob = sort(img_probability)

huf_img = huffman(sorted_img_prob,codeBook)

print('code book :')
print(codeBook)

# compress(red_cropped_img,codeBook,"cropped")
# compress(red_img,codeBook,"original")

# file = open("cropped.txt","r")
# decodedImg_cropped = decode(file,codeBook,16,16,'cropped')
# cv2.imwrite("cropped_decoded.jpg", decodedImg_cropped) 

# file = open("original.txt","r",)
# decodedImg = decode(file,codeBook,red_img.shape[0],red_img.shape[1],'original')
# cv2.imwrite("original_decoded.jpg", decodedImg)

file = open("huffman_builtin.txt","r",)
decodedImg_huff_bi = decode(file,codeBook,red_img.shape[0],red_img.shape[1],'decodedImg_huff_bi')
cv2.imwrite("decodedImg_huff_bi.jpg", decodedImg_huff_bi)