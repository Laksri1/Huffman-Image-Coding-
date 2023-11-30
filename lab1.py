# 
# EE596 Lab 1
# E/18/023
# 

import numpy as np
import cv2

x = 0*60
y = 23*4


# Quantizer --------------------------------------------------------
def quantiser(array):

    max=np.max(red_array)
    min=np.min(red_array)

    q = (max-min)/8

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
            elif array[i][j]>=min+15*q/2 and array[i][j]<min+16*q/2 :
                    array[i][j]= min + 8*q
    # print(array)

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

def huffman(sorted_dict,input_dict2):
    for key in input_dict2:
        input_dict2[key] = ""

    while (len(sorted_dict)>1):
        new_value = list(sorted_dict.values())[0] + list(sorted_dict.values())[1]
        key = str(list(sorted_dict.keys())[0])+"_"+str(list(sorted_dict.keys())[1])

        sorted_dict[key] = new_value

        result = str(list(sorted_dict.keys())[0]).split('_')

        for i in result:
            input_dict2[int(i)] = "0" + input_dict2[int(i)]

        sorted_dict.pop(list(sorted_dict.keys())[0])

        result = str(list(sorted_dict.keys())[0]).split('_')

        for i in result:
            input_dict2[int(i)] = "1"+input_dict2[int(i)]

        sorted_dict.pop(list(sorted_dict.keys())[0])

        # Step 4: Sort the dictionary again using bubblesort
        sorted_dict = sort(sorted_dict)

    return input_dict2

# ------------------------------------------------------------------

# compress ---------------------------------------------------------

def compress(arr,arr2,name):
    q=list(arr2.keys())[1]-list(arr2.keys())[0]

    # arr1=quantise(arr)

    hight=arr.shape[0]
    width=arr.shape[1]

    file_path = str(name)+".txt"
    with open(file_path, "w") as file:
        
        for i in range(hight):
            for j in range(width):
                for k in list(arr2.keys()):
                    if arr[i][j]>=k-q/2 and arr[i][j]<k+q/2:
                        # Write data to the file
                        file.write(arr2[k])
                        # file.write("")

                    elif arr[i][j]<11:
                        file.write("a")

    return 'done'

#-------------------------------------------------------------------


original_img = cv2.imread("pattern.jpg")
cropped_img = original_img[y:y+16, x:x+16]

blue_img = cropped_img[:,:,0]
green_img = cropped_img[:,:,1]
red_img = cropped_img[:,:,2]
gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) 

# blue_img,green_img,red_img = cv2.split(cropped_img)
# zero_matrix = np.zeros(blue_img.shape, np.uint8)
# blue_img = cv2.merge([blue_img,zero_matrix,zero_matrix])
# green_img = cv2.merge([zero_matrix,green_img,zero_matrix])
# red_img = cv2.merge([zero_matrix,zero_matrix,red_img])

cv2.imshow('origional',original_img)
cv2.imshow('cropped', red_img)

cv2.waitKey(0) 
cv2.destroyAllWindows()

red_array = np.array(red_img)
green_array = np.array(green_img)
blue_array = np.array(blue_img)
gray_array = np.array(gray_img)

quantised_img =quantiser(red_array)

img_probability = probability(quantised_img)
img_probability_for_huff = img_probability.copy()

sorted_img_prob = sort(img_probability)

print(img_probability_for_huff)
print(sorted_img_prob)

huf_img = huffman(sorted_img_prob,img_probability_for_huff)

print(huf_img)

# quantized_red = quantise(d)
 