
# 
# EE596 Lab 1
# E/18/023
# 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict

x = 0*60
y = 23*4


#Quantizing the matrix
def quantize_matrix(array):
    max=np.max(array)
    min=np.min(array)

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

    return np.array(array)

class Node:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

    
def calculate_probabilities(data):
    unique_values, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return zip(unique_values, probabilities)

def build_huffman_tree(probabilities):
    heap = [Node(value, freq) for value, freq in probabilities]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def generate_huffman_codes(node, code="", mapping=None):
    if mapping is None:
        mapping = {}

    if node is not None:
        if node.value is not None:
            mapping[node.value] = code
        generate_huffman_codes(node.left, code + "0", mapping)
        generate_huffman_codes(node.right, code + "1", mapping)

    return mapping

def huffman_encode(data, huffman_mapping):
    encoded_data = "".join(huffman_mapping[value] for value in data.flatten())
    return encoded_data

def huffman_decode(encoded_data, huffman_tree, original_shape):
    decoded_data = []
    current_node = huffman_tree

    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.value is not None:
            decoded_data.append(current_node.value)
            current_node = huffman_tree

    return np.array(decoded_data).reshape(original_shape)


def main_process(image,description):
     
    #quantize matrix 
    quantized_img = quantize_matrix(image)
    print(description +" : Quantized Matrix")
    print(image)

    # Calculate probabilities
    probabilities = list(calculate_probabilities(quantized_img))
    print("Probability of each symbol distribution:")
    print(probabilities)

    # Build Huffman tree
    huffman_tree = build_huffman_tree(probabilities)

    # Generate Huffman codes
    huffman_mapping = generate_huffman_codes(huffman_tree)

    print("Huffman Mapping:")
    print(huffman_mapping)

    # Encode each color channel using Huffman codes
    encoded_data = huffman_encode(quantized_img, huffman_mapping)

    print( description +" : Encoded Data")
    print(encoded_data) 

    return encoded_data


def decode_ready(data,image):
    #quantize matrix 
    quantized_img = quantize_matrix(image)

    # Calculate probabilities
    probabilities = list(calculate_probabilities(quantized_img))

    # Build Huffman tree
    huffman_tree = build_huffman_tree(probabilities)

    return huffman_decode(data,huffman_tree,quantized_img.shape)


###################################################################################



image = cv2.imread("pattern.jpg") # Image Read

# Select a 16x16 cropped sub-image

cropped_image = image[y:y+16, x:x+16]
cropped_image[8,8] = (238,238,238)
print("Cropped_image")
print(cropped_image)

# Specify the color channel to quantize (0 for Blue, 1 for Green, 2 for Red)
red_filter_image= cropped_image[:, :, 2]
green_filter_image= cropped_image[:, :, 1]
blue_filter_image= cropped_image[:, :, 0]

# Encode each color channel using Huffman codes
red_encoded_data = main_process(red_filter_image,'E/18/023 Red Cropped image')
green_encoded_data = main_process(green_filter_image,'E/18/023 Red Cropped image')
blue_encoded_data = main_process(blue_filter_image,'E/18/023 Red Cropped image')

print("Red Encoded Data:")
print(red_encoded_data)

print("Green Encoded Data:")
print(green_encoded_data)

print("Blue Encoded Data:")
print(blue_encoded_data)

# Extract and quantize the R,G,B channels in original image
red_filter_fullimage= image[:, :, 2]
green_filter_fullimage= image[:, :, 1]
blue_filter_fullimage= image[:, :, 0]

# Encode each color channel using Huffman codes
red_fullencoded_data = main_process(red_filter_fullimage,'E/18/023 Red Full image')
green_fullencoded_data = main_process(green_filter_fullimage,'E/18/023 Red Full image')
blue_fullencoded_data = main_process(blue_filter_fullimage,'E/18/023 Red Full image')

print("Red Encoded Data in Full Image:")
print(red_fullencoded_data)
print("Green Encoded Data in Full Image:")
print(green_fullencoded_data)
print("Blue Encoded Data in Full Image:")
print(blue_fullencoded_data)

# Specify the path for saving the compressed data
output_file_path = "compressed_data.txt"

# Save the encoded data to a text file
with open(output_file_path, "w") as file:
    file.write("Red Encoded Data in Full Image:\n")
    file.write(red_fullencoded_data + "\n\n")

    file.write("Green Encoded Data in Full Image:\n")
    file.write(green_fullencoded_data + "\n\n")

    file.write("Blue Encoded Data in Full Image:\n")
    file.write(blue_fullencoded_data + "\n\n")

print(f"Compressed data saved to {output_file_path}")

# Decode each color channel using Huffman codes
red_decoded_data = decode_ready(red_fullencoded_data, red_filter_fullimage)
green_decoded_data = decode_ready(green_fullencoded_data, green_filter_fullimage)
blue_decoded_data = decode_ready(blue_fullencoded_data, blue_filter_fullimage)


print("Red Decoded Data in Full Image:")
print(red_decoded_data)

print("Green Decoded Data in Full Image:")
print(green_decoded_data)

print("Blue Decoded Data in Full Image:")
print(blue_decoded_data)

# Concatenate the decoded matrices along the third axis
decoded_image = np.stack([blue_decoded_data, green_decoded_data, red_decoded_data], axis=-1)

# Display the complete decoded image
cv2.imshow("Decoded Image", decoded_image.astype(np.uint8))

# Display the original and cropped images (optional)
cv2.imshow("Original Image", image)
cv2.imshow("Cropped Image", cropped_image)
cv2.imshow("Red Quantized Image", quantize_matrix(red_filter_image).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

