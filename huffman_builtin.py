import huffman
import cv2
import numpy as np

img = cv2.imread("pattern.jpg")

def count_unique_values(data):
    unique_values, counts = np.unique(data, return_counts=True)
    
    value_counts = dict(zip(unique_values, counts))
    
    return value_counts

def quantise(img):

    hight=img.shape[0]
    width=img.shape[1]

    img=np.array(img)

    max=np.max(img)
    min=np.min(img)

    q=(max-min)/7

    while (min<=max):
        for h in range(hight):
            for w in range(width):
                for j in range(3):
                    if img[h][w][j]>=min-q/2 and img[h][w][j]<min+q/2 :
                        img[h][w][j]=min
                    elif img[h][w][j]>=min+q/2 and img[h][w][j]<min+q/2+q:
                        img[h][w][j]=min+q
                
        min=min+q
    return img

def compress(img,cb,name):
    arr1=np.array(img[:,:,2])

    q=36.4285714

    hight=arr1.shape[0]
    width=arr1.shape[1]

    file_path = str(name)+".txt"
    with open(file_path, "w") as file:
        count=0
        
        for i in range(hight):
            for j in range(width):
                for l in list(cb.keys()):
                    if arr1[i][j]>=(l-q/2) and arr1[i][j]<(l+q/2):
                        
                        file.write(cb[l])
                        count+=1
                        break



def decompress(file,cb,hight,width):

    code=list(cb.values())
    key=list(cb.keys())

    f = open(file, "r")
    f1=f.readline()
 
    arr19=np.zeros((hight,width, 3),dtype=np.uint8)

    for i in range(hight):
            for j in range(width):
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
                            
                            
                            
    cv2.imwrite(str(file)+'_decompresed_image.jpg', arr19)
    # cv2.imshow('Image_0', arr19)

def avg_len(arr,arr2):
    result_dict = {key: len(arr[key]) * arr2[key] for key in arr}
    # print(result_dict)
    count=sum(result_dict.values())
    print("avg length per code(one color) ",count)
    print("avg length for full croped img(rgb after compretion) ",count*256*3)

prob={0:"0.046266037",36:"0.049",}

img=quantise(img)
list_of_tuples = list(count_unique_values(img).items())
# print(list(count_unique_values(img).items()))
cb=huffman.codebook(list_of_tuples)

# print(cb)



compress(img,cb,"huffman_builtin")
decompress("huffman_builtin.txt",cb,img.shape[0],img.shape[1])
# print(a)

# Calculate the sum of values in the dictionary
sum_of_values = sum(count_unique_values(img).values())
print(sum_of_values)