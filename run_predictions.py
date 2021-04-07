import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

####### FIRST GET IMAGE TEMPLATE
# set the path to the downloaded data: 
#data_path = 'C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw1_data/RedLights2011_Medium' #'../data/RedLights2011_Medium'

# get sorted list of files: 
#file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
#file_names = [f for f in file_names if '.jpg' in f] 

# read image using PIL:
#I = Image.open(os.path.join(data_path,file_names[0]))

#get image size for reference
#I_size = I.size


#resize image to just contain red light
#size = (70,190)
#box = (316,153,323,172)
#resize image
#template = I.resize(size, box=box)
#template.show()
#template.save("red_light_template.jpg")
#######

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    #dpath = 'C:/Users/kbdra/OneDrive/Documents/GitHub/caltech-ee148-spring2020-hw01/red_light_inputs'
    # get sorted list of input examples: 
    # used to compare and find redlights
    #input_names = sorted(os.listdir(dpath)) 

    # remove any non-JPEG files: 
    #input_names = [f for f in file_names if '.jpg' in f]
    #templates = []
    #for i in range(len(input_names)):
        #templates[i] = Image.open(os.path.join(dpath,input_names[i]))
    template = Image.open('C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw1_data/red_light_inputs/red_light_temp_crop5.jpg')
    temp_arr = np.asarray(template)
    #normalize between -1 and 1
    temp = temp_arr

    #print(np.shape(temp))
    
    #I_back = I
    #I = I/(max(I/2) -1
    
    scan_step = 10
    dim = np.shape(I[:, :, 0])
    
    temp_vec = np.reshape(temp, (-1,))
    temp_vec = temp_vec/(max(temp_vec)/2) -1
    
    best = np.dot(temp_vec, temp_vec)
    
    # grab template size section from image convolve with template
    # box format (row0, col0, row1, col1)
    #scan across columns
    boxes = []
    column = 0
    i = 0
    dots = []
    while column < (dim[1]-len(temp_arr[0,:,0])):
        # scan across rows
        row = 0
        while row < (dim[0] - + len(temp_arr[:,0,0])):
            box = [row, column, row + len(temp_arr[:,0,0]) , column + len(temp_arr[0,:,0])]
            sample = I[row:(row + len(temp_arr[:,0,0])), column:(column + len(temp_arr[0,:,0])), :]
            sample_vec = np.reshape(sample, (-1,))
            sample_vec = sample_vec/(max(sample_vec)/2) -1
            #print(np.shape(sample_vec))
            #print(np.shape(temp_vec))
            dot = np.dot(sample_vec, temp_vec)
            dots.append(dot)
            #print(dot)
            #if (dot > 82): #and (dot < 1100)):
            boxes.append(box)
            row += scan_step
        column += scan_step 
        
    for i in range(len(dots)):
        if (dots[i] > np.percentile(dots, 99.9)):
        #if ((dots[i] > (best - 10)) or (dots[i] < (best + 10))):
            bounding_boxes.append(boxes[i])
    dots = np.array(dots)
    top_box = np.percentile(dots, 99.9)
    #print(top_box)
    #print(dots)
    #print(max(dots))
  
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    
    #box_height = 8
    #box_width = 6
    
    #num_boxes = np.random.randint(1,5) 
    
    #for i in range(num_boxes):
        #(n_rows,n_cols,n_channels) = np.shape(I)
        
        #tl_row = np.random.randint(n_rows - box_height)
        #tl_col = np.random.randint(n_cols - box_width)
        #br_row = tl_row + box_height
        #br_col = tl_col + box_width
        
        #bounding_boxes.append([tl_row,tl_col,br_row,br_col]) 
    
    '''
    END YOUR CODE
    '''
    
    #for i in range(len(bounding_boxes)):
        #assert len(bounding_boxes[i]) == 4


    return bounding_boxes

# set the path to the downloaded data: 
data_path = 'C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw1_data/RedLights2011_Medium' #'../data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = 'C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw1_data/hw01_preds' #'../data/hw01_preds'
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)
    
################### visualize ############################
    plt.imshow(I)
    boxes = preds[file_names[i]]
    #print(boxes)
    for box in boxes:
        plt.gca().add_patch(Rectangle((box[1], box[0]), 26,49, fill = False, color="purple",
                       linewidth=2))
    plt.show()
    
 ###########################################################   
    
    
    #for box in preds[file_names[i]]:
        #Image.Draw(I).rectangle([box[1],box[0],box[3], box[2]])

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
