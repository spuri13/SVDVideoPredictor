import cv2
import numpy as np
import matplotlib.pyplot as plt



vid = cv2.VideoCapture("walk_cycle.mp4")


#currentFrame = 0 
sequence_num = 8

list_frames=[]
gray_images_list = []
r = 5
rank_estimate_lst = []
frames_list = []

while (True):
    success, frame = vid.read()
    if success:
        list_frames.append(frame)
            #print("in")
    if not success:
        print("There are no more frames")
        break
        
for image_rgb in list_frames:
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.resize(image_gray, (64, 64))  
    gray_images_list.append(image_gray)


for gray_image in gray_images_list:
    img_matrix = np.array(gray_image)
    U, S, Vt =np.linalg.svd(img_matrix)
    U_r = U[:, :r]
    S_r = np.diag(S[:r]) #gives diagonal but 2 x 2 with other values as 0 
    Vt_r = Vt[:r, :]
    
    rank_estimate_lst.append(U_r @ S_r @ Vt_r)
    flattenU_r = U_r.flatten()
    flatten_sing_values=np.diag(S_r)
    flattenVt = Vt.flatten()
    # print("U", flattenU_r.shape)
    # print("S", flatten_sing_values.shape)
    # print("Vt", flattenVt.shape)
    frame_vector = np.concatenate([flattenU_r, flatten_sing_values, flattenVt])
    frames_list.append(frame_vector)# frames list contains multiple frame vectors

# print("frame_list",frames_list)

# print(frames_list[0])      


#-------------------------
#window func is context 

window_list = []
start = 0
end = sequence_num

for i in range(len(frames_list)-sequence_num):
    window_list.append(frames_list[start:end])
    start+=1
    end+=1
    #end+=sequence_num
#print(window_list)
X=[]
y=[] #target
hidden_size = 128 #number of neurons 
input_size = 4421
# print(X.shape)
for i in range(len(window_list)):
    X.append(window_list[i])
    y.append([frames_list[i+sequence_num]]) #1->8 predicts 9 2-> predicts 10




#-------------------------------------------------------------------------------
# Forget Gate
 
X_np = np.array(X) 
weight1_f = np.random.uniform(-0.1, 0.1, size=(128, 128))
weight2_f= np.random.uniform(-0.1, 0.1, size=(hidden_size, input_size))
bias = np.ones((hidden_size,))


#-----------------------------------------------------------------------
#Input Gate

input_weight_ig = np.random.uniform(-0.1, 0.1, size=(hidden_size, input_size))
short_term_weight_ig = np.random.uniform(-0.1, 0.1, size=(hidden_size, hidden_size))
bias_ig = np.ones((hidden_size,))



#--------------------------------------------------------------------------
#Potential Memory
input_weight_pm = np.random.uniform(-0.1, 0.1, size=(hidden_size, input_size))
short_term_weight_pm = np.random.uniform(-0.1, 0.1, size=(hidden_size, hidden_size))
bias_pm = np.ones((hidden_size,))


#---------------------------------------------------------------------------
#Ouput Gate

input_weight_og = np.random.uniform(-0.1, 0.1, size=(hidden_size, input_size))
short_term_weight_og = np.random.uniform(-0.1, 0.1, size=(hidden_size, hidden_size))
bias_ig = np.ones((hidden_size,))

predictions = []

for input_sequence in X_np:  
    long_term = np.zeros(hidden_size)
    short_term = np.zeros(hidden_size)
    for vector_input in input_sequence:
        #forget gate
        # y_coord = 1 / (1 + np.exp(-((short_term @ weight1_f) + (weight2_f @ vector_input) + bias)))
        
        z = (short_term @ weight1_f) + (weight2_f @ vector_input) + bias
        z = np.clip(z, -50, 50)  #.clip limits values in list to specified range
        y_coord = 1 / (1 + np.exp(-z))

        forget_value = y_coord * long_term

        # print(vector_input.shape)
        # print(input_weight_ig.shape)
        

        #input gate + pot memory 
        input__gate_var = (short_term @ short_term_weight_ig) + (input_weight_ig @vector_input) + bias
        input__gate_clipped = np.clip(input__gate_var, -50, 50)
        input_gate = 1 / (1 + np.exp(-input__gate_clipped))

        #print(input_gate)
        pot_memory = np.tanh((short_term @ short_term_weight_pm) + (input_weight_pm @ vector_input) + bias_pm)
        product = input_gate * pot_memory #takes only some of the potential memory


        #Long term update
        long_term = forget_value + product #keep whats relavent, add new, throws out rest


        #Output gate, updates short_term 
        output_gate_var = (short_term @ short_term_weight_og) + (input_weight_og @vector_input) + bias
        output_gate_clipped = np.clip(output_gate_var, -50, 50)
        output_gate = 1 / (1 + np.exp(-output_gate_clipped))
        # output_gate = 1/(1 + np.exp(-((short_term @ short_term_weight_og) + (input_weight_og @vector_input) + bias)))
        short_term = np.tanh(long_term) * output_gate

        # print(short_term)
        # print(short_term)


        # print(input_gate)

# def dense(inputs, weights, biases):
print(short_term)
weights_dl = np.random.uniform(-0.036, 0.036, size=(hidden_size, input_size))
biases_dl = np.zeros((input_size,))
# print(short_term.shape)
# print(weights_dl.shape)
# print(biases_dl.shape)
logits = (short_term @ weights_dl) + biases_dl
predictions.append(logits)

# print("Logits", logits) #logits contains prediction for 1 sequence


predictions = np.array(predictions)  
y_np = np.array(y).squeeze()  


squared_list = []
for i in range(len(predictions)):
    squared_list.append(np.square(y[i]-predictions[i]))
print("Error: ", np.sum(squared_list)/len(squared_list))


# print(rank_estimate_lst)

# for rank_img in rank_estimate_lst:
#     plt.imshow(rank_img)
#     plt.show()

    
