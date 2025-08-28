import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


vid = cv2.VideoCapture("walk_cycle.mp4")


#currentFrame = 0 
sequence_num = 8

list_frames=[]
gray_images_list = []
r=55


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


U_count = 0
U_sum = 0.0
U_sumsq = 0.0

sing_count = 0
sing_sum = 0.0
sing_sumsq = 0.0

Vt_count = 0
Vt_sum = 0.0
Vt_sumsq = 0.0

eps = 1e-8
U_blocks, S_blocks, Vt_blocks = [], [], []

for gray_image in gray_images_list:
    
    img_matrix = np.array(gray_image)
    U, S, Vt =np.linalg.svd(img_matrix)
    U_r = U[:, :r]
    S_r = np.diag(S[:r]) #gives diagonal but 2 x 2 with other values as 0 
    Vt_r = Vt[:r, :]
    
    rank_estimate_lst.append(U_r @ S_r @ Vt_r)
    flattenU_r = U_r.flatten()
    flatten_sing_values= S[:r]       
    flattenVt = Vt_r.flatten()

    single_U_sum=flattenU_r.sum()
    U_count+=flattenU_r.size
    U_sum += single_U_sum
    U_sumsq += (flattenU_r**2).sum()

    single_sing_sum=flatten_sing_values.sum()
    sing_count+=flatten_sing_values.size
    sing_sum += single_sing_sum
    sing_sumsq += (flatten_sing_values**2).sum()
    
    single_Vt_sum=flattenVt.sum()
    Vt_count+=flattenVt.size
    Vt_sum += single_Vt_sum
    Vt_sumsq += (flattenVt**2).sum()

    U_blocks.append(flattenU_r)
    S_blocks.append(flatten_sing_values)
    Vt_blocks.append(flattenVt)

U_mean = (U_sum)/(U_count) 
U_var_sample = (U_sumsq - U_count * (U_mean ** 2)) / (U_count - 1)
U_std_sample = np.sqrt(max(U_var_sample, 0.0)) + eps

sing_mean = (sing_sum)/(sing_count) 
sing_var_sample = (sing_sumsq - sing_count * (sing_mean ** 2)) / (sing_count - 1)
sing_std_sample = np.sqrt(max(sing_var_sample, 0.0)) + eps

Vt_mean = (Vt_sum)/(Vt_count) 
Vt_var_sample = (Vt_sumsq - Vt_count * (Vt_mean ** 2)) / (Vt_count - 1)
Vt_std_sample = np.sqrt(max(Vt_var_sample, 0.0)) + eps

# U_normalized = (flattenU_r-U_mean)/(U_std_sample)
# Sing_normalized = (flatten_sing_values-sing_mean)/(sing_std_sample)
# Vt_normalized = (flattenVt-Vt_mean)/(Vt_std_sample)



frames_list = []
for i in range(len(U_blocks)):
    Uf, Sf, Vtf = U_blocks[i], S_blocks[i], Vt_blocks[i]
    U_norm  = (Uf  - U_mean)  / U_std_sample
    S_norm  = (Sf  - sing_mean)  / sing_std_sample
    Vt_norm = (Vtf - Vt_mean) / Vt_std_sample
    frames_list.append(np.concatenate([U_norm, S_norm, Vt_norm]))


    
# print("frame_list",frames_list)

# print(frames_list[0])      


#-------------------------
#window func is context 

window_list = [] 
start = 0
end = sequence_num

for i in range(len(frames_list)-sequence_num):
    window_list.append(frames_list[start:end]) #window list full of multiple lists that have 8 frames in each list
    start+=1
    end+=1
    #end+=sequence_num
#print(window_list)
X=[]
y=[] #target
hidden_size = 128 #number of neurons 
input_size = 129 * r
# print(X.shape)
for i in range(len(window_list)):
    X.append(window_list[i])
    y.append(frames_list[i+sequence_num])   #1->8 predicts 9 2->9 predicts 10




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
bias_og = np.ones((hidden_size,))

predictions = []

#-----------------
#initalize weights and biases- logits 
weights_dl = np.random.uniform(-0.036, 0.036, size=(hidden_size, input_size))
biases_dl = np.zeros((input_size,))
learning_rate = 0.01

def stable_sigmoid(z):
    return np.where(z >= 0, 1.0/(1.0+np.exp(-z)), np.exp(z)/(1.0+np.exp(z)))

X_val_graph = []
Y_val_graph = []

for epoch in range(30):
    squared_list = []
    print("Epoch", epoch)
    # for input_sequence in X_np:  
    for seq_idx, input_sequence in enumerate(X_np):
        long_term = np.zeros(hidden_size)
        short_term = np.zeros(hidden_size)
        for vector_input in input_sequence:

            short_prev = short_term.copy()
            long_prev  = long_term.copy()

            #forget
            z_f        = (short_prev @ weight1_f) + (weight2_f @ vector_input) + bias   
            forget_gate = stable_sigmoid(z_f)
            forget_value = forget_gate * long_prev

            #input
            z_i        = (short_prev @ short_term_weight_ig) + (input_weight_ig @ vector_input) + bias_ig
            input_gate  = stable_sigmoid(z_i)  

            #pot memory
            z_g        = (short_prev @ short_term_weight_pm) + (input_weight_pm @ vector_input) + bias_pm
            pot_memory  = np.tanh(z_g)

            #long term 
            long_term   = forget_gate * long_prev + input_gate * pot_memory

            #output 
            z_o         = (short_prev @ short_term_weight_og) + (input_weight_og @ vector_input) + bias_og
            output_gate  = stable_sigmoid(z_o) 

            # short term
            short_term   = np.tanh(long_term) * output_gate
        
        # print(short_term.shape)
        # print(weights_dl.shape)
        # print(biases_dl.shape)
        logits = (short_term @ weights_dl) + biases_dl #logits = raw predictions
        # predictions.append(logits)

        # print("Logits", logits) #logits contains prediction for 1 sequence


        # predictions = np.array(predictions)  
        y_np = np.array(y).squeeze()  
        
        # for i in range(len(predictions)):
        squared_list.append(np.square(y[seq_idx]-logits)) #MSE loss formula 
        error = np.sum(squared_list)/len(squared_list)
        print("Error: ", error)
        target = y[seq_idx]                 # shape (7095,)
        dloss_dlogits = (2.0 / target.size) * (logits - target) 
        if epoch % 2 == 0:
            X_val_graph.append(epoch)
            Y_val_graph.append(error)

        #Gate backpropogation-----------------------------------------------------------------

        #short term was common when passed back so we take w/r/t to short term which was in logits and before statement including long term 
        dlogits_dshortterm  = weights_dl @ dloss_dlogits
        dshortterm_dlongterm = dlogits_dshortterm * output_gate * (1 - np.tanh(long_term)**2)

        # grads wrt forget gate
        dloss_dy_coord = dshortterm_dlongterm * long_prev          
        dycord_dz      = forget_gate * (1 - forget_gate)           
        dz_f           = dloss_dy_coord * dycord_dz

        d_weight1_f = np.outer(short_prev, dz_f)
        d_weight2_f = np.outer(dz_f, vector_input)
        d_bias_f    = dz_f

   

        #Input Gate:

        dlogits_dlogits = weights_dl @ dloss_dlogits
        dlogits_dshortterm = dlogits_dlogits * output_gate * (1 - np.tanh(long_term)**2)
        dshortterm_dproduct = dlogits_dshortterm #just 1
        dproduct_dinputgate = dshortterm_dproduct * pot_memory
        dinput_gateclipped = dproduct_dinputgate * input_gate * (1 - input_gate)
        d_short_term_weight_ig = np.outer(short_prev, dinput_gateclipped)
        d_input_weight_ig = np.outer(dinput_gateclipped, vector_input) 
        d_bias_ig = dinput_gateclipped                         

        #potential memory part 
        dproduct_dpotmemory = dshortterm_dproduct * input_gate
        g_zg = dproduct_dpotmemory * (1 - pot_memory**2)        # dL/dz_g

        d_short_term_weight_pm = np.outer(short_prev, g_zg)  # (H,H)
        d_input_weight_pm      = np.outer(g_zg, vector_input)     # (H,D)
        d_bias_pm              = g_zg                             # (H,)



        # upstream from logits to short_term
        dlogits_dlogits    = weights_dl @ dloss_dlogits          
        tanh_c             = np.tanh(long_term)

        # grad wrt output_gate
        dshortterm_dout    = dlogits_dlogits * tanh_c            
        g_z_o              = dshortterm_dout * output_gate * (1 - output_gate)


        d_short_term_weight_og = np.outer(short_prev, g_z_o)
        d_input_weight_og      = np.outer(g_z_o, vector_input)
        d_bias_og              = g_z_o


        dloss_logits = 2 * (logits - y[seq_idx]) / logits.size
        dlogits_dweights = short_term
        # print("dloss_logits",dloss_logits)
        # print("dlogits_dweights",dlogits_dweights)
        dloss_dweights = np.outer(dloss_logits, dlogits_dweights)
        dlogits_dbiases = dloss_logits


        gate_lr = 0.002   
        head_lr = 0.02  #added two LR as it helped improve the loss drastically, before the imrpovement was very small

        # gates/cell
        weight1_f          -= gate_lr * d_weight1_f
        weight2_f          -= gate_lr * d_weight2_f
        bias               -= gate_lr * d_bias_f

        short_term_weight_ig -= gate_lr * d_short_term_weight_ig
        input_weight_ig      -= gate_lr * d_input_weight_ig
        bias_ig              -= gate_lr * d_bias_ig

        short_term_weight_pm -= gate_lr * d_short_term_weight_pm
        input_weight_pm      -= gate_lr * d_input_weight_pm
        bias_pm              -= gate_lr * d_bias_pm

        short_term_weight_og -= gate_lr * d_short_term_weight_og
        input_weight_og      -= gate_lr * d_input_weight_og
        bias_og              -= gate_lr * d_bias_og

        # head
        weights_dl          -= head_lr * dloss_dweights.T
        biases_dl           -= head_lr * dlogits_dbiases


        




#plots loss on graph
def animate(i):
    plt.cla()
    plt.plot(X_val_graph, Y_val_graph)

    # biases_dl = biases_dl  - (learning_rate * dlogits_dbiases)

ani = FuncAnimation(plt.gcf(), animate, interval = 1000)
plt.show()


H, W, r = 64, 64, 55
u_len, s_len, vt_len = H*r, r, r*W
total = u_len + s_len + vt_len
if logits.shape[-1] != total:
    raise ValueError(f"Expected logits length {total}, got {logits.shape[-1]}")

U_norm  = logits[:u_len]
S_norm  = logits[u_len:u_len+s_len]
Vt_norm = logits[u_len+s_len:]

U_denorm_1d  = U_norm  * U_std_sample   + U_mean
S_denorm_1d  = S_norm  * sing_std_sample + sing_mean
Vt_denorm_1d = Vt_norm * Vt_std_sample   + Vt_mean

U_mat  = U_denorm_1d.reshape(H, r)
S_vec  = np.maximum(0.0, S_denorm_1d)    
Vt_mat = Vt_denorm_1d.reshape(r, W)

reconstructed = (U_mat * S_vec) @ Vt_mat
pred_image    = np.clip(reconstructed, 0, 255).astype(np.uint8)

plt.figure()
plt.imshow(pred_image, cmap='gray') 
plt.title("Predicted Frame") 
plt.axis('off') 
plt.show() 
