import numpy as np
import random
import os

vid_list_file = "D:/Projects/Transformer/Transformer/breakfast/splits/train.split1.bundle"
vid_list_file_tst = "D:/Projects/Transformer/Transformer/breakfast/splits/test.split1.bundle"

gt_path = "D:/Projects/Transformer/Transformer/breakfast/groundTruth/"

arrays= "D:/Projects/Transformer/Transformer/breakfast/predictions/"


train_pth = "D:/Projects/Transformer/Transformer/breakfast/train"
test_pth = "D:/Projects/Transformer/Transformer/breakfast/test"

mapping_file = "D:/Projects/Transformer/Transformer/breakfast/mapping.txt"

file_ptr = open(mapping_file, 'r') 
actions = file_ptr.read().split('\n')[:-1]
actions_dict=dict()
for a in actions:
    actions_dict[a.split()[1]] = (int(a.split()[0]))

SOS_index = actions_dict['SOS']
EOS_index = actions_dict['EOS']

file_ptr = open(vid_list_file, 'r')
list_of_vid = file_ptr.read().split('\n')[:-1]
file_ptr.close()

def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    for i in range(len(content)):
        if content[i] != content[start]:
            label_seq.append(content[start])
            length_seq.append(i-start)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content)-start)
    
    return label_seq, length_seq




    

n_iterations = 3
nClasses = len(actions_dict)
max_seq_sz = 25
alpha = 6


def read_data(list_of_videos):
    list_of_examples = []
    for vid in list_of_videos:
            
        file_ptr = open(gt_path +vid, 'r')
        content = file_ptr.read().split('\n')[:-1]
        
        label_seq, length_seq = get_label_length_seq(content) 
        T = len(content)
    
        for itr in range(n_iterations):
            #list of partial length of each label in the sequence
            rand_cuts = []
            for i in range(len(label_seq)-1):
                rand_cuts.append( int( length_seq[i] * float(itr+.5)/n_iterations  ) )
                
            for i in range(len(rand_cuts)):
                seq_len = i+1
                p_seq = []
                for j in range(seq_len):
                    p_seq.append(np.zeros((2)))
                    if j == seq_len-1:
                        p_seq[-1][-1] = rand_cuts[j]/T
                    else:
                        p_seq[-1][-1] = length_seq[j]/T
                    p_seq[-1][0] = actions_dict[label_seq[j]]
                    
                #for j in range(max_seq_sz - seq_len):
                #    p_seq.append(np.zeros((nClasses+1)))
                
                p_tar = []
                #target length

                tar_len = length_seq[i+1]/T
                #remaining length
                rem_len = (length_seq[i]-rand_cuts[i])/T
                #target action
                current_act= actions_dict[label_seq[i]]
            
                p_tar=[[20,0],[current_act,rem_len]]
                for k in range(i+1,len(label_seq)):
                    p_tar.append([actions_dict[label_seq[k]],length_seq[k]/T])
                p_tar.append([0,0])
                
                
                example = [p_seq, p_tar, seq_len]
                
                list_of_examples.append(example)
                
    random.shuffle(list_of_examples) 
    return list_of_examples

read_data(list_of_vid)

index = 0

def next_batch(batch_size,list_of_examples,index):
    batch = np.array(sorted(list_of_examples[index:index+batch_size], key=lambda x: x[2], reverse=True) )
    index += batch_size
    batch_vid = list(batch[:,0])
    batch_target = list(batch[:,1])
            
    return batch_vid, batch_target