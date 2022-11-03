import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import string

batch_size = 50

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    print("READING DATA")
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data2/')):
        with tarfile.open("reviews.tar.gz", "r") as tarball:
            dir = os.path.dirname(__file__)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tarball, os.path.join(dir,"data2/"))
    
    dir = os.path.dirname(__file__)
    data = []
#    dir = os.path.dirname("")
    file_list = glob.glob(os.path.join(dir,
                                        'data2/pos/*'))
    file_list= file_list[:12500]
    file_list.extend(glob.glob(os.path.join(dir,
                                        'data2/neg/*')))
    file_list= file_list[:25000]
    print("Parsing %s files" % len(file_list))
    stop_words=('i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
                'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
                'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
                'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                'should', 'now')
    for f in file_list:
        with open(f, "r",encoding = 'utf-8') as openf:
            s = openf.read()
            no_punct = ''.join(c.lower() for c in s if c not in string.punctuation)         
#            np.append(data, no_punct.split())
        l = []
        for i in no_punct.split():
            if i not in stop_words:
                if i in glove_dict:
                    l.append(np.float32(glove_dict[i]))
                else:
                    l.append(np.float32(0))
        if len(l)<40:
            l.extend((40-len(l))*[np.float32(0)])
        else:
            l = l[:40]
        data.append(l)
#    for i in range(len(data)):
#        if len(data[i]) > 40:
#            data[i] = data[:40]
#        if len(data[i]) < 40:
#            data[i].append((40-len(data[i]))*[0])
    data = np.array(data,dtype=np.float32)
#    data.dtype = 'float64'
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
#    input_data = data.read()
    count= 1
#    index = 50
    word_index_dict ={}
    word_index_dict["UNK"]= 0
    line = []
    embeddings = [[]]

    for l in data:
        line = []
        for i in l.split(" ")[1:]:
            
            line.append(np.float32(i))
        embeddings.append(line)
        word_index_dict[l.split(" ")[0]]= count
        count+=1
    dimension_size = len(embeddings[1])
    embeddings[0] =  dimension_size*[np.float32(0)]
    embeddings = np.array(embeddings,dtype=np.float32)    
    
            
#    embeddings.dtype = 'float32'
#    embeddings = tf.cast(embeddings, tf.float32)
    return  embeddings,word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""
   
    tf.reset_default_graph()
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(),name="dropout_keep_prob")
    _,dimension_size = glove_embeddings_arr.shape
    Units_N = 32
    Class_N = 2
    Layer_N = 2
    
    labels = tf.placeholder(tf.float32, [batch_size, Class_N],name="labels")
    input_data = tf.placeholder(tf.int32, [batch_size, 40],name="input_data")
    data = tf.Variable(tf.zeros([batch_size, 40, dimension_size]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(glove_embeddings_arr,input_data)
    def lstm_cell():

        lstm = tf.contrib.rnn.BasicLSTMCell(Units_N, reuse=tf.get_variable_scope().reuse)
            
        return tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=dropout_keep_prob,output_keep_prob=dropout_keep_prob)

    
    basic_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(Layer_N)])

    
    initial_state = basic_cell.zero_state(batch_size, tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(basic_cell, data,
                                                 initial_state=initial_state) 

    weight = tf.Variable(tf.truncated_normal([Units_N, Class_N]))
    bias = tf.Variable(tf.constant(0.1, shape=[Class_N]))
    value = tf.transpose(outputs, [1, 0, 2])
    last_elem = tf.gather(value, int(value.get_shape()[0]) - 1)

    prediction = (tf.matmul(last_elem, weight) + bias)

    Prendiction_Correct = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(Prendiction_Correct, tf.float32) ,name="accuracy")
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name="loss")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
#    graph = tf.Graph()
#    with graph.as_default():
#    ix = tf.Variable(tf.truncated_normal([batch_size, 40], -0.1, 0.1))
#    im = tf.Variable(tf.truncated_normal([batch_size, 40], -0.1, 0.1))
#    ib = tf.Variable(tf.zeros([1, 40]))
#    fx = tf.Variable(tf.truncated_normal([batch_size, 40], -0.1, 0.1))
#    fm = tf.Variable(tf.truncated_normal([batch_size, 40], -0.1, 0.1))
#    fb = tf.Variable(tf.zeros([1, 40]))
#    cx = tf.Variable(tf.truncated_normal([batch_size, 40], -0.1, 0.1))
#    cm = tf.Variable(tf.truncated_normal([batch_size, 40], -0.1, 0.1))
#    cb = tf.Variable(tf.zeros([1, 40]))
#    ox = tf.Variable(tf.truncated_normal([batch_size, 40], -0.1, 0.1))
#    om = tf.Variable(tf.truncated_normal([batch_size, 40], -0.1, 0.1))
#    ob = tf.Variable(tf.zeros([1, 40]))
#    saved_output = tf.Variable(tf.zeros([batch_size, 40]), trainable=False)
#    saved_state = tf.Variable(tf.zeros([batch_size, 40]), trainable=False)
#  # Classifier weights and biases.
#    w = tf.Variable(tf.truncated_normal([40, batch_size], -0.1, 0.1))
#    b = tf.Variable(tf.zeros([batch_size]))
#    input_data =  tf.placeholder(tf.float32, [batch_size, 40])
#    labels = tf.placeholder(tf.int32, [batch_size, 40])
#    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
#    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
#    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
#    state = forget_gate * state + input_gate * tf.tanh(update)
#    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    return input_data, labels,dropout_keep_prob, optimizer, accuracy, loss
