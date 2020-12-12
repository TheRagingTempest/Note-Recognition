from flask import Flask,request,send_from_directory,render_template
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys
from pathlib import Path

#functions
def resize(image, height):
  width = int(float(height * image.shape[1]) / image.shape[0])
  sample_img = cv2.resize(image, (width, height))
  return sample_img

def normalize(image):
  return (255. - image)/255.

def sparse_tensor_to_strs(sparse_tensor):
  indices= sparse_tensor[0][0]
  values = sparse_tensor[0][1]
  dense_shape = sparse_tensor[0][2]

  strs = [ [] for i in range(dense_shape[0]) ]

  string = []
  ptr = 0
  b = 0

  for idx in range(len(indices)):
      if indices[idx][0] != b:
          strs[b] = string
          string = []
          b = indices[idx][0]

      string.append(values[ptr])

      ptr = ptr + 1

  strs[b] = string

  return strs

#def checkFile(filename):
#  if not filename:
#    print('Please enter a valid music note picture path.')
#    filename = input()
#  else:
#    chk_file = Path(filename)
#    if not chk_file.is_file():
#      print('Not a valid path.')
#      checkFile(None)
#    else:
#      return filename

#MAIN
#read note dictionary
#if len(sys.argv) != 2:
#  filename = checkFile(None)

#else: 
#  filename = checkFile(sys.argv[1])

#print("THIS IS THE FILENAME", filename)
#checkFile(filename)


voc_file = "vocabulary_semantic.txt"

dict_file = open(voc_file,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
  word_idx = len(int2word)
  int2word[word_idx] = word
dict_file.close()

#initialize tf
model = "Semantic-Model/semantic_model.meta"
tf.reset_default_graph()
sess = tf.InteractiveSession()

saver = tf.train.import_meta_graph(model)
saver.restore(sess,model[:-5])

graph = tf.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

#Opening Image File
f = open("Pic.png", "rb")
img = f
image = Image.open(img).convert('L')
image = np.array(image)
image = resize(image, HEIGHT)
image = normalize(image)
image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

#inference section
seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]
prediction = sess.run(decoded,
                    feed_dict={
                        input: image,
                        seq_len: seq_lengths,
                        rnn_keep_prob: 1.0,
                    })
str_predictions = sparse_tensor_to_strs(prediction)

array_of_notes = []

for w in str_predictions[0]:
    array_of_notes.append(int2word[w])
notes=[]
for i in array_of_notes:
    if i[0:5]=="note-":
        if not i[6].isdigit():
            notes.append(i[5:7])
        else:
            notes.append(i[5])
f = open("test1.txt", "w")
for i in notes:
    f.write(i)
    f.write("\n")

f.close()