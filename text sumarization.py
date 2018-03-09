# Text summarization of Amazon reviews
#import libraries
import nltk as nt
import pandas as pd
import numpy as np
import tensorflow as tf
import string
import os
import re,
import sys
import requests
import io
from collections import Counter
from tensorflow.python.layers.core import Dense
import random
from nltk.corpus import stopwords
import time
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import seq2seq
import seq2seq_model
import data_utils
import translate
sess = tf.Session()
#setting learning parameters
VOCAB_THRESHOLD = 10
UCKETS = [(10,50),(20,100)] #First try buckets you can tweak these
EPOCHS = 20
BATCH_SIZE = 64
RNN_SIZE = 512
NUM_LAYERS = 3
ENCODING_EMBED_SIZE = 512
DECODING_EMBED_SIZE = 512
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.9 #nisam siguran da cu ovo koristiti
MIN_LEARNING_RATE = 0.0001
KEEP_PROBS = 0.5
CLIP_RATE = 4
tf.__version__
stopwords=pd.read_csv('english')
stopwords[:2]
reviews=pd.read_csv("Reviews.csv")
reviews.shape #returns a 1-D integer tensor representing the shape of input.
reviews.head(2)
# Remove null values and unneeded features
reviews = reviews.dropna()
reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time'], 1)
reviews = reviews.reset_index(drop=True)
reviews.head(2)
# Inspecting some of the reviews
for i in range(5):
print("Review",i+1)
print(reviews.Summary[i])
print(reviews.Text[i])
print()
def clean_text(text, remove_stopwords = True):
'''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
# Convert words to lower case
text = text.lower()
# Replace contractions with their longer forms 
if True:
text = text.split()
new_text = []
for word in text:
if word in contractions:
new_text.append(contractions[word])
else:
new_text.append(word)
text = \" \".join(new_text)
# Format words and remove unwanted characters
text = re.sub(r'https?:\\/\\/.*[\\r\\n]*', '', text, flags=re.MULTILINE)
text = re.sub(r'\\<a href', ' ', text)
text = re.sub(r'&amp;', '', text) 
text = re.sub(r'[_\"\\-;%()|+&=*%.,!?:#$@\\[\\]/]', ' ', text)
text = re.sub(r'<br />', ' ', text)
text = re.sub(r'\\'', ' ', text)
return text
# Clean the summaries and texts
clean_summaries = []
for summary in reviews.Summary:
clean_summaries.append(clean_text(summary, remove_stopwords=False))
print("Summaries are complete")
len(clean_summaries)
clean_texts = []
for text in reviews.Text:
clean_texts.append(clean_text(text))
print("Texts are complee")
len(clean_texts)
cs =clean_summaries[:500]
ct = clean_texts[:500]
 max=0
for x in range(len(cs)):
if (len(cs[x])>=max):
 max=len(cs[x])
 max
max=0,
for x in range(len(ct)):
if (len(ct[x])>=max):
max=len(ct[x])
max
ct[299]
# Inspect the cleaned summaries and texts to ensure they have been cleaned well
for i in range(5):
print("Clean Review",i+1)
print(cs[i])
print(ct[i])
print()
def create_vocab(ct,cs):
assert len(ct) == len(cs)
vocab = []\n",
for i in range(len(ct)):
              words = ct[i].split()
for word in words:
vocab.append(word)
   words = cs[i].split()
    for word in words:
    vocab.append(word)
    vocab = Counter(vocab)
    new_vocab = []
    for key in vocab.keys():
    if vocab[key] >= VOCAB_THRESHOLD:
    new_vocab.append(key)
    new_vocab = ['<PAD>', '<GO>', '<UNK>', '<EOS>'] + new_vocab
    word_to_id = {word:i for i, word in enumerate(new_vocab)}
    id_to_word = {i:word for i, word in enumerate(new_vocab)}
    return new_vocab, word_to_id, id_to_word
    def encoder_data(data, word_to_id, targets=False):
    encoded_data = [] 
  for i in range(len(data)):
    encoded_line = []
    twords = data[i].split()
    for word in words:
    if word not in word_to_id.keys():
   encoded_line.append(word_to_id['<UNK>'])
             encoded_line.append(word_to_id[word])
 if targets:
  encoded_line.append(word_to_id['<EOS>'])
   encoded_data.append(encoded_line)
  return np.array(encoded_data)
    def pad_data(data, word_to_id, max_len, target=False):
   if target:
  return data + [word_to_id['<PAD>']] * (max_len - len(data))
  else:
   return [word_to_id['<PAD>']] * (max_len - len(data)) + data
 def bucket_data(ct, cs, word_to_id):
 tassert len(ct) == len(cs)
bucketed_data = []
    already_added = []
    for bucket in BUCKETS:
    data_for_bucket = []
    encoder_max = bucket[0]
    decoder_max = bucket[1]
    for i in range(len(ct)):
    if len(ct[i]) <= encoder_max and len(cs[i]) <= decoder_max:
  if i not in already_added:
    data_for_bucket.append((pad_data(cs[i], word_to_id, encoder_max), pad_data(ct[i], word_to_id, decoder_max, True)))
    already_added.append(i)
  bucketed_data.append(data_for_bucket)
 return bucketed_data
 def grap_inputs():
   inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        keep_probs = tf.placeholder(tf.float32, name='dropout_rate')
 encoder_seq_len = tf.placeholder(tf.int32, (None, ), name='encoder_seq_len')
        decoder_seq_len = tf.placeholder(tf.int32, (None, ), name='decoder_seq_len')
        max_seq_len = tf.reduce_max(decoder_seq_len, name='max_seq_len')
   return inputs, targets, keep_probs, encoder_seq_len, decoder_seq_len, max_seq_len
    def encoder(inputs, rnn_size, number_of_layers, encoder_seq_len, keep_probs, encoder_embed_size, encoder_vocab_size):
        def cell(units, rate):
            layer = tf.contrib.rnn.BasicLSTMCell(units)
            return tf.contrib.rnn.DropoutWrapper(layer, rate)
 encoder_cell = tf.contrib.rnn.MultiRNNCell([cell(rnn_size, keep_probs) for _ in range(number_of_layers)])
        encoder_embedings = tf.contrib.layers.embed_sequence(inputs, encoder_vocab_size, encoder_embed_size) #used to create embeding
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell, 
                                                            encoder_embedings, 
                                                            encoder_seq_len, 
                                                            dtype=tf.float32)
 return encoder_outputs, encoder_states
 def decoder_inputs_preprocessing(targets, word_to_id, batch_size):
 endings = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1]) 
     return tf.concat([tf.fill([batch_size, 1], word_to_id['<GO>']), endings], 1)
 def decoder(decoder_inputs, enc_states, dec_cell, decoder_embed_size, vocab_size
               dec_seq_len, max_seq_len, word_to_id, batch_size):
  #Defining embedding layer for the Decoder
       embed_layer = tf.Variable(tf.random_uniform([vocab_size, decoder_embed_size]))
       embedings = tf.nn.embedding_lookup(embed_layer, decoder_inputs) 
       #Creating Dense (Fully Connected) layer at the end of the Decoder -  used for generating probabilities for each word in the vocabulary
 output_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1))
 with tf.variable_scope('decoder'):
       #Training helper used only to read inputs in the TRAINING stage
      train_helper = tf.contrib.seq2seq.TrainingHelper(embedings, 
                                                        dec_seq_len)
 #Defining decoder - You can change with BeamSearchDecoder, just beam size
      train_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                     enc_states, 
                                                      output_layer)
  #Finishing the training decoder\n",
        train_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, 
                                                                      impute_finished=True, 
                                                                      maximum_iterations=max_seq_len)
   with tf.variable_scope('decoder', reuse=True): #we use REUSE option in this scope because we want to get same params learned in the previouse 'decoder' scope
           #getting vector of the '<GO>' tags in the int representation
            starting_id_vec = tf.tile(tf.constant([word_to_id['<GO>']], dtype=tf.int32), [batch_size], name='starting_id_vec')
                   #using basic greedy to get next word in the inference time (based only on probs)
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embed_layer, 
                                                                       starting_id_vec, 
                                                                       word_to_id['<EOS>'])
  #Defining decoder - for inference time\n",
          inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                             inference_helper, 
                                                                enc_states, 
                                                                output_layer)
         inference_dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, 
                                                                           impute_finished=True, 
                                                                           maximum_iterations=max_seq_len)

        return train_dec_outputs, inference_dec_output
 
    def attention_mech(rnn_size, keep_probs, encoder_outputs, encoder_states, encoder_seq_len, batch_size):       
             #using internal function to easier create RNN cell
        def cell(units, probs):
            layer = tf.contrib.rnn.BasicLSTMCell(units)
            return tf.contrib.rnn.DropoutWrapper(layer, probs)

        #defining rnn_cell
        decoder_cell = cell(rnn_size, keep_probs)
   
        #using helper function from seq2seq sub_lib for Bahdanau attention
             attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, 
                                                                   encoder_outputs, 
                                                                   encoder_seq_len)
 
       #finishin attention with the attention holder - Attention Wrapper
        dec_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, 
                                                       attention_mechanism, 
                                                       rnn_size/2)
     
        #Here we are usingg zero_state of the LSTM (in this case) decoder cell, and feed the value of the last encoder_state to it
        attention_zero = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        enc_state_new = attention_zero.clone(cell_state=encoder_states[-1])

        return dec_cell, enc_state_new
     ef opt_loss(outputs, targets, dec_seq_len, max_seq_len, learning_rate, clip_rate):

      logits = tf.identity(outputs.rnn_output)

       mask_weigts = tf.sequence_mask(dec_seq_len, max_seq_len, dtype=tf.float32)
 
       with tf.variable_scope('opt_loss'):
           #using sequence_loss to optimize the seq2seq model
           loss = tf.contrib.seq2seq.sequence_loss(logits,
                                                   targets, 
                                                  mask_weigts)
   
           #Define optimizer
           opt = tf.train.AdamOptimizer(learning_rate)
   
          #Next 3 lines used to clip gradients {Prevent gradient explosion problem}
           gradients = tf.gradients(loss, tf.trainable_variables())
           clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_rate)
           traiend_opt = opt.apply_gradients(zip(clipped_grads, tf.trainable_variables()))

       return loss, traiend_opt
    class Chatbot(object):
def __init__(self, learning_rate, batch_size, enc_embed_size, dec_embed_size, rnn_size, number_of_layers, vocab_size,word_to_id,clip_rate):
  
          tf.reset_default_graph()
         
           self.inputs, self.targets, self.keep_probs, self.encoder_seq_len, self.decoder_seq_len, max_seq_len = grap_inputs()
     
           enc_outputs, enc_states = encoder(self.inputs, 
                                             rnn_size,
                                             number_of_layers, 
                                             self.encoder_seq_len, 
                                             self.keep_probs, 
                                             enc_embed_size, 
                                             vocab_size)
           
           dec_inputs = decoder_inputs_preprocessing(self.targets, 
                                                    word_to_id, 
                                                     batch_size)
           
           
           decoder_cell, encoder_states_new = attention_mech(rnn_size, 
                                                             self.keep_probs, 
                                                             enc_outputs, 
                                                             enc_states,
                                                             self.encoder_seq_len, 
                                                             batch_size)
     
           train_outputs, inference_output = decoder(dec_inputs, 
                                                     encoder_states_new, 
                                                     decoder_cell,
                                                     dec_embed_size, 
                                                     vocab_size, 
                                                     self.decoder_seq_len, 
                                                     max_seq_len, 
                                                     word_to_id, 
                                                     batch_size)
                    self.predictions  = tf.identity(inference_output.sample_id, name='preds')
           
           self.loss, self.opt = opt_loss(train_outputs, 
                                          self.targets, 
                                          self.decoder_seq_len, 
                                          max_seq_len,
                                          learning_rate, 
                                          clip_rate)
     def get_accuracy(target, logits):
    
 
       max_seq = max(target.shape[1], logits.shape[1])
       if max_seq - target.shape[1]:
           target = np.pad(
               target
               [(0,0),(0,max_seq - target.shape[1])]
               'constant')\n",
       if max_seq - logits.shape[1]
           logits = np.pad(
               logits
               [(0,0),(0,max_seq - logits.shape[1])]
               'constant')
       return np.mean(np.equal(target, logits))
  
    vocab, word_to_id, id_to_word = create_vocab(cs, ct)
  
    len(vocab)
      encoded_questions = encoder_data(ct, word_to_id)
 
    encoded_answers = encoder_data(cs, word_to_id, True)
 
    bucketed_data = bucket_data(encoded_questions, encoded_answers, word_to_id)
 

    print(len(bucketed_data[0]))
    print(len(bucketed_data[1]))
  
    model = Chatbot(LEARNING_RATE, 
                    BATCH_SIZE, 
                    ENCODING_EMBED_SIZE, 
                    DECODING_EMBED_SIZE, 
                    RNN_SIZE, 
                    NUM_LAYERS,
                    len(vocab), 
                    word_to_id, 
                    CLIP_RATE) #4=clip_rate
      session = tf.Session()
   session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
  
    for i in range(EPOCHS):
        epoch_accuracy = []
        epoch_loss = []
        for b in range(len(bucketed_data)):
            bucket = bucketed_data[b]
            questions_bucket = []
            answers_bucket = []
            bucket_accuracy = []
            bucket_loss = []
            for k in range(len(bucket)):
                questions_bucket.append(np.array(bucket[k][0]))
                answers_bucket.append(np.array(bucket[k][1]))
            print(len(questions_bucket))
            print(len(answers_bucket))
            for ii in range(len(questions_bucket) //  BATCH_SIZE):
       
                starting_id = ii * BATCH_SIZE

                X_batch = questions_bucket[starting_id:starting_id+BATCH_SIZE]
                y_batch = answers_bucket[starting_id:starting_id+BATCH_SIZE]
                print(np.array(X_batch).shape)
                print(np.array(y_batch).shape)
                feed_dict = {model.inputs:X_batch, 
                             model.targets:y_batch, 
                             model.keep_probs:KEEP_PROBS, 
                             model.decoder_seq_len:[len(y_batch[0])]*BATCH_SIZE,
                             model.encoder_seq_len:[len(X_batch[0])]*BATCH_SIZE}
            
                cost, _, preds = session.run([model.loss, model.opt, model.predictions], feed_dict=feed_dict)
    
                #epoch_accuracy.append(get_accuracy(np.array(y_batch), np.array(preds)))
                #bucket_accuracy.append(get_accuracy(np.array(y_batch), np.array(preds)))
              
                bucket_loss.append(cost)
                epoch_loss.append(cost)
             
            #print(\"Bucket {}:\".format(b+1)
           #       \" | Loss: {}\".format(np.mean(bucket_loss))
            #     \" | Accuracy: {}\".format(np.mean(bucket_accuracy)))
            
       #print(\"EPOCH: {}/{}\".format(i, EPOCHS)
       # Epoch loss: {}\".format(np.mean(epoch_loss))
        # Epoch accuracy: {}\".format(np.mean(epoch_accuracy)))
        print(\"Saving model at epoch : \" + str(i))
       saver.save(session, \"./sumarization/chatbot_{}.ckpt\".format(i))
     source": [
   def convert_string2int(data, word2int):
       question, x = [], []
       question.append(data)
       x.append(clean_text(question[0]))
       x = encoder_data(x, word_to_id)
       return x[0]
tf.reset_default_graph()
sumarization_session = tf.Session()
    import_meta.restore(sumarization_session, tf.train.latest_checkpoint('./sumarization/'))
    All_varaibles = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name(\"inputs:0\")
    keep_probs = graph.get_tensor_by_name(\"dropout_rate:0\")
    encoder_seq_len = graph.get_tensor_by_name(\"encoder_seq_len:0\")
    decoder_seq_len = graph.get_tensor_by_name(\"decoder_seq_len:0\")
    preds = graph.get_tensor_by_name(\"preds:0\")"
def sumary():
        users_input = raw_input("You:")
        users_inpur = users_input.lower()
        que = convert_string2int(users_input, word_to_id)
 bucket_lengths = [50,100]\n",
  length = [x for x in bucket_lengths if len(que) <= x] 
 x = [word_to_id['<PAD>']] * (length[0] - len(que))
x = np.array(x)\n",
 que = np.append(x, que)\n",
 fake_batch = np.zeros((BATCH_SIZE, length[0]))
fake_batch[0] = que\n",
feed2_dict = {inputs:fake_batch,  
                             keep_probs:1.0, 
                             decoder_seq_len:[length[0]]*BATCH_SIZE
                             encoder_seq_len:[length[0]]*BATCH_SIZE}
 ans = sumarization_session.run(preds, feed2_dict)
 def clean_ans(text):
 ans = [id_to_word[i] for i in text]
if(ans[0] == '<UNK>' or ans[0] == '<EOS>'):
return
ans_2 = []
for i in range(len(ans)):
if(ans[i] == '<UNK>' or ans[i] == '<EOS>'):
 break
else:
 ans_2.append(ans[i])
str1 = ' '.join(str(e) for e in ans_2)
return str1
print("summary:" + str(clean_ans(ans)))   
 return 1
sumary()  
