import theano
import theano.sandbox.cuda
#theano.sandbox.cuda.use("gpu0")

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, LSTM, TimeDistributed, Dense
from keras.engine import merge

from ALSTM import ALSTM, RepeatTimeDistributedVector, HierarchicalSoftmax

voc_size = 35000
voc_dim = 50
middle_dim = 100
max_out = 20


EOS = "<EOS>"
characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
characters.append(EOS)
int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}
print(char2int)

VOCAB_SIZE = len(characters)

input_seq = Input(shape=(None,), dtype='int32')

embedded = Embedding(VOCAB_SIZE, voc_dim, name='embd')(input_seq)
#drop_out = Dropout(0.1, name='d_o')(embedded)

forward = LSTM(middle_dim, return_sequences=True, consume_less='mem', name='fwd')(embedded)
backward = LSTM(middle_dim, return_sequences=True, go_backwards=True, name='bwd')(embedded)

sum_res = merge([forward, backward], mode='sum', name='mrg')

repeat = RepeatTimeDistributedVector(max_out, name='RTD')(sum_res)

alstm = ALSTM(voc_dim, return_sequences=True, name='ALSTM')(repeat)

dense = TimeDistributed(Dense(VOCAB_SIZE, name='d_t_d'), name='t_d1')(alstm)

out = TimeDistributed(HierarchicalSoftmax(levels=2, name='HSM'), name='t_d2')(dense)

model = Model(input_seq,out)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

sentence = "May the force be with you"
sentence = [EOS] + list(sentence) + [EOS]
sentence = [char2int[c] for c in sentence]

out_sentence = "force is with you"
out_sentence = [EOS] + list(out_sentence) + [EOS]
while len(out_sentence)<20:
    out_sentence += [EOS]
out_sentence = [char2int[c] for c in out_sentence]

Xtrain = np.asarray([sentence])
Ytrain = [[[1 if c==i else 0 for i in range(VOCAB_SIZE)] for c in out_sentence]]
Ytrain = np.asarray(Ytrain)

print("Fitting...")
model.fit(Xtrain,Ytrain, nb_epoch=1000)

p = model.predict(Xtrain)
res = ''
for c in p[0]:
    res += int2char[int(np.argmax(c))] if int2char[int(np.argmax(c))]!=EOS else ''

print(res)