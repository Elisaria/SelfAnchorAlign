"""
Read the pre-trained parameters
"""
import numpy as np

weights = np.load("TEm_TDe_IEn.npy", allow_pickle=True)  # 加载时需允许pickle
weights = weights.item()
for key in weights.keys():
    print(key)

text_embedding = weights['saa/embedding/embeddings']
print(text_embedding.shape)
vit_in_ = [weights['saa/vit/in_/kernel'], weights['saa/vit/in_/bias']]
vit_CLS = weights['saa/vit/CLS']
vit_linear_projection = [weights['saa/vit/linear_projection/kernel'], weights['saa/vit/linear_projection/bias']]
vit_ABs = []
for i in range(8):
    vit_ABs.append({'query_kernel': weights['saa/vit/AB' + str(i) + '/query_kernel'],
                    'key_kernel': weights['saa/vit/AB' + str(i) + '/key_kernel'],
                    'value_kernel': weights['saa/vit/AB' + str(i) + '/value_kernel'],
                    'W1': weights['saa/vit/AB' + str(i) + '/W1'],
                    'W2': weights['saa/vit/AB' + str(i) + '/W2'],
                    'W3': weights['saa/vit/AB' + str(i) + '/W3'],
                    'gamma': weights['saa/vit/AB' + str(i) + '/gamma']})

text_decoder_predict_head = [weights['saa/text__decoder/predict_head/kernel'],
                             weights['saa/text__decoder/predict_head/bias']]
text_decoder_ABs = []
for i in range(4):
    text_decoder_ABs.append({'query_kernel': weights['saa/text__decoder/AB' + str(i) + '/query_kernel'],
                             'key_kernel': weights['saa/text__decoder/AB' + str(i) + '/key_kernel'],
                             'value_kernel': weights['saa/text__decoder/AB' + str(i) + '/value_kernel'],
                             'W1': weights['saa/text__decoder/AB' + str(i) + '/W1'],
                             'W2': weights['saa/text__decoder/AB' + str(i) + '/W2'],
                             'W3': weights['saa/text__decoder/AB' + str(i) + '/W3'],
                             'gamma': weights['saa/text__decoder/AB' + str(i) + '/gamma']})
