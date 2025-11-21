"""
Forward propagation implementation: After loading pre-trained parameters,
it can be directly used for fine-tuning or transfer learning
—all parameters are frozen.
"""

from components import Attention_Block
from keras.src.layers.layer import Layer
from keras.src.layers import Dense
from tensorflow import shape
from keras.src import ops
import parameters as para


class Vit(Layer):
    def __init__(self, num_layers=8, trainable=False):
        super().__init__()
        # initialization
        self.in_ = Dense(512, trainable=trainable)
        self.ABs = [Attention_Block(trainable=trainable) for _ in range(num_layers)]
        self.linear_projection = Dense(512, trainable=trainable)
        self.CLS = self.add_weight(name='CLS',
                                   shape=[1, 1, 512], trainable=trainable)
        # building
        self.in_.build([None, None, 3072])
        for j in range(num_layers):
            self.ABs[j].build([None, None, 512])
        self.linear_projection.build([None, None, 512])

        # assigning
        self.in_.kernel.assign(para.vit_in_[0])
        self.in_.bias.assign(para.vit_in_[1])
        for i in range(num_layers):
            self.ABs[i].load_own_variables(para.vit_ABs[i])
        self.linear_projection.kernel.assign(para.vit_linear_projection[0])
        self.linear_projection.bias.assign(para.vit_linear_projection[1])
        self.CLS.assign(para.vit_CLS)

    # input : patches (B, L, patch_pixels)
    # output : (B, L+1, 512)
    def call(self, x, *args, **kwargs):
        # (B, L, 512)
        x = self.in_(x)
        x = ops.concatenate([ops.tile(self.CLS, [shape(x)[0], 1, 1]), x], axis=-2)
        for AB in self.ABs:
            x = AB(x)
        return self.linear_projection(x)


class Text_Decoder(Layer):
    def __init__(self, vocab_size, num_layers=4, trainable=False):
        super().__init__()
        # initialization
        self.ABs = [Attention_Block(causal_mask=True, trainable=trainable) for _ in range(num_layers)]
        self.predict_head = Dense(vocab_size, activation='softmax', trainable=trainable)
        # building
        # Dynamic dimensions, specified with None, do not affect the composition of model weights.
        for j in range(num_layers):
            self.ABs[j].build([None, None, 512])
        self.predict_head.build([None, None, 512])
        # assigning
        for i in range(num_layers):
            self.ABs[i].load_own_variables(para.text_decoder_ABs[i])
        self.predict_head.kernel.assign(para.text_decoder_predict_head[0])
        self.predict_head.bias.assign(para.text_decoder_predict_head[1])

    # input : (B, L, 512)
    # output : (B, L, vocab_size)
    def call(self, x, *args, **kwargs):
        for AB in self.ABs:
            x = AB(x)
        return self.predict_head(x)

