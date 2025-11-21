from components import Text_Encoder, Vit, Text_Decoder
from keras import Model, metrics, losses
from keras.src.layers import Layer, Concatenate, Embedding
from keras.src import ops
from tensorflow import stop_gradient, newaxis, reduce_sum


class masked_Concat(Layer):
    def __init__(self):
        super().__init__()
        self.concat = Concatenate(axis=-2)

    def compute_mask(self, inputs, previous_mask):
        return getattr(inputs[1], '_keras_mask', None)

    def call(self, inputs):
        P_CLS, T = inputs
        x = self.concat([P_CLS, T[:, 1:, :]])
        return x


class SAA(Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.Text_Embedding = Embedding(input_dim=vocab_size,
                                        output_dim=512,
                                        mask_zero=True)
        self.text_encoder = Text_Encoder()
        self.vit = Vit()
        self.text_decoder = Text_Decoder(vocab_size)
        self.concat = masked_Concat()

        self.EP_consistency_loss_tracker = metrics.Mean(name="EP_consistency_loss")
        self.PP_consistency_loss_tracker = metrics.Mean(name="PP_consistency_loss")
        self.average_pairwise_distances = metrics.Mean(name="APD")
        self.anchor_variance = metrics.Mean(name="anchor_variance")
        self.norm = metrics.Mean(name="anchor_norm")

    def pairwise_diversity(self, features):
        a = ops.expand_dims(features, 1)
        b = ops.expand_dims(features, 0)
        distances = ops.norm(a - b, axis=-1)
        B = ops.cast(ops.shape(features)[0], 'float32')
        mask = 1.0 - ops.eye(B, dtype='float32')
        return reduce_sum(distances * mask) / ops.cast(B * (B - 1.0), 'float32')

    def _compute_consistency_loss(self, CLS, anchor):
        mse = losses.mean_squared_error(anchor, CLS)
        mae = losses.mean_absolute_error(anchor, CLS)
        CL = 0.8 * mse + 0.2 * mae
        return CL

    # inputs = texts(B, 30), patches, augmented_patche(B, 49, 3072)
    def call(self, inputs, *args, **kwargs):
        texts, patches, augmented_patches = inputs
        T = self.Text_Embedding(texts)
        # (B, 512)
        E_CLS = self.text_encoder(T)[:, 0, :]
        P_CLS = self.vit(patches)[:, 0, :]
        self.anchor_variance.update_state(ops.var(P_CLS, axis=0))
        self.norm.update_state(ops.mean(ops.norm(P_CLS, axis=-1)))
        self.average_pairwise_distances.update_state(self.pairwise_diversity(P_CLS))

        P_CLS_aug = self.vit(augmented_patches)[:, 0, :]

        anchor = stop_gradient(P_CLS)
        # Consistency loss
        EP_CL = self._compute_consistency_loss(E_CLS, anchor)
        self.add_loss(EP_CL)
        self.EP_consistency_loss_tracker.update_state(EP_CL)

        PP_CL = self._compute_consistency_loss(P_CLS_aug, anchor)
        self.add_loss(0.01 * PP_CL)
        self.PP_consistency_loss_tracker.update_state(PP_CL)

        y = self.text_decoder(self.concat([P_CLS[:, newaxis, :], T]))
        return y
