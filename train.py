from keras import losses, optimizers
from keras.src import ops, metrics
from tensorflow import reduce_sum, argmax, data
import numpy as np
from SelfAnchorAlign import SAA


def mask_loss(y_true, y_pred):
    PL = losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = ops.cast(y_true != 0, 'float32')
    PL = reduce_sum(PL * mask) / reduce_sum(mask)
    return PL


def accuracy(y_true, y_pred):
    predictions = ops.cast(argmax(y_pred, axis=-1), 'float32')
    y_true = ops.cast(y_true, 'float32')
    mask = ops.cast(y_true != 0, 'float32')
    temp = ops.cast(ops.equal(predictions, y_true), 'float32')
    acc = reduce_sum(temp * mask) / reduce_sum(mask)
    return acc


# Save model weights using dictionary
def save_weights(model, file):
    weights = {}
    for weight in model.weights:
        weights[weight.path] = weight.numpy()
    np.save(file, weights)


if __name__ == '__main__':
    # (B, L)
    input_text = ''
    # (B, N, C)
    patches = ''
    # (B, N, C)
    aug_patches = ''
    # (B, L)
    output_text = ''

    train_dataset = data.Dataset.from_tensor_slices(((input_text, patches, patches), output_text)).batch(512)

    model = SAA(vocab_size=10002)
    schedule = optimizers.schedules.CosineDecay(initial_learning_rate=0.0001,
                                                decay_steps=100000,
                                                alpha=0.0001)
    model.compile(optimizer=optimizers.AdamW(learning_rate=schedule,
                                             weight_decay=0.000001),
                  loss=mask_loss,
                  metrics=[metrics.MeanMetricWrapper(fn=accuracy, name='accuracy')]
                  )
    model.fit(train_dataset,
              epochs=256
              )
    save_weights(model, 'weights.npy')

