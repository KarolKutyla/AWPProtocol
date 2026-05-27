import tensorflow as tf

from losses.loss import AdversarialLoss
from losses.loss_context import LossContext


class AdversarialSparseCategoricalCrossEntropy(AdversarialLoss):
    def __init__(self):
        super().__init__()

    @tf.function
    def calculate(self, loss_context: LossContext) -> tf.Tensor:
        y = loss_context.y_batch
        logits_adv = loss_context.logits_adv
        loss = tf.losses.sparse_categorical_crossentropy(y, logits_adv, from_logits=True)
        return loss
