import tensorflow as tf

from awp_protocol.losses.loss import AdversarialLoss
from awp_protocol.losses.loss_context import LossContext

class TradesLoss(AdversarialLoss):
    def __init__(self, regularization_parameter: float = 1.0, eps: float = 1e-3):
        super().__init__()
        self._value_instead_of_zero = eps
        if regularization_parameter < 0.0:
            raise Exception(f"Beta parameter must be greater than 0. Passed value is {regularization_parameter}")
        self._regularization_parameter = regularization_parameter
        self._sparse_categorical_cross_entropy = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        # self._kl_divergence = tf.keras.losses.KLDivergence(reduction="sum_over_batch_size")

    @tf.function
    def calculate(self, loss_context: LossContext) -> tf.Tensor:
        y = loss_context.y_true
        logits = loss_context.logits_out
        logits_adv = loss_context.logits_pert
        # probabilities = tf.nn.softmax(logits)
        # probabilities_adv = tf.nn.softmax(logits_adv)

        # min_boundry = self._value_instead_of_zero
        # max_boundry = tf.dtypes.as_dtype(logits.dtype).max
        # logits_clipped = tf.clip_by_value(logits, clip_value_min=min_boundry, clip_value_max=max_boundry)

        loss_clean = self._sparse_categorical_cross_entropy(y, logits)
        loss_kl = _kld_loss(logits, logits_adv)
        loss = loss_clean + self._regularization_parameter * loss_kl
        return loss

def _kld_loss(logits, logits_adv):
    p = tf.nn.softmax(logits)
    log_q = tf.nn.log_softmax(logits_adv)
    return tf.reduce_mean(
        tf.reduce_sum(p * (tf.math.log(p + 1e-12) - log_q), axis=-1)
    )
