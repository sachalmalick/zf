import tensorflow as tf

def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x

class QNet(tf.keras.Model):
    def __init__(self,
            kernel_len=25,
            dim=64,
            use_batchnorm=False,
            phaseshuffle_rad=0,
            num_categ=10):
        super(QNet, self).__init__()

        self.kernel_len = kernel_len
        self.dim = dim
        self.use_batchnorm = use_batchnorm
        self.phaseshuffle_rad = phaseshuffle_rad
        self.num_categ = num_categ
        self.padding="SAME"
        self.conv1 = tf.keras.layers.Conv1D(dim, kernel_len, strides=4, padding=self.padding)
        self.conv2 = tf.keras.layers.Conv1D(dim * 2, kernel_len, strides=4, padding=self.padding)
        self.conv3 = tf.keras.layers.Conv1D(dim * 4, kernel_len, strides=4, padding=self.padding)
        self.conv4 = tf.keras.layers.Conv1D(dim * 8, kernel_len, strides=4, padding=self.padding)
        self.conv5 = tf.keras.layers.Conv1D(dim * 16, kernel_len, strides=4, padding=self.padding)
        self.d = tf.keras.layers.Dense(num_categ)

        if self.use_batchnorm:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.bn4 = tf.keras.layers.BatchNormalization()

        
    def call(self, x, training=True):
        print(x.shape)
        batch_size = tf.shape(x)[0]
        
        if self.phaseshuffle_rad > 0:
            phaseshuffle = lambda x: apply_phaseshuffle(x, self.phaseshuffle_rad)
        else:
            phaseshuffle = lambda x: x

        # Layer 0
        # [16384, 1] -> [4096, 64]
        output = x
        output = self.conv1(output)
        output = lrelu(output)
        output = apply_phaseshuffle(output, self.phaseshuffle_rad)

        output = self.conv2(output)
        if self.use_batchnorm:
            output = self.bn1(output, training=training)
        output = lrelu(output)
        output = apply_phaseshuffle(output, self.phaseshuffle_rad)

        output = self.conv3(output)
        if self.use_batchnorm:
            output = self.bn2(output, training=training)
        output = lrelu(output)
        output = apply_phaseshuffle(output, self.phaseshuffle_rad)

        output = self.conv4(output)
        if self.use_batchnorm:
            output = self.bn3(output, training=training)
        output = lrelu(output)
        output = apply_phaseshuffle(output, self.phaseshuffle_rad)

        output = self.conv5(output)
        if self.use_batchnorm:
            output = self.bn4(output, training=training)
        output = lrelu(output)

        # Flatten
        output = tf.reshape(output, [batch_size, -1])

        # Connect to single logit
        output = self.d(output)

        # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

        return output