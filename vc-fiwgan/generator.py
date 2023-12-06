import tensorflow as tf

class Conv1dTranspose(tf.keras.layers.Layer):
    def __init__(self,
        filters,
        kernel_width,
        stride=4,
        padding='same',
        upsample='zeros'):

        super(Conv1dTranspose, self).__init__()
        self.filters = filters
        self.kernel_width = kernel_width
        self.stride = stride
        self.padding = padding
        self.upsample = upsample
        if self.upsample == 'zeros':
            self.conv_transpose = tf.keras.layers.Conv2DTranspose(
                self.filters, (1, self.kernel_width),
                strides=(1, self.stride), padding="same")
        else:
            self.conv1d = tf.keras.layers.Conv1D(
                self.filters, self.kernel_width,
                strides=1, padding=self.padding)

    def call(self, inputs):
        if self.upsample == 'zeros':
            x = tf.expand_dims(inputs, axis=1)
            x = self.conv_transpose(x)
            return x[:, 0]
        else:
            _, w, nch = inputs.get_shape().as_list()

            x = inputs

            x = tf.expand_dims(x, axis=1)
            x = tf.image.resize(x, [1, w * self.stride], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = x[:, 0]
            x = self.conv1d(x)
            return x


class Generator(tf.keras.Model):
    def __init__(self,
            slice_len=16384, #sample length
            nch=1, #number of channels
            kernel_len=25, #kernel / filter length
            dim=64,
            use_batchnorm=False,
            upsample='zeros',
            padding='same'):
        super(Generator, self).__init__()

        self.slice_len = slice_len
        self.nch = nch
        self.kernel_len = kernel_len
        self.dim = dim
        self.use_batchnorm = use_batchnorm
        self.upsample = upsample
        self.padding = padding
        
        self.dim_mul = 16 if slice_len == 16384 else 32
        self.d1 = tf.keras.layers.Dense(4 * 4 * dim * self.dim_mul)
        self.a1 = tf.keras.layers.ReLU()
        self.b1 = tf.keras.layers.BatchNormalization()
        self.r1 = tf.keras.layers.Reshape([16, self.dim * self.dim_mul])
        
        dim_mul2 = self.dim_mul // 2
        self.c1dt2 = Conv1dTranspose(dim * dim_mul2, kernel_len, 4, upsample=upsample)
        self.a2 = tf.keras.layers.ReLU()
        self.b2 = tf.keras.layers.BatchNormalization()

        dim_mul3 = dim_mul2 // 2
        self.c1dt3 = Conv1dTranspose(dim * dim_mul3, kernel_len, 4, upsample=upsample)
        self.a3 = tf.keras.layers.ReLU()
        self.b3= tf.keras.layers.BatchNormalization()

        dim_mul4 = dim_mul3 // 2
        self.c1dt4 = Conv1dTranspose(dim * dim_mul4, kernel_len, 4, upsample=upsample)
        self.a4 = tf.keras.layers.ReLU()
        self.b4= tf.keras.layers.BatchNormalization()

        dim_mul5 = dim_mul4 // 2
        self.c1dt5 = Conv1dTranspose(dim * dim_mul5, kernel_len, 4, upsample=upsample)
        self.a5 = tf.keras.layers.ReLU()
        self.b5 = tf.keras.layers.BatchNormalization()


        self.c1dt6= Conv1dTranspose(nch, kernel_len, 4, upsample=upsample)
        self.b6= tf.keras.layers.BatchNormalization()
        
    def call(self, z, training=True):
        x = z

        #L1
        x = self.d1(x)
        x = self.r1(x)
        if(self.use_batchnorm):
            x = self.b1(x, training=training)
        x = self.a1(x)

        #L2
        x = self.c1dt2(x)
        if(self.use_batchnorm):
            x = self.b2(x, training=training)
        x = self.a2(x)
        #L3
        x = self.c1dt3(x)
        if(self.use_batchnorm):
            x = self.b3(x, training=training)
        x = self.a3(x)
        #L4
        x = self.c1dt4(x)
        if(self.use_batchnorm):
            x = self.b4(x, training=training)
        x = self.a4(x)
        #todo: extend to non 16384 sample lengths

        #L4
        x = self.c1dt5(x)
        if(self.use_batchnorm):
            x = self.b5(x, training=training)
        x = self.a5(x)
        #todo: extend to non 16384 sample lengths

        #L5
        x = self.c1dt6(x)
        if(self.use_batchnorm):
            x = self.b6(x, training=training)
        x = tf.nn.tanh(x)
        #todo add batch norm option
        return x