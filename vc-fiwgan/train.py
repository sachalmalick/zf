'''
fiwGAN: Featural InfoWaveGAN
Gasper Begus (begus@uw.edu) 2020
Based on WaveGAN (Donahue et al. 2019) and InfoGAN (Chen et al. 2016), partially also on code by Rodionov (2018).
Unlike InfoGAN, the latent code is binomially distributed (features) and training is performed with sigmoid cross-entropy. 
'''

from __future__ import print_function

try:
  import cPickle as pickle
except:
  import pickle
from functools import reduce
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from six.moves import xrange

import loader
from descriminator import Descriminator
from generator import Generator
from qnet import QNet

#tf.compat.v1.disable_v2_behavior() 

"""
  Trains a WaveGAN
"""
def train(fps, args):
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices:, mirrored strategy {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():            #loads input waveforns in filepaths
        x = loader.decode_extract_and_batch(
            fps,
            batch_size=args.train_batch_size,
            slice_len=args.data_slice_len,
            decode_fs=args.data_sample_rate,
            decode_num_channels=args.data_num_channels,
            decode_fast_wav=args.data_fast_wav,
            decode_parallel_calls=4,
            slice_randomize_offset=False if args.data_first_slice else True,
            slice_first_only=args.data_first_slice,
            slice_overlap_ratio=0. if args.data_first_slice else args.data_overlap_ratio,
            slice_pad_end=True if args.data_first_slice else args.data_pad_end,
            repeat=True,
            shuffle=True,
            shuffle_buffer_size=4096,
            prefetch_size=args.train_batch_size * 4,
            prefetch_gpu_num=args.data_prefetch_gpu_num)
        

        # Make z vector
        generator = Generator(**args.wavegan_g_kwargs)
        discriminator = Descriminator(**args.wavegan_d_kwargs)
        qnet = QNet(**args.wavegan_q_kwargs)

        #wgan-gp loss
        g_opt = tf.optimizers.Adam(
                learning_rate=1e-4)
        d_opt = tf.optimizers.Adam(
            learning_rate=1e-4)
        q_opt = tf.optimizers.Adam(
            learning_rate=1e-4)
        
        def discriminator_loss(real, fake):
            return tf.reduce_mean(fake) - tf.reduce_mean(real)

        def generator_loss(fake):
            return -tf.reduce_mean(fake)

        def qnet_loss(z, guessed_z):
            z_q_loss = z[:, : args.num_categ]
            q_q_loss = guessed_z[:, : args.num_categ]
            q_sigmoid = tf.nn.sigmoid_cross_entropy_with_logits(labels=z_q_loss, logits=q_q_loss)
            return tf.reduce_mean(q_sigmoid)

        def make_z():
            categ = categ = tfp.distributions.Bernoulli(probs=0.5, dtype=tf.float32).sample(sample_shape=(args.train_batch_size, args.num_categ))
            uniform = tf.random.uniform([args.train_batch_size,args.wavegan_latent_dim-args.num_categ],-1.,1.)
            return tf.concat([categ,uniform],1)
        
        example_z = make_z()

        def train_step(real_waves):
            real_waves = real_waves[:, :, 0]
            print("single batch shape", real_waves.shape)
            z = make_z()
            with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape, tf.GradientTape() as qnet_tape:
                generated_waves = generator(z, training=True)
                real_output = discriminator(real_waves, training=True)
                fake_output = discriminator(generated_waves, training=True)
                z_guess = qnet(generated_waves, training=True)

                d_loss = discriminator_loss(real_output, fake_output)
                g_loss = generator_loss(fake_output)
                q_loss = qnet_loss(z, z_guess)

                # tf.summary.scalar('G_loss', g_loss)
                # tf.summary.scalar('D_loss', d_loss)
                # tf.summary.scalar('Q_loss', q_loss)

            gen_grd = gen_tape.gradient(g_loss, generator.trainable_variables)
            dis_grd = dis_tape.gradient(d_loss, discriminator.trainable_variables)
            qnet_grd = qnet_tape.gradient(q_loss, qnet.trainable_variables)

            g_opt.apply_gradients(zip(gen_grd, generator.trainable_variables))
            d_opt.apply_gradients(zip(dis_grd, discriminator.trainable_variables))
            q_opt.apply_gradients(zip(qnet_grd, qnet.trainable_variables))

            return (g_loss, d_loss, q_loss)
        
        @tf.function
        def distributed_train_step(dist_inputs):
            per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
            g_loss, d_loss, q_loss = per_replica_losses
            def normalize_loss(loss):
                return strategy.reduce(tf.distribute.ReduceOp.SUM, loss,
                         axis=None)
            return (normalize_loss(g_loss), normalize_loss(d_loss),
                                            normalize_loss(q_loss))
        
        # #check points
        checkpoint = tf.train.Checkpoint(generator_optimizer=g_opt,
                    discriminator_optimizer=d_opt,
                    qnet_optimizer=q_opt,
                    generator=generator,
                    discriminator=discriminator,
                    qnet=qnet)
        
        NUM_EPOCHS = 50
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        #distributed the dataset
        x = strategy.experimental_distribute_dataset(x)
        writer = tf.summary.create_file_writer(args.train_dir)
        def train_loop():
            for epoch in range(0, NUM_EPOCHS):
                print("Epoch", epoch)
                step = 0
                for dist_inputs in x:
                    loss = distributed_train_step(dist_inputs)
                    g_loss, d_loss, q_loss = loss
                    print("step", step, "g loss ", g_loss.numpy(),
                        "d loss ", d_loss.numpy(), "q loss ", q_loss.numpy())
                    with writer.as_default():
                        tf.summary.scalar('Generator Loss', g_loss, step=step)
                        tf.summary.scalar('Descriminator Loss', d_loss, step=step)
                        tf.summary.scalar('QNet Loss', q_loss, step=step)
                        writer.flush()
                    step+=1
        print('Training has started. Please use \'tensorboard --logdir={}\' to monitor.'.format(args.train_dir))
        train_loop()


"""
  Creates and saves a MetaGraphDef for simple inference
  Tensors:
    'samp_z_n' int32 []: Sample this many latent vectors
    'samp_z' float32 [samp_z_n, latent_dim]: Resultant latent vectors
    'z:0' float32 [None, latent_dim]: Input latent vectors
    'flat_pad:0' int32 []: Number of padding samples to use when flattening batch to a single audio file
    'G_z:0' float32 [None, slice_len, 1]: Generated outputs
    'G_z_int16:0' int16 [None, slice_len, 1]: Same as above but quantizied to 16-bit PCM samples
    'G_z_flat:0' float32 [None, 1]: Outputs flattened into single audio file
    'G_z_flat_int16:0' int16 [None, 1]: Same as above but quantized to 16-bit PCM samples
  Example usage:
    import tensorflow as tf
    tf.reset_default_graph()

    saver = tf.train.import_meta_graph('infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'model.ckpt-10000')

    z_n = graph.get_tensor_by_name('samp_z_n:0')
    _z = sess.run(graph.get_tensor_by_name('samp_z:0'), {z_n: 10})

    z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z})
"""
def infer(args):
  infer_dir = os.path.join(args.train_dir, 'infer')
  if not os.path.isdir(infer_dir):
    os.makedirs(infer_dir)

  # Subgraph that generates latent vectors
  samp_z_n = tf.compat.v1.placeholder(tf.int32, [], name='samp_z_n')
  samp_z = tf.random.uniform([samp_z_n, args.wavegan_latent_dim], -1.0, 1.0, dtype=tf.float32, name='samp_z')

  # Input zo
  z = tf.compat.v1.placeholder(tf.float32, [None, args.wavegan_latent_dim], name='z')
  flat_pad = tf.compat.v1.placeholder(tf.int32, [], name='flat_pad')

  # Execute generator
  with tf.compat.v1.variable_scope('G'):
    G_z = WaveGANGenerator(z, train=False, **args.wavegan_g_kwargs)
    if args.wavegan_genr_pp:
      with tf.compat.v1.variable_scope('pp_filt'):
        G_z = tf.compat.v1.layers.conv1d(G_z, 1, args.wavegan_genr_pp_len, use_bias=False, padding='same')
  G_z = tf.identity(G_z, name='G_z')

  # Flatten batch
  nch = int(G_z.get_shape()[-1])
  G_z_padded = tf.pad(G_z, [[0, 0], [0, flat_pad], [0, 0]])
  G_z_flat = tf.reshape(G_z_padded, [-1, nch], name='G_z_flat')

  # Encode to int16
  def float_to_int16(x, name=None):
    x_int16 = x * 32767.
    x_int16 = tf.clip_by_value(x_int16, -32767., 32767.)
    x_int16 = tf.cast(x_int16, tf.int16, name=name)
    return x_int16
  G_z_int16 = float_to_int16(G_z, name='G_z_int16')
  G_z_flat_int16 = float_to_int16(G_z_flat, name='G_z_flat_int16')

  # Create saver
  G_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='G')
  global_step = tf.compat.v1.train.get_or_create_global_step()
  saver = tf.compat.v1.train.Saver(G_vars + [global_step])

  # Export graph
  tf.io.write_graph(tf.compat.v1.get_default_graph(), infer_dir, 'infer.pbtxt')

  # Export MetaGraph
  infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
  tf.compat.v1.train.export_meta_graph(
      filename=infer_metagraph_fp,
      clear_devices=True,
      saver_def=saver.as_saver_def())

  # Reset graph (in case training afterwards)
  tf.compat.v1.reset_default_graph()


"""
  Generates a preview audio file every time a checkpoint is saved
"""
def preview(args):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from scipy.io.wavfile import write as wavwrite
  from scipy.signal import freqz

  preview_dir = os.path.join(args.train_dir, 'preview')
  if not os.path.isdir(preview_dir):
    os.makedirs(preview_dir)

  # Load graph
  infer_metagraph_fp = os.path.join(args.train_dir, 'infer', 'infer.meta')
  graph = tf.compat.v1.get_default_graph()
  saver = tf.compat.v1.train.import_meta_graph(infer_metagraph_fp)

  # Generate or restore z_i and z_o
  z_fp = os.path.join(preview_dir, 'z.pkl')
  if os.path.exists(z_fp):
    with open(z_fp, 'rb') as f:
      _zs = pickle.load(f)
  else:
    # Sample z
    samp_feeds = {}
    samp_feeds[graph.get_tensor_by_name('samp_z_n:0')] = args.preview_n
    samp_fetches = {}
    samp_fetches['zs'] = graph.get_tensor_by_name('samp_z:0')
    with tf.compat.v1.Session() as sess:
      _samp_fetches = sess.run(samp_fetches, samp_feeds)
    _zs = _samp_fetches['zs']

    # Save z
    with open(z_fp, 'wb') as f:
      pickle.dump(_zs, f)

  # Set up graph for generating preview images
  feeds = {}
  feeds[graph.get_tensor_by_name('z:0')] = _zs
  feeds[graph.get_tensor_by_name('flat_pad:0')] = int(args.data_sample_rate / 2)
  fetches = {}
  fetches['step'] = tf.compat.v1.train.get_or_create_global_step()
  fetches['G_z'] = graph.get_tensor_by_name('G_z:0')
  fetches['G_z_flat_int16'] = graph.get_tensor_by_name('G_z_flat_int16:0')
  if args.wavegan_genr_pp:
    fetches['pp_filter'] = graph.get_tensor_by_name('G/pp_filt/conv1d/kernel:0')[:, 0, 0]

  # Summarize
  G_z = graph.get_tensor_by_name('G_z_flat:0')
  summaries = [
      tf.compat.v1.summary.audio('preview', tf.expand_dims(G_z, axis=0), args.data_sample_rate, max_outputs=1)
  ]
  fetches['summaries'] = tf.compat.v1.summary.merge(summaries)
  summary_writer = tf.compat.v1.summary.FileWriter(preview_dir)

  # PP Summarize
  if args.wavegan_genr_pp:
    pp_fp = tf.compat.v1.placeholder(tf.string, [])
    pp_bin = tf.io.read_file(pp_fp)
    pp_png = tf.image.decode_png(pp_bin)
    pp_summary = tf.compat.v1.summary.image('pp_filt', tf.expand_dims(pp_png, axis=0))

  # Loop, waiting for checkpoints
  ckpt_fp = None
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      print('Preview: {}'.format(latest_ckpt_fp))

      with tf.compat.v1.Session() as sess:
        saver.restore(sess, latest_ckpt_fp)

        _fetches = sess.run(fetches, feeds)

        _step = _fetches['step']

      preview_fp = os.path.join(preview_dir, '{}.wav'.format(str(_step).zfill(8)))
      wavwrite(preview_fp, args.data_sample_rate, _fetches['G_z_flat_int16'])

      summary_writer.add_summary(_fetches['summaries'], _step)

      if args.wavegan_genr_pp:
        w, h = freqz(_fetches['pp_filter'])

        fig = plt.figure()
        plt.title('Digital filter frequncy response')
        ax1 = fig.add_subplot(111)

        plt.plot(w, 20 * np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')

        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        plt.plot(w, angles, 'g')
        plt.ylabel('Angle (radians)', color='g')
        plt.grid()
        plt.axis('tight')

        _pp_fp = os.path.join(preview_dir, '{}_ppfilt.png'.format(str(_step).zfill(8)))
        plt.savefig(_pp_fp)

        with tf.compat.v1.Session() as sess:
          _summary = sess.run(pp_summary, {pp_fp: _pp_fp})
          summary_writer.add_summary(_summary, _step)

      print('Done')

      ckpt_fp = latest_ckpt_fp

    time.sleep(1)


"""
  Computes inception score every time a checkpoint is saved
"""
def incept(args):
  incept_dir = os.path.join(args.train_dir, 'incept')
  if not os.path.isdir(incept_dir):
    os.makedirs(incept_dir)

  # Load GAN graph
  gan_graph = tf.Graph()
  with gan_graph.as_default():
    infer_metagraph_fp = os.path.join(args.train_dir, 'infer', 'infer.meta')
    gan_saver = tf.compat.v1.train.import_meta_graph(infer_metagraph_fp)
    score_saver = tf.compat.v1.train.Saver(max_to_keep=1)
  gan_z = gan_graph.get_tensor_by_name('z:0')
  gan_G_z = gan_graph.get_tensor_by_name('G_z:0')[:, :, 0]
  gan_step = gan_graph.get_tensor_by_name('global_step:0')

  # Load or generate latents
  z_fp = os.path.join(incept_dir, 'z.pkl')
  if os.path.exists(z_fp):
    with open(z_fp, 'rb') as f:
      _zs = pickle.load(f)
  else:
    gan_samp_z_n = gan_graph.get_tensor_by_name('samp_z_n:0')
    gan_samp_z = gan_graph.get_tensor_by_name('samp_z:0')
    with tf.compat.v1.Session(graph=gan_graph) as sess:
      _zs = sess.run(gan_samp_z, {gan_samp_z_n: args.incept_n})
    with open(z_fp, 'wb') as f:
      pickle.dump(_zs, f)

  # Load classifier graph
  incept_graph = tf.Graph()
  with incept_graph.as_default():
    incept_saver = tf.compat.v1.train.import_meta_graph(args.incept_metagraph_fp)
  incept_x = incept_graph.get_tensor_by_name('x:0')
  incept_preds = incept_graph.get_tensor_by_name('scores:0')
  incept_sess = tf.compat.v1.Session(graph=incept_graph)
  incept_saver.restore(incept_sess, args.incept_ckpt_fp)

  # Create summaries
  summary_graph = tf.Graph()
  with summary_graph.as_default():
    incept_mean = tf.compat.v1.placeholder(tf.float32, [])
    incept_std = tf.compat.v1.placeholder(tf.float32, [])
    summaries = [
        tf.compat.v1.summary.scalar('incept_mean', incept_mean),
        tf.compat.v1.summary.scalar('incept_std', incept_std)
    ]
    summaries = tf.compat.v1.summary.merge(summaries)
  summary_writer = tf.compat.v1.summary.FileWriter(incept_dir)

  # Loop, waiting for checkpoints
  ckpt_fp = None
  _best_score = 0.
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      print('Incept: {}'.format(latest_ckpt_fp))

      sess = tf.compat.v1.Session(graph=gan_graph)

      gan_saver.restore(sess, latest_ckpt_fp)

      _step = sess.run(gan_step)

      _G_zs = []
      for i in xrange(0, args.incept_n, 100):
        _G_zs.append(sess.run(gan_G_z, {gan_z: _zs[i:i+100]}))
      _G_zs = np.concatenate(_G_zs, axis=0)

      _preds = []
      for i in xrange(0, args.incept_n, 100):
        _preds.append(incept_sess.run(incept_preds, {incept_x: _G_zs[i:i+100]}))
      _preds = np.concatenate(_preds, axis=0)

      # Split into k groups
      _incept_scores = []
      split_size = args.incept_n // args.incept_k
      for i in xrange(args.incept_k):
        _split = _preds[i * split_size:(i + 1) * split_size]
        _kl = _split * (np.log(_split) - np.log(np.expand_dims(np.mean(_split, 0), 0)))
        _kl = np.mean(np.sum(_kl, 1))
        _incept_scores.append(np.exp(_kl))

      _incept_mean, _incept_std = np.mean(_incept_scores), np.std(_incept_scores)

      # Summarize
      with tf.compat.v1.Session(graph=summary_graph) as summary_sess:
        _summaries = summary_sess.run(summaries, {incept_mean: _incept_mean, incept_std: _incept_std})
      summary_writer.add_summary(_summaries, _step)

      # Save
      if _incept_mean > _best_score:
        score_saver.save(sess, os.path.join(incept_dir, 'best_score'), _step)
        _best_score = _incept_mean

      sess.close()

      print('Done')

      ckpt_fp = latest_ckpt_fp

    time.sleep(1)

  incept_sess.close()


if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'preview', 'incept', 'infer'])
  parser.add_argument('train_dir', type=str,
      help='Training directory')

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_dir', type=str,
      help='Data directory containing *only* audio files to load')
  data_args.add_argument('--data_sample_rate', type=int,
      help='Number of audio samples per second')
  data_args.add_argument('--data_slice_len', type=int, choices=[16384, 32768, 65536],
      help='Number of audio samples per slice (maximum generation length)')
  data_args.add_argument('--data_num_channels', type=int,
      help='Number of audio channels to generate (for >2, must match that of data)')
  data_args.add_argument('--data_overlap_ratio', type=float,
      help='Overlap ratio [0, 1) between slices')
  data_args.add_argument('--data_first_slice', action='store_true', dest='data_first_slice',
      help='If set, only use the first slice each audio example')
  data_args.add_argument('--data_pad_end', action='store_true', dest='data_pad_end',
      help='If set, use zero-padded partial slices from the end of each audio file')
  data_args.add_argument('--data_normalize', action='store_true', dest='data_normalize',
      help='If set, normalize the training examples')
  data_args.add_argument('--data_fast_wav', action='store_true', dest='data_fast_wav',
      help='If your data is comprised of standard WAV files (16-bit signed PCM or 32-bit float), use this flag to decode audio using scipy (faster) instead of librosa')
  data_args.add_argument('--data_prefetch_gpu_num', type=int,
      help='If nonnegative, prefetch examples to this GPU (Tensorflow device num)')

  wavegan_args = parser.add_argument_group('WaveGAN')
  wavegan_args.add_argument('--wavegan_latent_dim', type=int,
      help='Number of dimensions of the latent space')
  wavegan_args.add_argument('--wavegan_kernel_len', type=int,
      help='Length of 1D filter kernels')
  wavegan_args.add_argument('--wavegan_dim', type=int,
      help='Dimensionality multiplier for model of G and D')
  wavegan_args.add_argument('--num_categ', type=int,
      help='Number of categorical variables')
  wavegan_args.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',
      help='Enable batchnorm')
  wavegan_args.add_argument('--wavegan_disc_nupdates', type=int,
      help='Number of discriminator updates per generator update')
  wavegan_args.add_argument('--wavegan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],
      help='Which GAN loss to use')
  wavegan_args.add_argument('--wavegan_genr_upsample', type=str, choices=['zeros', 'nn'],
      help='Generator upsample strategy')
  wavegan_args.add_argument('--wavegan_genr_pp', action='store_true', dest='wavegan_genr_pp',
      help='If set, use post-processing filter')
  wavegan_args.add_argument('--wavegan_genr_pp_len', type=int,
      help='Length of post-processing filter for DCGAN')
  wavegan_args.add_argument('--wavegan_disc_phaseshuffle', type=int,
      help='Radius of phase shuffle operation')

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_batch_size', type=int,
      help='Batch size')
  train_args.add_argument('--train_save_secs', type=int,
      help='How often to save model')
  train_args.add_argument('--train_summary_secs', type=int,
      help='How often to report summaries')

  preview_args = parser.add_argument_group('Preview')
  preview_args.add_argument('--preview_n', type=int,
      help='Number of samples to preview')

  incept_args = parser.add_argument_group('Incept')
  incept_args.add_argument('--incept_metagraph_fp', type=str,
      help='Inference model for inception score')
  incept_args.add_argument('--incept_ckpt_fp', type=str,
      help='Checkpoint for inference model')
  incept_args.add_argument('--incept_n', type=int,
      help='Number of generated examples to test')
  incept_args.add_argument('--incept_k', type=int,
      help='Number of groups to test')

  parser.set_defaults(
    data_dir=None,
    data_sample_rate=16000,
    data_slice_len=16384,
    data_num_channels=1,
    data_overlap_ratio=0.,
    data_first_slice=False,
    data_pad_end=False,
    data_normalize=False,
    data_fast_wav=False,
    data_prefetch_gpu_num=0,
    wavegan_latent_dim=100,
    wavegan_kernel_len=25,
    wavegan_dim=64,
    num_categ=3,
    wavegan_batchnorm=False,
    wavegan_disc_nupdates=5,
    wavegan_loss='wgan-gp',
    wavegan_genr_upsample='zeros',
    wavegan_genr_pp=False,
    wavegan_genr_pp_len=512,
    wavegan_disc_phaseshuffle=2,
    train_batch_size=64,
    train_save_secs=300,
    train_summary_secs=120,
    preview_n=32,
    incept_metagraph_fp='./eval/inception/infer.meta',
    incept_ckpt_fp='./eval/inception/best_acc-103005',
    incept_n=5000,
    incept_k=10)

  args = parser.parse_args()

  # Make train dir
  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  # Save args
  with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

  # Make model kwarg dicts
  setattr(args, 'wavegan_g_kwargs', {
    'slice_len': args.data_slice_len,
    'nch': args.data_num_channels,
    'kernel_len': args.wavegan_kernel_len,
    'dim': args.wavegan_dim,
    'use_batchnorm': args.wavegan_batchnorm,
    'upsample': args.wavegan_genr_upsample
  })
  setattr(args, 'wavegan_d_kwargs', {
    'kernel_len': args.wavegan_kernel_len,
    'dim': args.wavegan_dim,
    'use_batchnorm': args.wavegan_batchnorm,
    'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
  })
  setattr(args, 'wavegan_q_kwargs', {
    'kernel_len': args.wavegan_kernel_len,
    'dim': args.wavegan_dim,
    'use_batchnorm': args.wavegan_batchnorm,
    'phaseshuffle_rad': args.wavegan_disc_phaseshuffle,
    'num_categ': args.num_categ
  })

  if args.mode == 'train':
    fps = glob.glob(os.path.join(args.data_dir, '*'))
    if len(fps) == 0:
      raise Exception('Did not find any audio files in specified directory')
    print('Found {} audio files in specified directory'.format(len(fps)))
    #infer(args)
    train(fps, args)
  elif args.mode == 'preview':
    preview(args)
  elif args.mode == 'incept':
    incept(args)
  elif args.mode == 'infer':
    infer(args)
  else:
    raise NotImplementedError()
