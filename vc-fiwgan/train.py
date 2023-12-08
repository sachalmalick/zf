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
        x = loader.decode_extract(
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
        
        # audio_data = loader.load_data_trf("../proc")
        # print(audio_data.shape)
        

        # Make z vector
        generator = Generator(**args.wavegan_g_kwargs)
        discriminator = Descriminator(**args.wavegan_d_kwargs)
        qnet = QNet(**args.wavegan_q_kwargs)

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                              reduction=tf.keras.losses.Reduction.NONE)

        #wgan-gp loss
        g_opt = tf.keras.optimizers.RMSprop(
                learning_rate=5e-5, clipnorm=1.0)
        d_opt = tf.keras.optimizers.RMSprop(
            learning_rate=5e-5, clipnorm=1.0)
        q_opt = tf.keras.optimizers.RMSprop(
            learning_rate=5e-5)
        
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
        
        basis_z = make_z()

        def train_step(real_waves):
            #real_waves = real_waves[:, :, 0]
            print("single batch shape", real_waves.shape)
            z = make_z()
            with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape, tf.GradientTape() as qnet_tape:
                generated_waves = generator(z, training=True)
                print(generated_waves.shape)
                real_output = discriminator(real_waves, training=True)
                fake_output = discriminator(generated_waves, training=True)
                z_guess = qnet(generated_waves, training=True)
                print(generated_waves[0, 300:350, 0])
                d_loss = discriminator_loss(real_output, fake_output)
                g_loss = generator_loss(fake_output)
                q_loss = qnet_loss(z, z_guess)

            gen_grd = gen_tape.gradient(g_loss, generator.trainable_variables)
            dis_grd = dis_tape.gradient(d_loss, discriminator.trainable_variables)
            qnet_grd = qnet_tape.gradient(q_loss, qnet.trainable_variables + generator.trainable_variables)

            g_opt.apply_gradients(zip(gen_grd, generator.trainable_variables))
            d_opt.apply_gradients(zip(dis_grd, discriminator.trainable_variables))
            q_opt.apply_gradients(zip(qnet_grd, qnet.trainable_variables + generator.trainable_variables))

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
        checkpoint = tf.train.Checkpoint(generator=generator,
                    discriminator=discriminator,
                    qnet=qnet)
        manager = tf.train.CheckpointManager(checkpoint, 'checkpoints', max_to_keep=3)
        
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        #distributed the dataset
        #x = strategy.experimental_distribute_dataset(x)
        writer = tf.summary.create_file_writer(args.train_dir)
        ds = tf.data.Dataset.from_tensor_slices(x)
        ds = ds.batch(args.train_batch_size, drop_remainder=True)

        def save_and_summarize(generator, step, epoch):
          basis_epoch_name = "basis_step_{}".format(step)
          checkpoint_epoch_name = "checkpoint_step_{}".format(step)
          generated_basis = generate_audio(basis_z, generator)
          checkpoint_z = make_z()
          generated_checkpoint= generate_audio(checkpoint_z, generator)

          with writer.as_default():
            tf.summary.audio(
              basis_epoch_name,
              tf.cast(generated_basis, tf.float32),
              args.data_sample_rate,
              step=step
            )
            tf.summary.audio(
              checkpoint_epoch_name,
              tf.cast(generated_checkpoint, tf.float32),
              args.data_sample_rate,
              step=step
            )

            save(generated_basis, basis_z, basis_epoch_name, args.data_sample_rate)
            save(generated_checkpoint, checkpoint_z, checkpoint_epoch_name, args.data_sample_rate)
            manager.save()

        def train_loop():
            step = 0
            for epoch in range(0, args.train_epochs):
                print("Epoch", epoch)
                for batch in ds:
                    loss = distributed_train_step(tf.convert_to_tensor(batch))
                    g_loss, d_loss, q_loss = loss
                    print("step", step, "g loss ", g_loss.numpy(),
                        "d loss ", d_loss.numpy(), "q loss ", q_loss.numpy())
                    if((step % args.train_summary_steps) == 0):
                      with writer.as_default():
                          tf.summary.scalar('Generator Loss', g_loss, step=step)
                          tf.summary.scalar('Descriminator Loss', d_loss, step=step)
                          tf.summary.scalar('QNet Loss', q_loss, step=step)
                          writer.flush()
                    step+=1
                if((epoch % args.train_summary_epochs) == 0):
                  save_and_summarize(generator, step, epoch)
                  batch = decode_audio(batch)
                  save(batch, basis_z, "batch_{}".format(epoch), args.data_sample_rate)
        print('Training has started. Please use \'tensorboard --logdir={}\' to monitor.'.format(args.train_dir))
        train_loop()



def generate_audio(z, generator):
  output = generator(z, training=False)
  output = output * 32767
  output = tf.clip_by_value(output, -32767., 32767.)
  output = tf.cast(output, tf.int16)
  return output

def sample_from_batch(batch, num_samples):
  random_indices = tf.random.uniform(shape=(num_samples,), minval=0, maxval=batch.shape[0], dtype=tf.int32)
  selected_slices = tf.gather(batch, random_indices)
  return selected_slices

def decode_audio(audio):
  output = audio
  output = output * 32767
  output = tf.clip_by_value(output, -32767., 32767.)
  output = tf.cast(output, tf.int16)
  return output

"""
  Generates a preview audio file every time a checkpoint is saved
"""
def save(generated_batch, z, name, sample_rate):
  from scipy.io.wavfile import write as wavwrite

  preview_dir = os.path.join(args.train_dir, 'preview')
  if not os.path.isdir(preview_dir):
    os.makedirs(preview_dir)

  #save z
  z_fp = os.path.join(preview_dir, 'z_{}.pkl'.format(name))
  with open(z_fp, 'wb') as f:
      pickle.dump(z, f)
  generated_samples = sample_from_batch(generated_batch, 3)
  for i, sample in enumerate(generated_samples):
      z_fp = os.path.join(preview_dir, 'z_{}_{}.pkl'.format(name, i))
      with open(z_fp, 'wb') as f:
          pickle.dump(z[i], f)

      # Save each sample as a separate WAV file
      output_wave = os.path.join(preview_dir, '{}_{}.wav'.format(name, i))
      wavwrite(output_wave, sample_rate, sample.numpy())


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
  data_args.add_argument('--√', type=int,
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
  train_args.add_argument('--train_save_epochs', type=int,
      help='How often to save model')
  train_args.add_argument('--train_summary_epochs', type=int,
      help='How often to report summaries')
  train_args.add_argument('--train_summary_steps', type=int,
      help='How often to report summaries')
  train_args.add_argument('--train_epochs', type=int,
      help='Number of epochs')

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
    train_save_epochs=100,
    train_summary_epochs=5,
    train_summary_steps=10,
    train_epochs = 5000,
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
