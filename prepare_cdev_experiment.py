import tensorflow as tf
tf.compat.v1.disable_v2_behavior() 
import os
tf.compat.v1.reset_default_graph()

import pickle
saver = tf.compat.v1.train.import_meta_graph('train_dir/infer/infer.meta')
graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.InteractiveSession()
saver.restore(sess, 'train_dir/model.ckpt-56823')

z_n = graph.get_tensor_by_name('samp_z_n:0')
_z = sess.run(graph.get_tensor_by_name('samp_z:0'), {z_n: 64})

@tf.function
def generate_z_groups(latents, features):
    features = tf.broadcast_to(features, [latents.shape[0], features.shape[0]])
    return tf.concat([features, latents], axis=1)

z = graph.get_tensor_by_name('z:0')
#_z_reshaped = tf.reshape(_z, [-1, 16384, 1])
_G_z = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z})
print(_G_z)



@tf.function
def sample_from_batch(batch, num_samples):
  random_indices = tf.random.uniform(shape=(num_samples,), minval=0, maxval=batch.shape[0], dtype=tf.int32)
  selected_slices = tf.gather(batch, random_indices)
  return selected_slices

@tf.function
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

  preview_dir = os.path.join("traom_dir", 'preview')
  if not os.path.isdir(preview_dir):
    os.makedirs(preview_dir)

  i = 0
  for sample in generated_batch:
      z_fp = os.path.join(preview_dir, 'z_{}_{}.pkl'.format(name, i))
      with open(z_fp, 'wb') as f:
          print(z[i][0:3])
          pickle.dump(z[i], f)

      # Save each sample as a separate WAV file
      output_wave = os.path.join("experiments", 'gz_{}_{}.wav'.format(name, i))
      wavwrite(output_wave, sample_rate, sample)
      i+=1

def generate_doses(features):
    doses = []
    for i in range(-3, 3):
       for feature in features:
          feature_name, feature_code = feature
          dose_name = feature_name + "_dose_" + str(i)
          dose_code = feature_code * i
          doses.append((dose_name, dose_code))
    return doses

@tf.function
def experiment():
    features = []
    features.append(("000_", tf.constant([0, 0, 0])))
    features.append(("001_", tf.constant([0, 0, 1])))
    features.append(("010_", tf.constant([0, 1, 0])))
    features.append(("011_", tf.constant([0, 1, 1])))
    features.append(("100_", tf.constant([1, 0, 0])))
    features.append(("101_", tf.constant([1, 0, 1])))
    features.append(("110_", tf.constant([1, 1, 0])))
    features.append(("111_", tf.constant([1, 1, 1])))

    latents = []
    latents.append(("lat1", tf.random.uniform([64, 97],-1.,1.)))
    latents.append(("lat2", tf.random.uniform([64, 97],-1.,1.)))
    latents.append(("lat3", tf.random.uniform([64, 97],-1.,1.)))
    features.extend(generate_doses(features))
    new_sess = tf.compat.v1.InteractiveSession()
    print("Generating files with all expected latent codes")
    for latent in latents:
        latent_name, latent_code = latent
        for feature in features:
            feature_name, feature_code = feature
            feature_code = tf.cast(feature_code, float)
            _z = generate_z_groups(latent_code, feature_code)
            print(_z)
            _z_val = new_sess.run(_z)
            _z_val = _z_val
            print(_z.shape)
            gz = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z_val})
            audio = new_sess.run(decode_audio(gz))
            sample = new_sess.run(sample_from_batch(audio, 3))
            print("saving", latent_name, feature_name)
            save(sample, _z_val, latent_name + "-" + feature_name, 16000)
            print("done!", latent_name, feature_name)



#tell how related to eachother
experiment()