# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import time
from collections import namedtuple
from ops import conv2d, deconv2d, lrelu, fc, batch_norm, init_embedding, conditional_instance_norm
from dataset_cns_font import TrainDataProvider, InjectDataProvider
from utils import scale_back, merge, save_concat_images
from tensorflow.python.ops import embedding_ops
from transformer_modules import get_token_embeddings, ff, positional_encoding, multihead_attention

# Auxiliary wrapper classes
# Used to save handles(important nodes in computation graph) for later evaluation
LossHandle = namedtuple("LossHandle", ["l1_loss"])
InputHandle = namedtuple("InputHandle", ["real_data", "embedding_ids", "cns_code", "seq_len"])
EvalHandle = namedtuple("EvalHandle", ["generator", "target", "source"])

'''
onehot + cns 
'''

class UNet(object):
    def __init__(self, experiment_dir=None, experiment_id=0, batch_size=16, input_width=64, output_width=64, generator_dim=64,
                 L1_penalty=100, input_filters=1, output_filters=1, cns_embedding_size=128):
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.input_width = input_width
        self.output_width = output_width
        self.generator_dim = generator_dim
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.L1_penalty = L1_penalty
        self.cns_vocab_size = 518
        self.cns_embedding_size = cns_embedding_size
        self.num_blocks = 3  # number of encoder/decoder blocks
        self.num_heads = 8  # number of attention heads
        self.d_ff = 512
        self.font_len = 28
        # init all the directories
        self.sess = None
        # experiment_dir is needed for training
        if experiment_dir:
            self.data_dir = os.path.join(self.experiment_dir, "data")
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
            self.sample_dir = os.path.join(self.experiment_dir, "sample")
            self.log_dir = os.path.join(self.experiment_dir, "logs")

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print("create checkpoint directory")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("create log directory")
            if not os.path.exists(self.sample_dir):
                os.makedirs(self.sample_dir)
                print("create sample directory")

    def decoder(self, encoded, ids, inst_norm, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = self.output_width
            s2, s4, s8, s16, s32 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32)

            def decode_layer(x, output_width, output_filters, layer, dropout=False):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_width,
                                               output_width, output_filters], scope="g_d%d_deconv" % layer)
                if layer != 8:
                    # IMPORTANT: normalization for last layer
                    # Very important, otherwise GAN is unstable
                    # Trying conditional instance normalization to
                    # overcome the fact that batch normalization offers
                    # different train/test statistics
                    dec = batch_norm(dec, is_training, scope="g_d%d_bn" % layer)
                if dropout:
                    dec = tf.nn.dropout(dec, 0.5)
                return dec

            d3 = decode_layer(encoded, s32, self.generator_dim * 8, layer=3, dropout=True)
            d4 = decode_layer(d3, s16, self.generator_dim * 8, layer=4, dropout=True)
            d5 = decode_layer(d4, s8, self.generator_dim * 4, layer=5)
            d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6)
            d7 = decode_layer(d6, s2, self.generator_dim, layer=7)
            d8 = decode_layer(d7, s, self.output_filters, layer=8)

            output = tf.nn.tanh(d8)  # scale to (-1, 1)
            return output

    def encoder(self, cns_code, seq_len, reuse=False):
        with tf.variable_scope("cns_encoder"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # src_masks
            src_masks = tf.math.equal(cns_code, 0)  # (N, T1)

            # embedding
            embedding_encoder = tf.get_variable("embedding_encoder", [self.cns_vocab_size, self.cns_embedding_size])
            enc = tf.nn.embedding_lookup(embedding_encoder, cns_code)  # (N, T1, d_model)
            enc *= self.cns_embedding_size**0.5  # scale

            enc += positional_encoding(enc, self.font_len)
            enc = tf.layers.dropout(enc, 0.3, training=True)

            # Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.num_heads,
                                              dropout_rate=0.3,
                                              training=True,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.d_ff, self.cns_embedding_size])
        memory = enc
        return memory

    def generator(self, embedding_ids, cns_code, seq_len, inst_norm, is_training, reuse=False):
        z = self.encoder(cns_code, seq_len, reuse=reuse)
        encoder_state = tf.reshape(z, [self.batch_size, 1, 1, self.cns_embedding_size * self.font_len])  # max_len * embedding size
        output = self.decoder(encoder_state, embedding_ids, inst_norm, is_training=is_training, reuse=reuse)
        return output

    def build_model(self, is_training=True, inst_norm=False):
        real_data = tf.placeholder(tf.float32,
                                   [self.batch_size, self.input_width, self.input_width, self.input_filters],
                                   name='real_image')
        embedding_ids = tf.placeholder(tf.int64, shape=None, name="embedding_ids")
        cns_code = tf.placeholder(tf.int64, shape=[None, None], name="cns_code")
        seq_len = tf.placeholder(tf.int64, shape=None, name="seq_len")

        # embedding = init_embedding(self.embedding_num, self.embedding_dim)
        fake = self.generator(embedding_ids, cns_code, seq_len, is_training=is_training, inst_norm=inst_norm)

        # L1 loss between real and generated images
        l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake - real_data))
        # latent_loss = - 0.5 * tf.reduce_sum(1 + z_sigma - z_mean**2 - tf.exp(z_sigma), 1)
        # total_loss = tf.reduce_mean(l1_loss + self.kulback_coef * latent_loss)

        # expose useful nodes in the graph as handles globally
        input_handle = InputHandle(real_data=real_data,
                                   embedding_ids=embedding_ids,
                                   cns_code=cns_code,
                                   seq_len=seq_len)

        loss_handle = LossHandle(l1_loss=l1_loss)

        eval_handle = EvalHandle(generator=fake,
                                 target=real_data,
                                 source=real_data)

        # those operations will be shared, so we need
        # to make them visible globally
        setattr(self, "input_handle", input_handle)
        setattr(self, "loss_handle", loss_handle)
        setattr(self, "eval_handle", eval_handle)

    def register_session(self, sess):
        self.sess = sess

    def retrieve_generator_vars(self):
        all_vars = tf.global_variables()
        generate_vars = [var for var in all_vars if 'generator' in var.name or "cns_encoder" in var.name]
        return generate_vars

    def retrieve_handles(self):
        input_handle = getattr(self, "input_handle")
        loss_handle = getattr(self, "loss_handle")
        eval_handle = getattr(self, "eval_handle")

        return input_handle, loss_handle, eval_handle

    def get_model_id_and_dir(self):
        model_id = "experiment_%d_batch_%d" % (self.experiment_id, self.batch_size)
        model_dir = os.path.join(self.checkpoint_dir, model_id)
        return model_id, model_dir

    def checkpoint(self, saver, step):
        model_name = "unet.model"
        model_id, model_dir = self.get_model_id_and_dir()

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

    def restore_model(self, saver, model_dir):

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)

    def generate_fake_samples(self, input_images, embedding_ids, cns_code, seq_len):
        input_handle, loss_handle, eval_handle = self.retrieve_handles()
        fake_images, real_images, l1_loss = self.sess.run([eval_handle.generator,
                                                 eval_handle.target,
                                                 loss_handle.l1_loss],
                                                feed_dict={
                                                    input_handle.real_data: input_images,
                                                    input_handle.embedding_ids: embedding_ids,
                                                    input_handle.cns_code: cns_code,
                                                    input_handle.seq_len: seq_len
                                                })
        return fake_images, real_images, l1_loss

    def validate_model(self, val_iter, epoch, step):
        cns_code, seq_len, labels, images = next(val_iter)
        fake_imgs, real_imgs, l1_loss = self.generate_fake_samples(images, labels, cns_code, seq_len)
        print("Sample: l1_loss: %.5f" % (l1_loss))

        merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size/16, 16])
        merged_real_images = merge(scale_back(real_imgs), [self.batch_size/16, 16])
        merged_pair = np.concatenate([merged_real_images, merged_fake_images], axis=1)

        model_id, _ = self.get_model_id_and_dir()

        model_sample_dir = os.path.join(self.sample_dir, model_id)
        if not os.path.exists(model_sample_dir):
            os.makedirs(model_sample_dir)

        sample_img_path = os.path.join(model_sample_dir, "sample_%02d_%04d.jpg" % (epoch, step))
        misc.imsave(sample_img_path, merged_pair)
        return l1_loss

    def validate_all(self, val_batch_iter):
        test = []
        for bid, batch in enumerate(val_batch_iter):
            cns_code, seq_len, labels, images = batch
            fake_imgs, real_imgs, l1_loss = self.generate_fake_samples(images, labels, cns_code, seq_len)
            # print(l1_loss)
            test.append(l1_loss)
        return sum(test)/len(test)

    def infer(self, source_obj, embedding_ids, model_dir, save_dir):
        source_provider = InjectDataProvider(source_obj)

        if isinstance(embedding_ids, int) or len(embedding_ids) == 1:
            embedding_id = embedding_ids if isinstance(embedding_ids, int) else embedding_ids[0]
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, embedding_id)
        else:
            source_iter = source_provider.get_random_embedding_iter(self.batch_size, embedding_ids)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)

        def save_imgs(imgs, count):
            p = os.path.join(save_dir, "inferred_%04d.jpg" % count)
            save_concat_images(imgs, img_path=p)
            print("generated images saved at %s" % p)

        count = 0
        batch_buffer = list()
        for cns_code, seq_len, labels, source_imgs in source_iter:
            fake_imgs = self.generate_fake_samples(source_imgs, labels, cns_code, seq_len)[0]
            img_path = os.path.join(save_dir, "inferred_%04d.jpg" % count)
            misc.imsave(img_path, fake_imgs.squeeze())
            count += 1

    def train(self, lr=0.0002, epoch=100, schedule=10, resume=True, flip_labels=False,
              fine_tune=None, sample_steps=50):
        # g_vars, d_vars = self.retrieve_trainable_vars(freeze_encoder=freeze_encoder)
        t_vars = tf.trainable_variables()
        print(t_vars)
        input_handle, loss_handle, _ = self.retrieve_handles()

        if not self.sess:
            raise Exception("no session registered")

        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.l1_loss, var_list=t_vars)

        tf.global_variables_initializer().run()
        real_data = input_handle.real_data
        embedding_ids = input_handle.embedding_ids
        cns_code = input_handle.cns_code
        seq_len = input_handle.seq_len

        # filter by one type of labels
        data_provider = TrainDataProvider(self.data_dir, filter_by=fine_tune)
        total_batches = data_provider.compute_total_batch_num(self.batch_size)
        # val_batch_iter = data_provider.get_val_iter(self.batch_size)

        saver = tf.train.Saver(max_to_keep=2)
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if resume:
            _, model_dir = self.get_model_id_and_dir()
            self.restore_model(saver, model_dir)

        current_lr = lr
        counter = 0
        start_time = time.time()
        best_l1loss = 100000

        for ei in range(epoch):
            train_batch_iter = data_provider.get_train_iter(self.batch_size)

            if (ei + 1) % schedule == 0:
                update_lr = current_lr / 2.0
                # minimum learning rate guarantee
                update_lr = max(update_lr, 0.0002)
                print("decay learning rate from %.5f to %.5f" % (current_lr, update_lr))
                current_lr = update_lr

            for bid, batch in enumerate(train_batch_iter):
                counter += 1
                cns, sequence_len, labels, batch_images = batch
                shuffled_ids = labels[:]
                if flip_labels:
                    np.random.shuffle(shuffled_ids)

                # Optimize cns_encoder and decoder
                _, batch_l1_loss = self.sess.run([optimizer, loss_handle.l1_loss],
                                                feed_dict={
                                                    real_data: batch_images,
                                                    embedding_ids: labels,
                                                    learning_rate: current_lr,
                                                    cns_code: cns,
                                                    seq_len: sequence_len
                                                })

                passed = time.time() - start_time
                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.4f, l1_loss: %.5f"
                print(log_format % (ei, bid, total_batches, passed, batch_l1_loss))

                if counter % sample_steps == 0:
                    # sample the current model states with val data
                    val_batch_iter = data_provider.get_val_iter(self.batch_size, shuffle=False)
                    valid_l1loss = self.validate_all(val_batch_iter)
                    print(valid_l1loss)
                    if valid_l1loss < best_l1loss:
                        best_l1loss = valid_l1loss
                        self.checkpoint(saver, counter)
                """
                if counter % checkpoint_steps == 0:
                    print("Checkpoint: save checkpoint step %d" % counter)
                    self.checkpoint(saver, counter)
                """
        # print("best_l1loss: ", best_l1loss)

        # save the last checkpoint
        print("Checkpoint: last checkpoint step %d" % counter)
        self.checkpoint(saver, counter)

