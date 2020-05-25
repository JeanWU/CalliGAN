# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import parser
from cns_encoder_font_attention import UNet

def main(_):
    args = parser.arg_parse()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = UNet(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                     input_width=args.image_size, output_width=args.image_size, L1_penalty=args.L1_penalty,
                     cns_embedding_size=args.cns_embedding_size)
        model.register_session(sess)
        model.build_model(is_training=True, inst_norm=args.inst_norm)
        model.train(lr=args.lr, epoch=args.epoch, resume=args.resume,
                    schedule=args.schedule, flip_labels=args.flip_labels, sample_steps=args.sample_steps)


if __name__ == '__main__':
    tf.app.run()


# CUDA_VISIBLE_DEVICES=2 python pretrain_cns_font_attention.py --experiment_dir=../experiment_font --experiment_id=2 --batch_size=512 --lr=0.001 --epoch 40 --sample_steps=50 --schedule=20 --L1_penalty=100 --cns_embedding_size=24 --image_size=64
# CUDA_VISIBLE_DEVICES=3 python pretrain_cns_font_attention.py --experiment_dir=../experiment_font --experiment_id=5 --batch_size=512 --lr=0.001 --epoch 40 --sample_steps=50 --schedule=20 --L1_penalty=100 --cns_embedding_size=64 --image_size=64
# CUDA_VISIBLE_DEVICES=3 python pretrain_cns_font_attention.py --experiment_dir=../experiment_font --experiment_id=7 --batch_size=512 --lr=0.001 --epoch 40 --sample_steps=50 --schedule=20 --L1_penalty=100 --cns_embedding_size=128 --image_size=64
