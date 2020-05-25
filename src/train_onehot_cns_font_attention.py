# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import parser
from unet_onehot_cns_font_attention import UNet

def main(_):
    args = parser.arg_parse()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = UNet(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                     input_width=args.image_size, output_width=args.image_size, embedding_num=args.embedding_num,
                     L1_penalty=args.L1_penalty, Lconst_penalty=args.Lconst_penalty,
                     Ltv_penalty=args.Ltv_penalty, Lcategory_penalty=args.Lcategory_penalty,
                     cns_encoder_dir=args.cns_encoder_dir, cns_embedding_size=args.cns_embedding_size)
        model.register_session(sess)
        if args.flip_labels:
            model.build_model(is_training=True, inst_norm=args.inst_norm, no_target_source=True)
        else:
            model.build_model(is_training=True, inst_norm=args.inst_norm)
        fine_tune_list = None
        if args.fine_tune:
            ids = args.fine_tune.split(",")
            fine_tune_list = set([int(i) for i in ids])
        model.train(lr=args.lr, epoch=args.epoch, resume=args.resume,
                    schedule=args.schedule, freeze_encoder=args.freeze_encoder, fine_tune=fine_tune_list,
                    sample_steps=args.sample_steps, flip_labels=args.flip_labels)


if __name__ == '__main__':
    tf.app.run()


# single: CUDA_VISIBLE_DEVICES=0 python train_onehot_cns.py --experiment_dir=../experiment_single/experiment_single_2 --experiment_id=0 --batch_size=16 --lr=0.001 --epoch 40 --sample_steps=50 --schedule=20 --L1_penalty=100 --Lconst_penalty=15 --cns_embedding_size=128
# resume: CUDA_VISIBLE_DEVICES=2 python train_onehot_cns.py --experiment_dir=../experiment_single/experiment_single_0 --experiment_id=5 --batch_size=16 --lr=0.0002 --epoch 20 --sample_steps=50 --schedule=20 --L1_penalty=100 --Lconst_penalty=15
# multiple: CUDA_VISIBLE_DEVICES=0 python train_cns_onehot_font.py --experiment_dir=../experiment_multiple --experiment_id=0 --batch_size=16 --lr=0.001 --epoch 40 --sample_steps=50 --schedule=20 --L1_penalty=500 --Lconst_penalty=100
