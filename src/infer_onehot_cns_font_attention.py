# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import parser
import tensorflow as tf
from unet_onehot_cns_font_attention import UNet


def main(_):
    args = parser.arg_parse()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = UNet(batch_size=args.batch_size, embedding_num=args.embedding_num, cns_embedding_size=args.cns_embedding_size)
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=args.inst_norm)
        embedding_ids = [int(i) for i in args.embedding_ids.split(",")]
        if not args.interpolate:
            if len(embedding_ids) == 1:
                embedding_ids = embedding_ids[0]
            model.infer(model_dir=args.model_dir, source_obj=args.source_obj, embedding_ids=embedding_ids,
                        save_dir=args.save_dir)
        else:
            if len(embedding_ids) < 2:
                raise Exception("no need to interpolate yourself unless you are a narcissist")
            chains = embedding_ids[:]
            if args.uroboros:
                chains.append(chains[0])
            pairs = list()
            for i in range(len(chains) - 1):
                pairs.append((chains[i], chains[i + 1]))
            for s, e in pairs:
                model.interpolate(model_dir=args.model_dir, source_obj=args.source_obj, between=[s, e],
                                  save_dir=args.save_dir, steps=args.steps)


if __name__ == '__main__':
    tf.app.run()


# single: CUDA_VISIBLE_DEVICES=2 python infer_onehot_cns.py --model_dir=../experiment_single/experiment_single_0/checkpoint/experiment_5_batch_16/ --batch_size=1 --source_obj=../experiment_single/experiment_single_0/data/cns_test.obj --save_dir ../experiment_single/all_results_5/style_0 --embedding_ids=0
# multiple: CUDA_VISIBLE_DEVICES=3 python infer_onehot_cns.py --model_dir=../experiment_multiple/checkpoint/experiment_9_batch_16/ --batch_size=1 --source_obj=../experiment_single/experiment_single_0/data/cns_test.obj --save_dir ../experiment_multiple/results_9/style_0 --embedding_ids=0
