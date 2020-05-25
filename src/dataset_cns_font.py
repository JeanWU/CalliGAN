# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import cPickle as pickle
import numpy as np
import random
import scipy.misc as misc
import os
from utils import pad_seq, bytes_to_file, shift_and_resize_image, normalize_image


class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 1000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples


def get_batch_iter(examples, batch_size, augment):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    padded = pad_seq(examples, batch_size)

    def process(img):
        image = bytes_to_file(img)
        try:
            img = misc.imread(image).astype(np.float)
            if augment:
                # augment the image by:
                # 1) enlarge the image
                # 2) random crop the image back to its original size
                # NOTE: image A and B needs to be in sync as how much
                # to be shifted
                w, h = img.shape
                multiplier = random.uniform(1.00, 1.20)
                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1
                shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
                shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
                img = shift_and_resize_image(img, shift_x, shift_y, nw, nh)
            img = normalize_image(img)
            img = np.expand_dims(img,axis=2)
            return img
        finally:
            image.close()

    def handle_cns(cns_code):
        cns_code_batch = []
        seq_len = []
        max_len = 28   # get max_len from check_cns_len.py
        for cns in cns_code:
            num = list(map(int, cns.split(',')))
            seq_len.append(len(num))
            num += [0] * (max_len-len(num))
            cns_code_batch.append(num)
        return cns_code_batch, seq_len

    def batch_iter():
        for i in range(0, len(padded), batch_size):
            batch = padded[i: i + batch_size]
            cns_code = [e[0] for e in batch]
            labels = [0 for e in batch]
            processed = [process(e[1]) for e in batch]
            cns_code, seq_len = handle_cns(cns_code)
            # stack into tensor
            yield cns_code, seq_len, labels, np.array(processed).astype(np.float32)

    return batch_iter()


class TrainDataProvider(object):
    def __init__(self, data_dir, train_name="cns_font_train.obj", val_name="cns_font_val.obj", filter_by=None):
        self.data_dir = data_dir
        self.filter_by = filter_by
        self.train_path = os.path.join(self.data_dir, train_name)
        self.val_path = os.path.join(self.data_dir, val_name)
        self.train = PickledImageProvider(self.train_path)
        self.val = PickledImageProvider(self.val_path)
        if self.filter_by:
            print("filter by label ->", filter_by)
            self.train.examples = filter(lambda e: e[0] in self.filter_by, self.train.examples)
            self.val.examples = filter(lambda e: e[0] in self.filter_by, self.val.examples)
        print("train examples -> %d, val examples -> %d" % (len(self.train.examples), len(self.val.examples)))

    def get_train_iter(self, batch_size, shuffle=True):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
        return get_batch_iter(training_examples, batch_size, augment=True)

    def get_val_iter(self, batch_size, shuffle=True):
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.shuffle(val_examples)
        return get_batch_iter(val_examples, batch_size, augment=True)

    def get_val_iter_bk(self, batch_size, shuffle=True):
        """
        Validation iterator runs forever
        """
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.shuffle(val_examples)
        while True:
            val_batch_iter = get_batch_iter(val_examples, batch_size, augment=False)
            for cns_code, seq_len, labels, examples in val_batch_iter:
                yield cns_code, seq_len, labels, examples

    def compute_total_batch_num(self, batch_size):
        """Total padded batch num"""
        return int(np.ceil(len(self.train.examples) / float(batch_size)))

    def get_all_labels(self):
        """Get all training labels"""
        return list({e[0] for e in self.train.examples})

    def get_train_val_path(self):
        return self.train_path, self.val_path


class InjectDataProvider(object):
    def __init__(self, obj_path):
        self.data = PickledImageProvider(obj_path)
        print("examples -> %d" % len(self.data.examples))

    def get_single_embedding_iter(self, batch_size, embedding_id):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, augment=False)
        for cns_code, seq_len, _, images in batch_iter:
            # inject specific embedding style here
            labels = [embedding_id] * batch_size
            yield cns_code, seq_len, labels, images

    def get_random_embedding_iter(self, batch_size, embedding_ids):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, augment=False)
        for cns_code, seq_len, _, images in batch_iter:
            # inject specific embedding style here
            labels = [random.choice(embedding_ids) for i in range(batch_size)]
            yield cns_code, seq_len, labels, images

"""
class NeverEndingLoopingProvider(InjectDataProvider):
    def __init__(self, obj_path):
        super(NeverEndingLoopingProvider, self).__init__(obj_path)

    def get_random_embedding_iter(self, batch_size, embedding_ids):
        while True:
            # np.random.shuffle(self.data.examples)
            rand_iter = super(NeverEndingLoopingProvider, self) \
                .get_random_embedding_iter(batch_size, embedding_ids)
            for labels, images in rand_iter:
                yield labels, images
"""
