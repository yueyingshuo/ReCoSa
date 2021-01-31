# -*- coding:utf-8 -*-
import numpy as np
import collections
import hyperparams as hp
import os
from  data_load import get_batch_data,create_data
import csv
# class DialogBatchInput(
#     collections.namedtuple("DialogBatchInput",
#                            ("source",
#                             "source_length",
#                             "source_topic",
#                             "target_input",
#                             "target_output",
#                             "target_length"))):
#     pass


class DialogBucket(object):
    def __init__(self,  batch_size):
        self._batch_size = batch_size
        self._bucket_data = []


    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_samples(self):
        return len(self._bucket_data)

    def add_sample(self, sample):
        self._bucket_data.append(sample)

    def generate_batches(self, shuffle=False):
        idx = list(range(self.num_samples))
        if shuffle:
            np.random.shuffle(idx)
        batches = []
        start_idx = 0
        while start_idx < self.num_samples:
            end_idx = start_idx + self.batch_size
            if end_idx > self.num_samples:
                end_idx = self.num_samples
            batch_idx = idx[start_idx:end_idx]
            batch_data = [self._bucket_data[i] for i in batch_idx]
            batches.append(batch_data)
            start_idx += self.batch_size
        return batches


class DialogIterator(object):
    def __init__(self, dialog_file,
                 batch_size,
                 max_len,
                 shuffle=False):

        self._dialog_file = dialog_file
        self._batch_size = batch_size
        self._max_len = max_len
        self._shuffle = shuffle

        # generate source response pairs
        ctx_resp_pairs = self.generate_context_response_pairs(self._dialog_file)
        self._num_samples = len(ctx_resp_pairs)
        self._dialog_buckets = self.make_bucket_data(ctx_resp_pairs)


    @property
    def num_samples(self):
        return self._num_samples

    def generate_context_response_pairs(self, dialog_file):
        ctx_resp_pairs = []
        with open(dialog_file, "r", encoding="utf-8") as read:
            csv_read = csv.reader(read)
            for line in csv_read:
                fields = [s.replace('__eou__', '.').replace('__eot__', '\n').strip() for s in line]
                context = fields[0]
                response = fields[1]
                ctx_resp_pairs.append((context,response))
        return ctx_resp_pairs




    def make_bucket_data(self, ctx_resp_pairs):
        # init dialog buckets from config
        dialog_buckets = []
        num_batch = self._num_samples // hp.batch_size
        for i in range(num_batch):
            dialog_bucket = DialogBucket(hp.batch_size)
            dialog_buckets.append(dialog_bucket)

        for (ctx,resp) in ctx_resp_pairs:
            turn_size = len(ctx)
            for dialog_bucket in dialog_buckets:
                if turn_size > dialog_bucket.bucket_size:
                    continue
                dialog_bucket.add_sample((ctx,resp))
                break
            else:
                if self._max_turn is not None:
                    assert len(ctx) <= self._max_turn
                if len(ctx) <= dialog_buckets[0].bucket_size:
                    dialog_buckets[0].add_sample((ctx, resp))
                else:
                    dialog_buckets[-1].add_sample((ctx, resp))
        bucket_samples = [dialog_bucket.num_samples for dialog_bucket in dialog_buckets]
        print('total bucket samples:\t', sum(bucket_samples))
        return dialog_buckets

    def _dynamic_padding(self, batch_data):
        lengths = [len(x) for x in batch_data]
        max_len = max(lengths)
        if self._max_len is not None and max_len > self._max_len:
            max_len = self._max_len

        pad_data = []
        pad_lengths = []
        for data, length in zip(batch_data, lengths):
            if length > max_len:
                pad_data += [data[:max_len]]
                pad_lengths += [max_len]
            else:
                pad_data += [data + [self._pad_idx] * (max_len - length)]
                pad_lengths += [length]

        pad_data = np.asarray(pad_data, dtype='int32')
        pad_lengths = np.asarray(pad_lengths, dtype='int32')
        return pad_data, pad_lengths

    def _dynamic_padding_multi_turn(self, batch_data,batch_topic):
        turns = [len(d) for d in batch_data]
        sent_lengths = [[len(sent) for sent in d] for d in batch_data]
        max_turn = max(turns)
        max_length = max(max(sent_lengths))

        if self._max_turn is not None and self._max_turn < max_turn:
            max_turn = self._max_turn
        if self._max_len is not None and self._max_len < max_length:
            max_length = self._max_len

        pad_dialog_data, pad_dialog_lengths,pad_dialog_topic = [], [],[]
        for turn, turn_data,turn_topic in zip(turns, batch_data,batch_topic):
            if turn > max_turn:
                turn_data = turn_data[-max_turn:]
            pad_sents = []
            pad_sent_lengths = []
            pad_sent_topic = []
            pad_sent_max_length = max_length
            for sent,tp in zip(turn_data,turn_topic):
                if len(sent) > max_length:
                    sent = sent[:max_length]

                pad_sents.append(sent + [self._pad_idx] * (pad_sent_max_length - len(sent)))
                pad_sent_lengths.append(len(sent))
                pad_sent_topic.append(tp)

            while len(pad_sents) < max_turn:
                pad_sents += [[self._pad_idx] * pad_sent_max_length]
                pad_sent_lengths += [0]
                pad_sent_topic+=[0]

            pad_dialog_data += [pad_sents]
            pad_dialog_lengths += [pad_sent_lengths]
            pad_dialog_topic+=[pad_sent_topic]

        pad_dialog_data = np.asarray(pad_dialog_data, dtype='int32')
        pad_sent_lengths = np.asarray(pad_dialog_lengths, dtype='int32')

        return pad_dialog_data,pad_dialog_topic, pad_sent_lengths

    def next_batch(self):
        batch_samples = []
        if not self._infer:
            for dialog_bucket in self._dialog_buckets:
                batch_samples += dialog_bucket.generate_batches(self._shuffle)

            # shuffle by batches
            if self._shuffle:
                np.random.shuffle(batch_samples)

            for batch_data in batch_samples:
                ctx_data = []
                ctx_topic = []
                resp_input_data = []
                resp_output_data = []
                for (ctx,tp, resp) in batch_data:
                    ctx_data.append(ctx)
                    ctx_topic.append(tp)
                    resp_input_data += [[self._sos_idx] + resp]
                    resp_output_data += [resp + [self._eos_idx]]

                pad_ctx,pad_ctx_topic, pad_ctx_length = self._dynamic_padding_multi_turn(ctx_data,ctx_topic)
                pad_resp_input, resp_length = self._dynamic_padding(resp_input_data)
                pad_resp_output, _ = self._dynamic_padding(resp_output_data)
                yield DialogBatchInput(source=pad_ctx,
                                       source_length=pad_ctx_length,
                                       source_topic = pad_ctx_topic,
                                       target_input=pad_resp_input,
                                       target_output=pad_resp_output,
                                       target_length=resp_length)

        else:
            # only use context when infer
            batch_ctx_data = []
            batch_ctx_topic = []
            for ctx_data,tp in zip(self._dialog_data,self._dialog_topic):
                batch_ctx_data += [ctx_data]
                batch_ctx_topic +=[tp]
                if len(batch_ctx_data) == self._batch_size:
                    pad_ctx,pad_ctx_topic, pad_ctx_length = self._dynamic_padding_multi_turn(batch_ctx_data,batch_ctx_topic)
                    yield DialogBatchInput(source=pad_ctx,
                                           source_length=pad_ctx_length,
                                           source_topic=pad_ctx_topic,
                                           target_input=None,
                                           target_output=None,
                                           target_length=None)
                    batch_ctx_data = []
                    batch_ctx_topic = []
            if len(batch_ctx_data) > 0:
                pad_ctx,pad_ctx_topic, pad_ctx_length = self._dynamic_padding_multi_turn(batch_ctx_data,batch_ctx_topic)
                yield DialogBatchInput(source=pad_ctx,
                                       source_length=pad_ctx_length,
                                       source_topic=pad_ctx_topic,
                                       target_input=None,
                                       target_output=None,
                                       target_length=None)


def get_dialog_data_iter(vocab,
                         dialog_file,
                         batch_size,
                         max_len=None,
                         shuffle=False):
    dialog_data = load_dialog_data(dialog_file, '</d>')
    dialog_idx = [[vocab.convert2idx(sent[1:]) for sent in dialog] for dialog in dialog_data]
    #dialog_topic = []
    #for dialog in dialog_data:
    #    print(dialog)
    #    dialog_topic.append([sent[0] for sent in dialog])
    #print(dialog_data)
    data_iter = DialogIterator(vocab=vocab,
                               dialog_data=dialog_idx,
                               batch_size=batch_size,
                               max_len=max_len,
                               shuffle=shuffle)
    return data_iter




def get_train_iter(data_dir, vocab, config):

    vocab_path = "/home/yueying/pycharm_workspace/Recose/preprocessed/train.vocab.tsv"
    X, X_length, Y, Sources, Targets = create_data(hp.train_set, vocab_path)

    train_file = os.path.join(data_dir, 'train.csv')
    train_iter = get_dialog_data_iter(vocab=vocab,
                                      dialog_file=train_file,
                                      batch_size=hp.batch_size,
                                      max_len=hp.maxlen,
                                      shuffle=True)

    valid_file = os.path.join(data_dir, 'valid.csv')
    valid_iter = get_dialog_data_iter(vocab=vocab,
                                      dialog_file=valid_file,
                                      batch_size=hp.batch_size,
                                      max_len=hp.maxlen,
                                      shuffle=True)

    test_file = os.path.join(data_dir, 'test.csv')
    test_iter = get_dialog_data_iter(vocab=vocab,
                                      dialog_file=test_file,
                                      batch_size=hp.batch_size,
                                      max_len=hp.maxlen,
                                      shuffle=True)

    return train_iter, valid_iter, test_iter


def get_infer_iter(context_file, vocab, config):
    infer_iter = get_dialog_data_iter(vocab=vocab,
                                      dialog_file=context_file,
                                      batch_size=config.infer_batch_size,
                                      max_len=config.max_len,
                                      max_turn=config.max_turn,
                                      model=config.model,
                                      infer=True,
                                      shuffle=False)
    return infer_iter
