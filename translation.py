import os
from mxnet import np, npx, init
import mxnet as mx
#from d2l import mxnet as d2l
import utils
import gluonnlp as nlp

npx.set_np(shape=True, array=True)

data_dir = './data'
vocab_size = 10000
batch_size = 128

def load_data_token(path, fname):
    post1 = []
    post2 = []
    post3 = []
    post4 = []
    # with open('%s/%s.post' % (path, fname)) as f:
    #     for line in f:
    #         tmp = line.strip().split("\t")
    #         #post1.append([p.lower().split() for p in tmp])
    #         post1 = [l.strip().lower().split() for l in tmp[0]]

    with open('%s/%s.post' % (path, fname), encoding="latin1") as f:
        # post1 = [line.strip().lower().split("\t")[0].split() for line in f.readlines()]
        # post2 = [line.strip().lower().split("\t")[1].split() for line in f.readlines()]
        # post3 = [line.strip().lower().split("\t")[2].split() for line in f.readlines()]
        # post4 = [line.strip().lower().split("\t")[3].split() for line in f.readlines()]
        for line in f.readlines():
            l = line.strip().lower().split("\t")
            post1.append([l[0]])
            post2.append([l[1]])
            post3.append([l[2]])
            post4.append([l[3]])

    with open('%s/%s.response' % (path, fname), encoding="latin1") as f:
        response = [line.strip().lower().split() for line in f.readlines()]
    # data = []
    # for p, r in zip(post, response):
    #     data.append({'post': p, 'response': r})

    return post1, post2, post3, post4, response

def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = np.array([
        truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])

    valid_len = (array != vocab['<pad>']).astype(np.int32).sum(1)

    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    source1, source2, source3, source4, target = load_data_token(data_dir, 'train')
    #Vocabulário único
    vocab = utils.Vocab(source1 + source2 + source3 + source4 + target, min_freq=5,
                          reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])

    #tgt_vocab = utils.Vocab(target, min_freq=5,
    #                     reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])
    src_array1, src_valid_len1 = build_array_nmt(source1, vocab, num_steps)
    src_array2, src_valid_len2 = build_array_nmt(source2, vocab, num_steps)
    src_array3, src_valid_len3 = build_array_nmt(source3, vocab, num_steps)
    src_array4, src_valid_len4 = build_array_nmt(source4, vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, vocab, num_steps)
    data_arrays = (src_array1, src_valid_len1, src_array2, src_valid_len2, src_array3, src_valid_len3, src_array4, src_valid_len4, tgt_array, tgt_valid_len)
    data_iter = utils.load_array(data_arrays, batch_size, is_train=False)
    return data_iter, vocab, vocab

#source, target = load_data_token(data_dir, 'train')
#print(len(source))
#exit()
#src_vocab = utils.Vocab(source, min_freq=5, reserved_tokens=['<pad>', '<bos>', '<eos>'])

#encoder_len = [max([len(item[i]) for item in source]) + 1 for i in range(4)]
#decoder_len = max([len(item) for item in target]) + 1


#train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)

# for X, X_valid_len, Y, Y_valid_len in train_iter:
#     print('X:', X.astype(np.int32))
#     print('valid lengths for X:', X_valid_len)
#     print('Y:', Y.astype(np.int32))
#     print('valid lengths for Y:', Y_valid_len)
#     break

embed_size, num_hiddens, num_layers, dropout = 200, 512, 2, 0.1
batch_size, num_steps = 128, 10
lr, num_epochs, device = 0.005, 5, utils.try_gpu()

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = utils.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
encoder.initialize(init.Xavier())
glove_6b200d = nlp.embedding.create('glove', source='glove.6B.200d')
encoder.embedding.weight.set_data(glove_6b200d.idx_to_vec[:len(src_vocab)].as_np_ndarray())


decoder = utils.Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

net = utils.EncoderDecoder(encoder, decoder)

utils.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
src_sentence = "Martha is cooking a special meal for her family .	She wants everything to be just right for when they eat .	Martha perfects everything and puts her dinner into the oven .	Martha goes to lay down for a quick nap ."
output, attention_weights = utils.predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=True)
print("sentence", src_sentence)
print("saida:", output)





