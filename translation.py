import os
from mxnet import np, npx
#from d2l import mxnet as d2l
import utils

npx.set_np()

data_dir = './data'
vocab_size = 10000
batch_size = 128

def load_data_token(path, fname):
    post = []
    # with open('%s/%s.post' % (path, fname)) as f:
    #     for line in f:
    #         #tmp = line.strip().split("\t")
    #         tmp = line.strip().split()
    #         post.append([p.lower().split() for p in tmp])

    with open('%s/%s.post' % (path, fname)) as f:
        post = [line.strip().lower().split() for line in f.readlines()]

    with open('%s/%s.response' % (path, fname)) as f:
        response = [line.strip().lower().split() for line in f.readlines()]
    # data = []
    # for p, r in zip(post, response):
    #     data.append({'post': p, 'response': r})

    return post, response

def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    if isinstance(lines[0][0], list):
        lines = [token for line in lines for token in line]
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = np.array([
        truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])

    if (len(array) > 10000):
        valid_len = (array[0:10000] != vocab['<pad>']).astype(np.int32).sum(1)
    else:
        valid_len = (array != vocab['<pad>']).astype(np.int32).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    source, target = load_data_token(data_dir, 'train')
    src_vocab = utils.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = utils.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = utils.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

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

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 128, 10
lr, num_epochs, device = 0.005, 20, utils.try_gpu()

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = utils.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
decoder = utils.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
net = utils.EncoderDecoder(encoder, decoder)

utils.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
src_sentence = "Donald was walking around the lake ."
output, attention_weights = utils.predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False)
print(src_sentence)
print(output)





