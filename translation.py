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

    with open('%s/%s.post' % (path, fname), encoding="latin1") as f:
        post = [line.strip().lower().split() for line in f.readlines()]

    with open('%s/%s.response' % (path, fname), encoding="latin1") as f:
        response = [line.strip().lower().split() for line in f.readlines()]

    return post, response

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
    source, target = load_data_token(data_dir, 'train')
    #Vocabulário único
    vocab = utils.Vocab(source + target, min_freq=5,
                          reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])

    #tgt_vocab = utils.Vocab(target, min_freq=5,
    #                     reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])
    src_array, src_valid_len = build_array_nmt(source, vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = utils.load_array(data_arrays, batch_size)

    return data_iter, vocab, vocab

embed_size, num_hiddens, num_layers, dropout = 200, 512, 2, 0.1
batch_size, num_steps = 128, 60
lr, num_epochs, device = 0.005, 30, utils.try_gpu()

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = utils.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
encoder.initialize(init.Xavier())
glove_6b200d = nlp.embedding.create('glove', source='glove.6B.200d')
encoder.embedding.weight.set_data(glove_6b200d.idx_to_vec[:len(src_vocab)].as_np_ndarray())


decoder = utils.Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

net = utils.EncoderDecoder(encoder, decoder)

utils.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

sentence1 = "My wife is collecting unemployment insurance .	The state required that she meet with a counselor .	She had to show her resume and work search logs .	My wife dreaded the meeting ."
resposta1 = "She went and said it was actually not all that bad ."

sentence2 = "Lucy loved computer coding .	So she decided to make her own app .	She then published it to the app store .	To her delight she made a lot of money ."
resposta2 = "Lucy was thrilled ."

sentence3 = "I was reading the novel Vanity Fair last week .	My wife was out food shopping .	She texted me asking if I wanted beer or ale .	I said any ale , but texted her later asking for English beer ."
resposta3 = "Unfortunately she did get the text and I got domestic beer ."

sentence4 = "Donald was walking around the lake .	He saw a duck swimming in it .	The duck paddled silently through the water .	When he was at the other side , he climbed out ."
resposta4 = "Donald loved watching the duck in nature ."

num_steps = 15
output1, attention_weights1 = utils.predict_seq2seq(net, sentence1, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=True)
print("sentence: ", sentence1)
print("resposta: ", resposta1)
print("saida: ", output1)

output2, attention_weights2 = utils.predict_seq2seq(net, sentence2, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=True)
print("sentence", sentence2)
print("resposta: ", resposta2)
print("saida:", output2)

output3, attention_weights3 = utils.predict_seq2seq(net, sentence3, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=True)
print("sentence", sentence3)
print("resposta: ", resposta3)
print("saida:", output3)

output4, attention_weights4 = utils.predict_seq2seq(net, sentence4, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=True)
print("sentence", sentence4)
print("resposta: ", resposta4)
print("saida:", output4)





