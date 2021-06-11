
def load_data(path, fname):
    post = []
    with open('%s/%s.post' % (path, fname)) as f:
        for line in f:
            print(line)
            tmp = line.strip().split("\t")
            print(tmp)
            post.append([p.split() for p in tmp])
            print(post)
            exit()

    with open('%s/%s.response' % (path, fname)) as f:
        response = [line.strip().split() for line in f.readlines()]
    data = []
    for p, r in zip(post, response):
        data.append({'post': p, 'response': r})
    return data

def load_data_imdb(batch_size, num_steps=500):
    train_tokens = aux.tokenize(train_pd['text_pt'], token='word')
    test_tokens = aux.tokenize(test_pd['text_pt'], token='word')
    vocab = aux.Vocab(train_tokens, min_freq=5)
    train_features = np.array([
        aux.truncate_pad(vocab[line], num_steps, vocab['<pad>'])
        for line in train_tokens])
    test_features = np.array([
        aux.truncate_pad(vocab[line], num_steps, vocab['<pad>'])
        for line in test_tokens])
    train_iter = aux.load_array((train_features, train_pd['sentiment']), batch_size)
    test_iter = aux.load_array((test_features, test_pd['sentiment']), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab

load_data('./data','train')