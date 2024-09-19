# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
import nltk


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class DeepAveragingNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int,
                 word_embeddings: torch.Tensor):

        super(DeepAveragingNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))

        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.relu = nn.ReLU()

    def forward(self, word_indices: torch.Tensor) -> torch.Tensor:

        batch_size, sequence_length = word_indices.size()

        embeddings = self.embedding(word_indices)

        averaged_embedding = embeddings.mean(dim=1)

        hidden = self.relu(self.fc(averaged_embedding))
        output = self.log_softmax(self.out(hidden))

        return output


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, model, word_embeddings):

        self.model = model
        self.word_embeddings = word_embeddings
        self.word_indexer = word_embeddings.word_indexer
        self.embedding_length = word_embeddings.get_embedding_length()

    def predict(self, ex_words: List[str], has_typos: bool) -> int:

        word_indices = [self.word_indexer.index_of(word) for word in ex_words]
        unk_index = self.word_indexer.index_of("UNK")
        word_indices = [index if index != -1 else unk_index for index in word_indices]
        word_indices_tensor = torch.tensor([word_indices], dtype=torch.long)

        with torch.no_grad():
            log_probs = self.model(word_indices_tensor)

        predicted_class = torch.argmax(log_probs, dim=1).item()

        return predicted_class


def prepare_batches(sentences, word_indexer, prefix_length):
    max_length = max(len(sentence.words) for sentence in sentences)
    word_indices_list = []
    labels = []

    for sentence in sentences:
        word_indices = []
        for word in sentence.words:

            #truncated_word = word[:prefix_length]
            truncated_word = word
            index = word_indexer.index_of(truncated_word)
            if index == -1:
                index = word_indexer.index_of("UNK")
            word_indices.append(index)

        word_indices += [0] * (max_length - len(word_indices))
        word_indices_list.append(word_indices)
        labels.append(sentence.label)
    return torch.tensor(word_indices_list, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def generate_batches(sentences, batch_size):
    batches = []
    for i in range(0, len(sentences), batch_size):
        batch = []
        for j in range(i, min(i + batch_size, len(sentences))):
            batch.append(sentences[j])
        batches.append(batch)
    return batches


def modify_embeddings_with_prefixes(word_embeddings, prefix_length=3):
    word_indexer = word_embeddings.word_indexer
    vectors = word_embeddings.vectors
    prefix_indexer = {}

    for word in word_indexer.objs_to_ints:

        index = word_indexer.index_of(word)

        if len(word) > prefix_length:
            prefix = word[:prefix_length]
            if prefix not in prefix_indexer:
                prefix_indexer[prefix] = []
            prefix_indexer[prefix].append(index)

    updated_vectors = np.copy(vectors)
    for prefix, indices in prefix_indexer.items():
        if len(indices) > 1:

            prefix_embedding = np.mean([vectors[i] for i in indices], axis=0)
            for j in indices:
                updated_vectors[j] = prefix_embedding

    word_embeddings.vectors = updated_vectors


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    vocab_size = len(word_embeddings.word_indexer)
    embedding_dim = word_embeddings.get_embedding_length()
    hidden_dim = 36
    output_dim = 2
    #batch_size = args.batch_size
    batch_size = 3

    #modify_embeddings_with_prefixes(word_embeddings, prefix_length=3)

    dan_model = DeepAveragingNetwork(vocab_size, embedding_dim, hidden_dim, output_dim,
                                     word_embeddings.vectors)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(dan_model.parameters(), lr=args.lr)

    correction_cache = {}

    num_epochs = 13
    for epoch in range(num_epochs):
        dan_model.train()

        total_loss = 0

        for batch in generate_batches(train_exs, batch_size):
            batch_data, batch_labels = prepare_batches(batch, word_embeddings.word_indexer, prefix_length=3)

            optimizer.zero_grad()

            log_probs = dan_model(batch_data)
            loss = criterion(log_probs, batch_labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {total_loss / len(train_exs)}')

    return NeuralSentimentClassifier(dan_model, word_embeddings)

