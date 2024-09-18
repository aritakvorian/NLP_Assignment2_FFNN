# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


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
        """
        :param vocab_size: The size of the vocabulary (number of unique tokens).
        :param embedding_dim: The dimensionality of the word embeddings.
        :param hidden_dim: The number of hidden units in the fully connected layer.
        :param output_dim: The number of output classes (in this case, 2 for binary sentiment classification).
        :param word_embeddings: Pre-trained word embeddings to initialize the nn.Embedding layer.
        """
        super(DeepAveragingNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))

        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.relu = nn.ReLU()

    def forward(self, word_indices: torch.Tensor) -> torch.Tensor:
        """
        :param word_indices: A tensor containing the word indices for a sentence.
        :return: Log-probabilities over the classes (positive, negative).
        """

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
        """
        :param model: The trained neural network (DAN) instance.
        :param word_embeddings: Word embeddings used for mapping words to vectors.
        :param vocab: Vocabulary object to convert words into indices.
        """
        self.model = model
        self.word_embeddings = word_embeddings
        self.word_indexer = word_embeddings.word_indexer
        self.embedding_length = word_embeddings.get_embedding_length()

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Predict the sentiment label for a given list of words.
        :param ex_words: A list of words representing a sentence.
        :param has_typos: A flag indicating whether we're evaluating on typo data (currently unused).
        :return: 0 (negative) or 1 (positive).
        """

        word_indices = [self.word_indexer.index_of(word) for word in ex_words]
        unk_index = self.word_indexer.index_of("UNK")
        word_indices = [index if index != -1 else unk_index for index in word_indices]
        word_indices_tensor = torch.tensor([word_indices], dtype=torch.long)

        with torch.no_grad():
            log_probs = self.model(word_indices_tensor)

        predicted_class = torch.argmax(log_probs, dim=1).item()

        return predicted_class


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Define the network parameters
    vocab_size = len(word_embeddings.word_indexer)
    embedding_dim = word_embeddings.get_embedding_length()  # Embedding dimension
    hidden_dim = 24  # Hidden layer size
    output_dim = 2  # Two output classes: positive (1) and negative (0)

    # Initialize the neural network
    dan_model = DeepAveragingNetwork(vocab_size, embedding_dim, hidden_dim, output_dim,
                                     word_embeddings.vectors)

    # Loss function: Negative Log-Likelihood Loss since we use LogSoftmax in the model
    criterion = nn.NLLLoss()

    # Optimizer: Adam optimizer
    optimizer = optim.Adam(dan_model.parameters(), lr=args.lr)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        dan_model.train()  # Set model to training mode

        total_loss = 0
        for example in train_exs:
            # Convert words to indices
            word_indices = [word_embeddings.word_indexer.index_of(word) for word in example.words]

            unk_index = word_embeddings.word_indexer.index_of("UNK")
            word_indices = [index if index != -1 else unk_index for index in word_indices]

            word_indices_tensor = torch.tensor([word_indices], dtype=torch.long)

            # Target label as tensor
            target_label = torch.tensor([example.label], dtype=torch.long)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass: Compute log-probabilities
            log_probs = dan_model(word_indices_tensor)

            # Compute loss
            loss = criterion(log_probs, target_label)

            # Backpropagate the loss
            loss.backward()

            # Update model parameters
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_exs)}')

    # After training, return the trained NeuralSentimentClassifier
    return NeuralSentimentClassifier(dan_model, word_embeddings)

