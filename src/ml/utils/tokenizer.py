import json
import logging
import os

import torch
from torchtext.data import get_tokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.tokenizer = get_tokenizer('basic_english')
        self.special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

    def fit_on_texts_and_embeddings(self, sentences: list[str], embeddings):
        # Add special tokens to the start of the vocab
        for special_token in self.special_tokens:
            self.vocab[special_token] = len(self.vocab)

        logger.info("Creating vocab from sentences and embeddings")
        # Add each unique word in the sentences to the vocab if there is also an embedding for it
        for sentence in tqdm(sentences):
            for word in self.tokenizer(sentence):
                if word not in self.vocab and word in embeddings.stoi:
                    self.vocab[word] = len(self.vocab)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # create an embbeding matrix from a set of pretrained embeddings based on the vocab
    def get_embeddings_matrix(self, embeddings):
        # Create a matrix of zeroes of the shape of the vocab size
        embeddings_matrix = torch.zeros((len(self.vocab), embeddings.dim))

        logger.info("Creating embeddings matrix from vocab")
        # For each word in the vocab get its index and add its embedding to the matrix
        for word, idx in self.vocab.items():
            if word in self.special_tokens:
                continue
            if word in embeddings.stoi:
                embeddings_matrix[idx] = embeddings[word]
            else:
                raise KeyError(f"Word {word} not in embeddings. Please create tokenizer based on embeddings")

        # Initialize the <pad> token with the mean of the embeddings of the vocab
        embeddings_matrix[1] = torch.mean(embeddings_matrix[len(self.special_tokens) :], dim=0)

        # Initialize the <sos> and <eos> tokens with the mean of the embeddings of the vocab
        # plus or minus a small amount of noise to avoid them matching the <unk> token
        # and avoiding having identical embeddings which the model can not distinguish
        noise = torch.normal(mean=0, std=0.1, size=(embeddings.dim,))
        embeddings_matrix[2] = torch.mean(embeddings_matrix[len(self.special_tokens) :] + noise, dim=0)
        embeddings_matrix[3] = torch.mean(embeddings_matrix[len(self.special_tokens) :] - noise, dim=0)

        return embeddings_matrix

    # add start of sentence and end of sentence tokens to the tokenizer sentence
    def add_special_tokens(self, tokens):
        return ["<sos>"] + tokens + ["<eos>"]

    # convert a sequence of words to a sequence of indices based on the vocab
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

    # convert a sequence of indices to a sequence words
    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab[idx] for idx in ids]

    def pad_sequences(self, sequences, max_length=None):
        # Pads the vectorized sequences

        # If max_length is not specified, pad to the length of the longest sequence
        if not max_length:
            max_length = max(len(seq) for seq in sequences)

        # Create a tensor for the lengths of the sequences
        sequence_lengths = torch.LongTensor([min(len(seq), max_length) for seq in sequences])

        # Create a tensor for the sequences with zeros
        seq_tensor = torch.zeros((len(sequences), max_length)).long()

        # Create a tensor for the masks with zeros
        seq_mask = torch.zeros((len(sequences), max_length)).long()

        # For each sequence add the values to the seq_tensor
        #  and add 1s to the seq_mask according to its length
        for idx, (seq, seq_len) in enumerate(zip(sequences, sequence_lengths)):
            # truncate the sequence if it exceeds the max length
            seq = seq[:seq_len]

            seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
            seq_mask[idx, :seq_len] = torch.LongTensor([1])

        return seq_tensor, seq_mask, sequence_lengths

    # split the text into tokens
    def tokenize(self, text):
        return self.tokenizer(text)

    def encode(self, texts, max_length=None):
        if isinstance(texts, str):
            texts = [texts]

        sequences = []
        for text in texts:
            tokens = self.tokenize(text)
            tokens = self.add_special_tokens(tokens)
            ids = self.convert_tokens_to_ids(tokens)
            sequences.append(ids)

        seq_tensor, seq_mask, sequence_lengths = self.pad_sequences(sequences, max_length)

        return seq_tensor, seq_mask, sequence_lengths

    # save the tokenizer to a json file
    def save(self, file_path: str, filename: str = "tokenizer.json"):
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        json_data = {}
        with open(os.path.join(file_path, filename), 'w') as tokenizer_file:
            json_data["vocab"] = self.vocab
            json_data["inv_vocab"] = self.inv_vocab
            json_data["special_tokens"] = self.special_tokens
            json.dump(json_data, tokenizer_file)
            logger.info(f"Successfully saved tokenizer {os.path.join(file_path, filename)}")

    # load the tokenizer from a json file
    def load(self, file_path: str, filename: str = "tokenizer.json"):
        if os.path.exists(file_path):
            with open(os.path.join(file_path, filename)) as tokenizer_file:
                json_data = json.load(tokenizer_file)
                self.vocab = json_data["vocab"]
                self.inv_vocab = json_data["inv_vocab"]
                self.special_tokens = json_data["special_tokens"]
                logger.info(f"Successfully loaded tokenizer from {os.path.join(file_path, filename)}")
        else:
            raise FileNotFoundError("The file path does not exist")

    def __call__(self, texts, max_length=None):
        return self.encode(texts, max_length)
