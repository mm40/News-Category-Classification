from typing import NamedTuple
from utilities.munging import apply_regex_list_to_text
import torch
from collections import Counter
import pickle


class Vocabulary():
    def __init__(self):
        """Vocabulary constructor
        """
        self._token_to_idx = {}
        self._idx_to_token = {}

    def insert_token(self, token):
        """Adds token to vocabulary (if it doesn't exist) and returns idx of it.
        Args:
           token (str) : token to be added
        Returns:
           idx (int) : idx of the token
        """
        token = token
        idx = self._token_to_idx.get(token, None)
        if idx is None:
            idx = len(self._token_to_idx)
            self._token_to_idx[token] = idx
            self._idx_to_token[idx] = token

        return idx

    def token_from_idx(self, idx):
        """Returns token from vocabulary, by it's idx. If it doesn't exist,
        throws an exception"""
        token = self._idx_to_token.get(idx, None)
        if token is None:
            raise KeyError("Tried to get token from idx which doesn't exist "
                           "in vocabulary (idx : {})".format(idx))
        else:
            return token

    def idx_from_token(self, token):
        """Returns idx of the token in vocabulary. If token doesn't exist,
        throws an exception

        Args:
           token (str) : token to lookup in vocabulary, and return it's idx
        Returns:
           idx (int) : idx of the token

        Raises:
           KeyError : if token is not found in vocabulary"""
        token = token
        idx = self._token_to_idx.get(token, None)
        if idx is None:
            raise KeyError("Tried to get idx from a token which doesn't exist "
                           "in vocabulary (token : {})".format(token))
        else:
            return idx

    def __len__(self):
        return len(self._idx_to_token)

    @classmethod
    def from_sentence_list(cls, sentences, regex_list, cnt_threshold):
        """Creates a vocabulary from list of text sentences. To each
            sentence in sentences, regex is applied, it's split by space
            into tokens, and than those tokens which occur more than specified
            threshold times, are added to vocabulary.

        Args:
            sentences (iterable object): each element is (str) sentence of text
            regex_list (list): list of pairs of regex rules to apply
            cnt_threshold (int): minimum number each token must appera, to be
                added to vocabulary

        Returns:
            (cls): vocabulary with words from sentences added.
        """
        vocab = cls()
        cntr = Counter()
        for line in sentences:
            line = apply_regex_list_to_text(regex_list, line.lower())
            for token in filter(None, line.split(" ")):
                cntr[token] += 1

        for token, cnt in cntr.items():
            if cnt >= cnt_threshold:
                vocab.insert_token(token)

        return vocab

    @classmethod
    def from_literal_token_list(cls, tokens):
        """Creates a vocabulary from list of items which will be added as tokens.

        Args:
            cls arg1
            tokens (list): list of (str) items, which will be added as tokens
                to the vocabulary AS THEY ARE WITH NO PROCESSING.


        Returns:
            (cls): vocabulary with tokens from the list added
        """
        vocab = cls()
        for token in set(tokens):  # convert to set to not iterate same items
            vocab.insert_token(token)
        return vocab

    def toFile(self, path):
        """Saves vocabulary to a file specified by file path (str)"""
        with open(path, "wb") as f:
            pickle.dump(self._idx_to_token, f)

    @classmethod
    def from_file(cls, path):
        """Instantiates a vocabulary from file saved using toFile function"""
        vocab = cls()
        with open(path, "rb") as f:
            vocab._idx_to_token = pickle.load(f)
        vocab._token_to_idx = {}
        for idx, token in vocab._idx_to_token.items():
            vocab._token_to_idx[token] = idx
        return vocab


class VocabularySeq(Vocabulary):
    """Vocabulary, but with sequence tokens added"""
    def __init__(self, unk_token='<UNK>', begin_seq_token='<seq_beg>',
                 end_seq_token='<seq_end>', mask_token='<mask>'):
        """Instantiates VocabularySeq

        Args:
           unk_token (str) : unknown token (default '<UNK>')
           begin_seq_token (str): token for seq. start (default '<seq_beg>')
           end_seq_token (str): token for sequence end (default '<seq_end>')
           mask_token (str): token for mask (default '<mask>')
        """
        super().__init__()

        class TokenInfo(NamedTuple):
            token: str
            idx: int

        self._specials = {"unknown":
                          TokenInfo(unk_token, self.insert_token(unk_token)),
                          "mask":
                          TokenInfo(mask_token, self.insert_token(mask_token)),
                          "beginSeq":
                          TokenInfo(begin_seq_token,
                                    self.insert_token(begin_seq_token)),
                          "endSeq":
                          TokenInfo(end_seq_token,
                                    self.insert_token(end_seq_token))}

        self._unkIdx = self._specials["unknown"].idx

    def get_specials(self):
        return self._specials  # No need to copy, NamedTuples are immutable

    def idx_from_token(self, token):
        return self._token_to_idx.get(token, self._unkIdx)


class SentenceTensorConverter():
    """Converts a list of tokens (str) into a tensor of their idx's (int)"""
    def __init__(self, vocabulary, fix_width_to=None):
        """Creates instance of converter which should use given vocabulary

        Args:
           vocabulary (nlp.VocabularySeq): vocabulary to use for conversion
           fix_width_to (int): if not None all conversions will have this width
              Note : this width INCLUDES begin and end markers (default None)
        """
        self._vocabulary = vocabulary
        self._fix_width_to = fix_width_to

    def tokens_to_idxs(self, token_list, device_str='cpu'):
        """Converts a list of tokens to a tensor of their idx's, with respect to
        instance's fix_width_to parameter passed to constructor.

        Args:
            token_list (list): list of tokens (str) to convert to their idx's

        Returns:
            1-d tensor of idx's (torch.Tensor)
        """
        list_width = len(token_list)
        width = list_width + 2 if self._fix_width_to is None \
            else self._fix_width_to

        if list_width + 2 > width:
            raise ValueError("Output tensor width is fixed to {}, but input "
                             "token list requires length of {}.".format(
                                 width, list_width + 2))
        v = self._vocabulary
        s = v.get_specials()

        result = torch.full((width,), fill_value=s["mask"].idx,
                            dtype=torch.int64, device=torch.device(device_str))
        result[0] = s["beginSeq"].idx
        for i, token in enumerate(token_list):
            result[i + 1] = v.idx_from_token(token)
        result[list_width + 1] = s["endSeq"].idx

        return result
