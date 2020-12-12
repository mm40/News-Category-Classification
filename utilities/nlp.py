from typing import NamedTuple
import torch


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


class VocabularySeq(Vocabulary):
    """Vocabulary, but with sequence tokens added"""
    def __init__(self, unkToken='<UNK>', beginSeqToken='<seq_beg>',
                 endSeqToken='<seq_end>', maskToken='<mask>'):
        """Instantiates VocabularySeq

        Args:
           unkToken (str) : unknown token (default '<UNK>')
           beginSeqToken (str): token for sequence start (default '<seq_beg>')
           endSeqToken (str): token for sequence end (default '<seq_end>')
           maskToken (str): token for mask (default '<mask>')
        """
        super().__init__()

        class TokenInfo(NamedTuple):
            token: str
            idx: int

        self._specials = {"unknown":
                          TokenInfo(unkToken, self.insert_token(unkToken)),
                          "mask":
                          TokenInfo(maskToken, self.insert_token(maskToken)),
                          "beginSeq":
                          TokenInfo(beginSeqToken,
                                    self.insert_token(beginSeqToken)),
                          "endSeq":
                          TokenInfo(endSeqToken,
                                    self.insert_token(endSeqToken))}

        self._unkIdx = self._specials["unknown"].idx

    def getSpecials(self):
        return self._specials  # No need to copy, NamedTuples are immutable

    def idx_from_token(self, token):
        return self._token_to_idx.get(token, self._unkIdx)


class SentenceTensorConverter():
    """Converts a list of tokens (str) into a tensor of their idx's (int)"""
    def __init__(self, vocabulary, fixWidthTo=None):
        """Creates instance of converter which should use given vocabulary

        Args:
           vocabulary (nlp.VocabularySeq): vocabulary to use for conversion
           fixWidthTo (int): if not None, all conversions will have this width.
              Note : this width INCLUDES begin and end markers (default None)
        """
        self._vocabulary = vocabulary
        self._fixWidthTo = fixWidthTo

    def tokensToIdxs(self, tokenList):
        """Converts a list of tokens to a tensor of their idx's, with respect to
        instance's fixWidthTo parameter passed to constructor.

        Args:
            tokenList (list): list of tokens (str) to convert to their idx's

        Returns:
            1-d tensor of idx's (torch.Tensor)
        """
        listWidth = len(tokenList)
        width = listWidth + 2 if self._fixWidthTo is None else self._fixWidthTo

        if listWidth + 2 > width:
            raise ValueError("Output tensor width is fixed to {}, but input "
                             "token list requires length of {}.".format(
                                 width, listWidth))
        v = self._vocabulary
        s = v.getSpecials()

        result = torch.full((width,), fill_value=s["mask"].idx,
                            dtype=torch.int64)
        result[0] = s["beginSeq"].idx
        for i, token in enumerate(tokenList):
            result[i + 1] = v.idx_from_token(token)
        result[listWidth + 1] = s["endSeq"].idx

        return result
