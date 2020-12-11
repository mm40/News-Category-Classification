class Vocabulary():
    def __init__(self):
        """Vocabulary constructor
        """
        self._token_to_id = {}
        self._id_to_token = {}

    def insert_token(self, token):
        """Adds token to vocabulary (if it doesn't exist) and returns id of it.
        Args:
           token (str) : token to be added (lowered)
        Returns:
           id (int) : id of the token
        """
        token = token.lower()
        id = self._token_to_id.get(token, None)
        if id is None:
            id = len(self._token_to_id)
            self._token_to_id[token] = id
            self._id_to_token[id] = token

        return id

    def token_from_id(self, id):
        """Returns token from vocabulary, by it's id. If it doesn't exist,
        throws an exception"""
        token = self._id_to_token.get(id, None)
        if token is None:
            raise KeyError("Tried to get token from id which doesn't exist "
                           "in vocabulary (id : {})".format(id))
        else:
            return token

    def id_from_token(self, token):
        """Returns id of the token in vocabulary. If token doesn't exist,
        throws an exception

        Args:
           token (str) : token to lookup in vocabulary, and return it's id
        Returns:
           id (int) : id of the token

        Raises:
           KeyError : if token is not found in vocabulary"""
        token = token.lower()
        id = self._token_to_id.get(token, None)
        if id is None:
            raise KeyError("Tried to get id from a token which doesn't exist "
                           "in vocabulary (token : {})".format(token))
        else:
            return id

    def __len__(self):
        return len(self._id_to_token)


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

        self._unkToken = unkToken
        self._beginSeqToken = beginSeqToken
        self._endSeqToken = endSeqToken
        self._maskToken = maskToken

        self.insert_token(unkToken)
        self.insert_token(maskToken)
        self.insert_token(beginSeqToken)
        self.insert_token(endSeqToken)

class SentenceTensorConverter():
    """Converts a list of tokens (str) into a tensor of their id's (int)"""
    def __init__(self, vocabulary, fixWidthTo=None):
        """Creates instance of converter which should use given vocabulary

        Args:
           vocabulary (nlp.VocabularySeq): vocabulary to use for conversion
           fixWidthTo (int): if not None, all conversions will have this width.
              Note : this width INCLUDES begin and end markers (default None)
        """
        self._vocabulary = vocabulary
        self._fixWidthTo = fixWidthTo

    def tokensToIds(self, tokenList):
        """Converts a list of tokens to a tensor of their id's, with respect to
        instance's fixWidthTo parameter passed to constructor.

        Args:
            tokenList (list): list of tokens (str) to convert to their id's

        Returns:
            1d tensor of id's (torch.Tensor)
        """
        listWidth = len(tokenList) + 2
        width = listWidth if self._fixWidthTo is None else self._fixWidthTo

        if listWidth > width:
            raise ValueError("Output tensor width is fixed to {}, but input "
                             "token list requires length of {}.".format(
                                 width, listWidth))
