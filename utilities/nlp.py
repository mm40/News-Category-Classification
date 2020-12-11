class Vocabulary():
    def __init__(self, unk=None):
        """Vocabulary constructor
        Args:
           unk (str) : String for <unknown> token to be added (if not None)
        """
        self._token_to_id = {}
        self._id_to_token = {}
        if unk is not None:
            self.insert_token(unk)

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
    pass
