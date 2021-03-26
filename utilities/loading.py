from torch.utils.data.dataset import Dataset
from utilities.munging import apply_regex_list_to_text
from utilities.nlp import SentenceTensorConverter
from pandas import unique
from torch.utils.data.dataloader import DataLoader


class DatasetNews(Dataset):
    def __init__(self, df, split_col_header):
        """Instantiates DataNews Object, which has a dictionary _data_splits
            of dataframes split by a unique label values, for later selection.
            NOTE : data is NOT selected by default. After instantiation,
                   selecting data with select_data function is required

        Args:
            df (pandas.DataFrame): dataframe with additional column, which
                contains label for each row of data. For instance, some rows
                have label 'train', some 'test', some 'eval', etc. For
                each unique value in this column, all rows with this value are
                selected into a single dataframe, for later selection by label
            split_col_header (str): name of the column, which contains labels
        """
        self._data_splits = {}
        for label in unique(df[split_col_header]):
            self._data_splits[label] = df[df[split_col_header] == label]\
                .drop(split_col_header, axis='columns').reset_index(drop=True)
        self._selected_data = None

    def select_data(self, label):
        if label in self._data_splits:
            self._selected_data = self._data_splits[label]
        else:
            raise ValueError('Data split label "{}" was tried to be selected, '
                             'however only these splits were created: {}'.
                             format(label, list(self._data_splits.keys())))

    def __getitem__(self, idx):
        if self._selected_data is None:
            # TODO : same exact error message in __getitem__ and __len__.
            # for a better style, merge them into a class, or unique varuable.
            raise LookupError('Data Split not selected. Available splits are:'
                              ' {}. Use function select_data to set one of'
                              ' them.'.format(list(self._data_splits.keys())))
        return self._selected_data.iloc[idx]

    def __len__(self):
        if self._selected_data is None:
            raise LookupError('Data Split not selected. Available splits are:'
                              ' {}. Use function select_data to set one of'
                              ' them.'.format(list(self._data_splits.keys())))
        return len(self._selected_data)


class DatasetNewsVectorized(DatasetNews):
    def __init__(self, df, split_col_header, vocab_cat, vocab_headl,
                 regex_list, header_category, header_headline,
                 device_str='cpu'):
        """Instantiates DatasetNewsVectorized object, which is different to
            DatasetNews in that it returns vectorized result, instead of plain.
            NOTE : df is modified in-place!

        Args:
            df (pandas.DataFrame): see help for DatasetNews constructor
            split_col_header (str): -||-
            vocab_cat (Vocabulary): vocabulary for translating category to idx
            vocab_headl (VocabularySeq): vocab for sequentializing headline
            header_category (str): header of category column of dataframe
            header_headline (str): header of headline column of dataframe
        """
        self._vocab_cat = vocab_cat
        self._vocab_headl = vocab_headl
        self._header_category = header_category
        self._header_headline = header_headline
        self._device_str = device_str

        class LocalMaxLengthKeeper():
            """Just an integer to be increased to max value"""
            def __init__(self):
                self.max_length = 0

            def update_length(self, new_length):
                if new_length > self.max_length:
                    self.max_length = new_length

        def local_text_process(text, keeper):
            """Besides modidying text, this function will set longest sequence.
            This hack is done to not iterate again for determining it later."""
            text = apply_regex_list_to_text(regex_list, text.lower())
            keeper.update_length(len(list(filter(None, text.split(" ")))))
            return text

        keeper = LocalMaxLengthKeeper()
        # Note : if sequence is not limited, and calculating max length is not
        # required, apply logic should be moved to __getitem__ from here
        df[header_headline] = df[header_headline].apply(local_text_process,
                                                        keeper=keeper)
        self._max_seq_length = keeper.max_length + 2  # + 2 for beg/end seq sym

        self._converter = SentenceTensorConverter(self._vocab_headl,
                                                  self._max_seq_length)

        super().__init__(df, split_col_header)  # called after df.apply

    def __getitem__(self, idx):
        """Vectorizes headline text, gets category id, returns dict of those"""
        row = super().__getitem__(idx)
        headline_list = list(filter(None,
                                    row[self._header_headline].split(" ")))
        category = row[self._header_category]
        return {"x": self._converter.tokens_to_idxs(headline_list,
                                                    self._device_str),
                "y": self._vocab_cat.idx_from_token(category)}

    def get_max_seq_length(self):
        return self._max_seq_length


class DataLoaderManager():
    """DataLoader wrapper with capabilities to select label of input dataset,
    using function select_data(label) of the dataset"""
    def __init__(self, dataset, *args, **kwargs):
        """
        Args:
            dataset (DataSetSubclass): some subclass of DataSet which has
                select_data function to select the split.
            args: to be passed to DataLoader constructor
            kwargs: -||-
        """
        self._dataset = dataset
        self._data_loader_args = args
        self._data_loader_kwargs = kwargs

    def create_data_loader(self, label):
        self._dataset.select_data(label)
        dl = DataLoader(self._dataset, *self._data_loader_args,
                        **self._data_loader_kwargs)
        return dl
