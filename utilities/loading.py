from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from utilities.munging import apply_regex_list_to_text
from utilities.nlp import SentenceTensorConverter
from pandas import unique
from collections import Counter


class DatasetNews(Dataset):
    def __init__(self, df, splitColHeader):
        """Instantiates DataNews Object, which has a dictionary _dataSplits
            of dataframes split by a unique label values, for later selection.
            NOTE : data is NOT selected by default. After instantiation,
                   selecting data with selectData function is required

        Args:
            df (pandas.DataFrame): dataframe with additional column, which
                contains label for each row of data. For instance, some rows
                have label 'train', some 'test', some 'eval', etc. For
                each unique value in this column, all rows with this value are
                selected into a single dataframe, for later selection by label
            splitColHeader (str): name of the column, which contains labels
        """
        self._dataSplits = {}
        for label in unique(df[splitColHeader]):
            self._dataSplits[label] = df[df[splitColHeader] == label]\
                .drop(splitColHeader, axis='columns').reset_index(drop=True)
        self._selectedData = None

    def selectData(self, label):
        if label in self._dataSplits:
            self._selectedData = self._dataSplits[label]
        else:
            raise ValueError('Data split label "{}" was tried to be selected, '
                             'however only the following splits were created: '
                             '{}'.format(label, list(self._dataSplits.keys())))

    def __getitem__(self, idx):
        if self._selectedData is None:
            # TODO : same exact error message in __getitem__ and __len__.
            # for a better style, merge them into a class, or unique varuable.
            raise LookupError('Data Split not selected. Available splits are:'
                              ' {}. Use function selectData to set one of'
                              ' them.'.format(list(self._dataSplits.keys())))
        return self._selectedData.iloc[idx]

    def __len__(self):
        if self._selectedData is None:
            raise LookupError('Data Split not selected. Available splits are:'
                              ' {}. Use function selectData to set one of'
                              ' them.'.format(list(self._dataSplits.keys())))
        return len(self._selectedData)


class DatasetNewsVectorized(DatasetNews):
    def __init__(self, df, splitColHeader, vocabCat, vocabHeadl, regexList,
                 headerCategory, headerHeadline, deviceStr='cpu'):
        """Instantiates DatasetNewsVectorized object, which is different to
            DatasetNews in that it returns vectorized result, instead of plain.
            NOTE : df is modified in-place!

        Args:
            df (pandas.DataFrame): see help for DatasetNews constructor
            splitColHeader (str): -||-
            vocabCat (Vocabulary): vocabulary for translating category to idx
            vocabHeadl (VocabularySeq): vocabulary for sequentializing headline
            headerCategory (str): header of category column of dataframe
            headerHeadline (str): header of headline column of dataframe
        """
        self._vocabCat = vocabCat
        self._vocabHeadl = vocabHeadl
        self._headerCategory = headerCategory
        self._headerHeadline = headerHeadline
        self._deviceStr = deviceStr

        class localMaxLengthKeeper():
            """Just an integer to be increased to max value"""
            def __init__(self):
                self.maxLength = 0

            def updateLength(self, newLength):
                if newLength > self.maxLength:
                    self.maxLength = newLength

        def localTextProcess(text, keeper):
            """Besides modidying text, this function will set longest sequence.
            This hack is done to not iterate again for determining it later."""
            text = apply_regex_list_to_text(regexList, text.lower())
            keeper.updateLength(len(list(filter(None, text.split(" ")))))
            return text

        keeper = localMaxLengthKeeper()
        df[headerHeadline] = df[headerHeadline].apply(localTextProcess,
                                                      keeper=keeper)
        self._maxSeqLength = keeper.maxLength

        self._converter = SentenceTensorConverter(self._vocabHeadl,
                                                  self._maxSeqLength)

        super().__init__(df, splitColHeader)  # called after df.apply

    def __getitem__(self, idx):
        """Vectorizes headline text, gets category id, returns dict of those"""
        row = super().__getitem__(idx)
        headlineList = row[self._headerHeadline].split(" ")
        category = row[self._headerCategory]
        return {"x": self._converter.tokensToIdxs(headlineList,
                                                  self._deviceStr),
                "y": self._vocabCat.idx_from_token(category)}
