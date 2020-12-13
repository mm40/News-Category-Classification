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
        return self._selectedData.iloc[idx]

    def __len__(self):
        return len(self._selectedData)


class DatasetNewsVectorized(DatasetNews):
    def __init__(self, df, splitColHeader, vocabCat, vocabHeadl, regexList,
                 headerCategory, headerHeadline):
        """Instantiates DatasetNewsVectorized object, which is different to
            DatasetNews in that it returns vectorized result, instead of plain.

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

        class localMaxLengthKeeper():
            """Just an integer to be increased to max value"""
            def __init__(self):
                self.maxLength = 0
                # TODO: use lengthsCntr to remove too long headlines (outliers)
                # with localTextProcess, add colun with width, for easy remove?
                self.lengthsCntr = Counter()

            def updateLength(self, newLength):
                self.lengthsCntr[newLength] += 1
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
        pass

    def __getitem__(self, idx):
        """Vectorizes headline text, gets category id, returns dict of those"""
        row = super().__getitem__(idx)
        headlineList = row[self._headerHeadline].split(" ")
        category = row[self._headerCategory]
        return {"x": self._converter.tokensToIdxs(headlineList),
                "y": self._vocabCat.idx_from_token(category)}
