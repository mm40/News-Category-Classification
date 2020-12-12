from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from pandas import unique


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
    def __init__(self, df, splitColHeader, vocabCat, vocabHeadl):
        """Instantiates DatasetNewsVectorized object, which is different to
            DatasetNews in that it returns vectorized result, instead of plain.

        Args:
            df (pandas.DataFrame): see help for DatasetNews constructor
            splitColHeader (str): -||-
            vocabCat (Vocabulary): vocabulary for translating category to idx
            vocabHeadl (VocabularySeq): vocabulary for sequentializing headline
        """
        super().__init__(df, splitColHeader)
        self._vocabCat = vocabCat
        self._vocabHeadl = vocabHeadl
        # TODO : determine maximum sequence width
