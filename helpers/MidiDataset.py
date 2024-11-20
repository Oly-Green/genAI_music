import torch
from torch import device, cuda
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from helpers.MusicTensor import MusicTensor


class MidiDataSet(Dataset):
    def __init__(self, midiDir, limit=None, padding_value=-1):
        super(MidiDataSet, self).__init__()
        self.midiDir = midiDir
        self.cudaDevice = device('cuda' if cuda.is_available() else 'cpu')
        if limit:
            self.midiNames = os.listdir(midiDir)[:limit]
        else:
            self.midiNames = os.listdir(midiDir)
        self.padding_value = padding_value
        self.allMidis = self._getAllMidis()

    def _getAllMidis(self):
        allMidis = []
        for midiName in tqdm(self.midiNames):
            midiPath = os.path.join(self.midiDir, midiName)
            midiTensorConverter = MusicTensor(midiPath)
            midiTensor = midiTensorConverter.toTensors(padding_value=self.padding_value)
            allMidis.append(midiTensor.to(torch.float).to(self.cudaDevice))

        print("\nPadding Tensors\n")
        maxLength = max([len(midiTens) for midiTens in allMidis])
        for i in tqdm(range(len(allMidis))):
            if len(allMidis[i]) < maxLength:
                diff = maxLength - len(allMidis[i])
                padTens = torch.full_like(allMidis[i][0], self.padding_value)
                for _ in range(diff):
                    allMidis[i] = torch.cat((allMidis[i], padTens.unsqueeze(0)), dim=0)
        return allMidis

    def __len__(self):
        return len(self.midiNames)

    def __getitem__(self, index):
        return self.allMidis[index], index
