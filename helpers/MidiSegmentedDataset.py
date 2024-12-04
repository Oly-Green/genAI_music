import torch
from torch import device, cuda
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from helpers.SegmentedMusicTensor import MusicTensorEncoder


class MidiDataSet(Dataset):
    def __init__(self, midiDir, limit=None, padding_value=-1, input_chunk_size = 30, output_chunk_size = 60):
        super(MidiDataSet, self).__init__()
        self.midiDir = midiDir
        self.cudaDevice = device('cuda' if cuda.is_available() else 'cpu')

        if limit:
            self.midiNames = os.listdir(midiDir)[:limit]
        else:
            self.midiNames = os.listdir(midiDir)

        self.input_chunk_size = input_chunk_size
        self.output_chunk_size = output_chunk_size
        self.total_chunks = 0
        self.input_chunks = []
        self.output_chunks = []
        self.chunk_heads = []

        self.head_len = 3

        self.padding_value = padding_value
        self.allMidis = self._getAllMidis()

        self._segmentMidis()

    def _getAllMidis(self):
        allMidis = []
        for midiName in tqdm(self.midiNames):
            midiPath = os.path.join(self.midiDir, midiName)
            midiTensorConverter = MusicTensorEncoder(midiPath)
            track0_tensors, track1_tensors, track1_timestamps_tensor = midiTensorConverter.toTensors(padding_value=self.padding_value)
            allMidis.append([track0_tensors.to(torch.float).to(self.cudaDevice),
                             track1_tensors.to(torch.float).to(self.cudaDevice),
                             track1_timestamps_tensor.to(torch.float).to(self.cudaDevice)])

        return allMidis

    def _segmentMidis(self):
        total_chunk_size = self.input_chunk_size + self.output_chunk_size

        for track0_tensors, track1_tensors, track1_timestamps_tensor in self.allMidis:
            # track1_head = track0_tensors[0]
            # track1_tail = track1_tensors[-1]
            # track1_tensors = track1_tensors[1:-1]

            num_midi_chunks = int(len(track1_tensors) / total_chunk_size)
            for i in range(num_midi_chunks):
                input_chunk_start = i * self.input_chunk_size
                input_chunk_end = input_chunk_start + self.input_chunk_size
                ouput_chunk_end = input_chunk_end + self.output_chunk_size
                input_chunk =  tuple([track1_tensors[input_chunk_start:input_chunk_end], track1_timestamps_tensor[input_chunk_start:input_chunk_end]])
                ouptut_chunk = tuple([track1_tensors[input_chunk_end:ouput_chunk_end], track1_timestamps_tensor[input_chunk_end:ouput_chunk_end]])

                # self.chunk_heads.append([track0_tensors, track1_head, track1_tail])
                self.chunk_heads.append(track0_tensors)
                self.input_chunks.append(input_chunk)
                self.output_chunks.append(ouptut_chunk)

                self.total_chunks += 1

    def getMidiByteLength(self):
        return len(self.allMidis[0][1][0])

    def getChunkHead(self, index):
        return self.chunk_heads[index]

    def getUniqueValues(self, index):
        unique_values = {}
        for i in range(self.allMidis[index][1].shape[1]):
            unique_values[i] = torch.unique(self.allMidis[index][1][:,i]).cpu().numpy().astype(int).tolist()
        return unique_values

    def getNumMidis(self):
        return len(self.allMidis)

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, index):
        return self.input_chunks[index], self.output_chunks[index], index
