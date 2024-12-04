import mido
import torch

class MusicTensorEncoder():
    def __init__(self, midiPath):
        self.midiPath = midiPath
        self.mid = mido.MidiFile(self.midiPath)

        self.allTrack_tensors = torch.Tensor()
        self.tensorSizes = []

        self.head_len = 3

    def _toBytes(self):
        track0_bytes = []
        for msg in self.mid.tracks[0]:
            byts = msg.copy().bytes() + [msg.time]
            track0_bytes.append(byts)

        track1_bytes = []
        track1_timestamps = []
        for msg in self.mid.tracks[1]:
            if msg.dict()["type"] != "track_name":
                byts = msg.copy().bytes()
                track1_bytes.append(byts)
                track1_timestamps.append(msg.time)

        return track0_bytes, track1_bytes, track1_timestamps

    def toTensors(self, padding_value = -1):
        track0_bytes, track1_bytes, track1_timestamps = self._toBytes()

        # append track1 head and tail to track0 bc they are bytes containing metadata more in line with the data contained in track 0
        track0_bytes = track0_bytes + [track1_bytes[0]] + [track0_bytes[-1]]
        track1_bytes = track1_bytes[1:-1]

        track0_tensors = [torch.tensor(byte) for byte in track0_bytes]
        # track0_tensors = torch.stack(track0_tensors, dim=0)
        track0_tensors = torch.nn.utils.rnn.pad_sequence(track0_tensors, batch_first=True, padding_value=padding_value)

        # TODO: remove padding value when decoding
        # track1_bytes[0] += [padding_value]
        track1_tensors = [torch.tensor(byte) for byte in track1_bytes]
        track1_tensors = torch.stack(track1_tensors, dim=0)

        track1_timestamps_tensor = torch.tensor(track1_timestamps)

        return track0_tensors, track1_tensors, track1_timestamps_tensor

    def getDict(self):
        msgs = []
        for msg in self.mid.tracks[1]:
            msgs.append(msg.dict())
        return msgs