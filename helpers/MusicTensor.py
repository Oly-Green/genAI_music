import mido
import torch

class MusicTensor():
    def __init__(self, midiPath, playback= False, port="Microsoft GS Wavetable Synth 0"):
        self.midiPath = midiPath
        self.mid = mido.MidiFile(self.midiPath)

        if playback:
            self.port = mido.open_output(port)

        self.allTrack_tensors = torch.Tensor()
        self.tensorSizes = []

    def testPlay(self):
        for msg in self.mid.play():
            self.port.send(msg)

    def closePort(self):
        self.port.close()

    def openPort(self):
        """
        port is open by default so only use if you've already called closePort()
        """
        self.port.open(self.midiPath)

    def _toBytes(self):
        track0_bytes = []
        for msg in self.mid.tracks[0]:
            byts = msg.copy().bytes() + [msg.time]
            track0_bytes.append(byts)

        track1_bytes = []
        for msg in self.mid.tracks[1]:
            if msg.dict()["type"] != "track_name":
                byts = msg.copy().bytes() + [msg.time]
                track1_bytes.append(byts)

        return track0_bytes, track1_bytes

    def toTensors(self, padding_value = -1):
        track0_bytes, track1_bytes = self._toBytes()
        track0_tensors = [torch.tensor(byte) for byte in track0_bytes]
        track1_tensors = [torch.tensor(byte) for byte in track1_bytes]

        self.allTrack_tensors = track0_tensors + track1_tensors

        self.tensorSizes = [tensor.size() for tensor in self.allTrack_tensors]

        self.allTrack_tensors = torch.nn.utils.rnn.pad_sequence(self.allTrack_tensors, batch_first=True, padding_value=padding_value)

        return self.allTrack_tensors

    def _unpadTensors(self):
        track0_unpadded_tensors = [tensor[0:self.tensorSizes[i][0]] for i, tensor in list(enumerate(self.allTrack_tensors))[:3]]
        track1_unpadded_tensors = [tensor[0:self.tensorSizes[i][0]] for i, tensor in list(enumerate(self.allTrack_tensors))[3:]]

        track0_unpadded_bytes = [track0_unpadded_tensors[i].tolist() for i in range(len(track0_unpadded_tensors))]
        track1_unpadded_bytes = [track1_unpadded_tensors[i].tolist() for i in range(len(track1_unpadded_tensors))]

        return track0_unpadded_bytes, track1_unpadded_bytes

    def toMidi(self):
        track0_unpadded_bytes, track1_unpadded_bytes = self._unpadTensors()

        track0_msgs = []
        for msg in track0_unpadded_bytes:
            new_msg = mido.MetaMessage.from_bytes(msg[:-1])
            new_msg.time = msg[-1]
            track0_msgs.append(new_msg)

        track1_msgs = []
        for msg in track1_unpadded_bytes[:-1]:
            new_msg = mido.Message.from_bytes(msg[:-1])
            new_msg.time = msg[-1]
            track1_msgs.append(new_msg)

        track_end_msg = mido.MetaMessage.from_bytes(track1_unpadded_bytes[-1][:-1])
        track_end_msg.time = track1_unpadded_bytes[-1][-1]
        track1_msgs.append(track_end_msg)

        reconstructed_mid = mido.MidiFile()
        track0 = mido.MidiTrack()
        track1 = mido.MidiTrack()

        for msg in track0_msgs:
            track0.append(msg)
        for msg in track1_msgs:
            track1.append(msg)

        reconstructed_mid.tracks.append(track0)
        reconstructed_mid.tracks.append(track1)

        return reconstructed_mid

    def play(self, mid):
        for msg in mid.play():
            self.port.send(msg)
