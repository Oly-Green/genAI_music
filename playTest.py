from helpers.SegmentedMusicTensor import MusicTensorEncoder
import numpy as np

MT = MusicTensorEncoder("./maestro-v3.0.0/2013/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_03_R1_2013_wav--2.midi")






# for msg in MT.getDict():
#     if msg["type"] != "control_change" and msg["type"] != "note_on":
#         print(msg["type"])
#
midTensors = MT.toTensors()
# for tens in midTensors:
#     print(tens)

#
max_values = midTensors[1].max(axis=0)[0]
min_values = midTensors[1].min(axis=0)[0]

print(max_values)
print(min_values)

arr = midTensors[1].cpu().numpy()
column_index = 0
matching_indices = np.where(arr[:, column_index] >= 255)[0]
print(len(matching_indices))

for idx in matching_indices:
    print(f"Row {idx}: {arr[idx]}")

print(len(arr))
#
# note_ons = np.sum(arr[:, 0] == 144)
# print(note_ons)
#
# filtered_arr = arr[np.all(arr <= 255, axis=1)]
# print(len(arr), len(filtered_arr))




# mid = MT.toMidi()
# MT.play(mid)
# MT.closePort()