import mido

port = mido.open_output('Microsoft GS Wavetable Synth 0')

mid = mido.MidiFile('./maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi')
for msg in mid.play():
    port.send(msg)