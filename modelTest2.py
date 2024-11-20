import torch
from torch import device, cuda
import json
import pickle
from trainModel import trainLoop, randMidiGen, initializeMask
from modelClasses.Generator import Generator

torch.cuda.empty_cache()

EPOCHS = 100
LIMIT = 20
BATCH = 1
PADDING = -1
LOGGING = 10
GSTEP = 1
TRAINING = True
TEST_ON = EPOCHS - 1
SAVE_AS = f"{EPOCHS}E_{LIMIT}L"

midiFolder = "C:/Users/olive/Documents/NortheasternDocuments/2024Fall/Machine Learning and Data Mining 2/Gen AI Music/genAI_music/maestro-v3.0.0/2004"

if TRAINING:
    trainLoop(midiFolder, EPOCHS, limit=LIMIT, batch_size=BATCH, padding_value=PADDING, log_every=LOGGING, g_step=GSTEP)

model_dir = "C:/Users/olive/Documents/NortheasternDocuments/2024Fall/Machine Learning and Data Mining 2/Gen AI Music/genAI_music/models/"
with open(f"{model_dir}params.json", "r") as json_file:
    params = json.load(json_file)

cudaDevice = device('cuda' if cuda.is_available() else 'cpu')
model = Generator(params['input_size'],
                    params['hidden_size_linear'],
                    params['hidden_size_attention'],
                    params['output_size'],
                    params["padding_value"]).to(cudaDevice)


def printMidi(midiTens):
    print(midiTens.cpu().detach().numpy().round(decimals=2))

with torch.no_grad():
    genPath = f"{model_dir}epoch_{TEST_ON}/Generator.model"
    model.load_state_dict(torch.load(genPath, weights_only=True))
    model.eval()

    with open(f"{model_dir}controlMidi.p", "rb") as control:
        controlMidi = pickle.load(control)

    # dataset = MidiDataSet(midiFolder, limit=LIMIT, padding_value=PADDING)
    # controlMidi = dataset[0][0]

    printMidi(controlMidi)

    print("Evaluating Model")
    randMidi = randMidiGen(controlMidi, cudaDevice)
    mask = initializeMask(controlMidi, PADDING)
    model.setMask(mask)

    testGenMidi = model.forward(randMidi)
    testGenMidi = testGenMidi.to(torch.int)
    printMidi(testGenMidi)

    with open(f"outputs/{SAVE_AS}", "wb") as midiOutput:
        pickle.dump(testGenMidi, midiOutput)

torch.cuda.empty_cache()
