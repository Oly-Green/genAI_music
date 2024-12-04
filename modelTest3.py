import torch
from torch import device, cuda
import json
import pickle
from trainSegmentedModel import trainLoop
from modelClasses.Generator import Generator

torch.cuda.empty_cache()

EPOCHS = 100
LIMIT = 25
BATCH = 64
PADDING = -1
LOGGING = 10
DSTEP = 10
# G_hidden_size_linear, G_hidden_size_attention, D_hidden_size
LAYERS = [32, 32, 8]
GLR = 0.001
DLR = 0.01
TRAINING = True
TEST_ON = EPOCHS - 1

model_dir = "C:/Users/olive/Documents/NortheasternDocuments/2024Fall/Machine Learning and Data Mining 2/Gen AI Music/genAI_music/segmentedModels/"
SAVE_AS = f"segmented_{EPOCHS}E_{LIMIT}L"

midiFolder = "C:/Users/olive/Documents/NortheasternDocuments/2024Fall/Machine Learning and Data Mining 2/Gen AI Music/genAI_music/maestro-v3.0.0/2004"

if TRAINING:
    trainLoop(midiFolder, model_dir, EPOCHS, limit=LIMIT, batch_size=BATCH, padding_value=PADDING, log_every=LOGGING, d_step=DSTEP, layer_sizes=LAYERS, g_lr=GLR, d_lr=DLR)

with open(f"{model_dir}params.json", "r") as json_file:
    params = json.load(json_file)

cudaDevice = device('cuda' if cuda.is_available() else 'cpu')
model = Generator(params['input_size'],
                    params['G_hidden_size_linear'],
                    params['G_hidden_size_attention'],
                    params['output_size'],
                    params["padding_value"],
                    params["uniques_values_0"],
                    params["uniques_values_1"],
                    params["uniques_values_2"]).to(cudaDevice)


def printMidi(midiTens):
    print(midiTens[0].cpu().detach().numpy().round(decimals=2))

with torch.no_grad():
    genPath = f"{model_dir}epoch_{TEST_ON}/Generator.model"
    model.load_state_dict(torch.load(genPath, weights_only=True))
    model.eval()

    with open(f"{model_dir}controlMidiInput.p", "rb") as control_input:
        controlMidiInput = pickle.load(control_input)
    with open(f"{model_dir}controlMidiOutput.p", "rb") as control_output:
        controlMidiOutput = pickle.load(control_output)

    print("Control Output: ")
    printMidi(controlMidiOutput)

    print("Evaluating Model")
    testGenMidi = model.forward(controlMidiInput[0])
    testGenMidi = model.quantize_to_unique_vals(testGenMidi)
    testGenMidi = testGenMidi.to(torch.int)
    print(testGenMidi.cpu().detach().numpy())

    with open(f"outputs/{SAVE_AS}", "wb") as midiOutput:
        pickle.dump(testGenMidi, midiOutput)

torch.cuda.empty_cache()
