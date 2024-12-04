import torch
from torch import device, cuda
from torch.utils.data import DataLoader
import pickle
import json
from tqdm import tqdm
import os

from helpers.MidiSegmentedDataset import MidiDataSet
from helpers.SegmentedTrainer import Trainer


def trainLoop(midiDir, model_dir, epochs, batch_size=32, limit=None, padding_value=-1, log_every=10, d_step=1, input_chunk_size = 100, output_chunk_size = 100, layer_sizes = [8, 8, 8], g_lr=1e-4, d_lr=1e-4):
    print("Initializing DataSet")
    dataset = MidiDataSet(midiDir, limit=limit, padding_value=padding_value, input_chunk_size = input_chunk_size, output_chunk_size = output_chunk_size)
    print("Total Chunks: ", len(dataset))

    # TODO: Feed the unique values from the first byte into the generator as discrete options to pick from while allowing the second and third byte to get picked from the continuous range set by the max and min of the unique values from those respective bytes
    # print(dataset.getUniqueValues(0))

    uniques_values_0 = set()
    uniques_values_1 = set()
    uniques_values_2 = set()
    for i in range(dataset.getNumMidis()):
        uniques = dataset.getUniqueValues(i)
        uniques_values_0.update(uniques[0])
        uniques_values_1.update(uniques[1])
        uniques_values_2.update(uniques[2])
    uniques_values_0 = list(uniques_values_0)
    uniques_values_1 = list(uniques_values_1)
    uniques_values_2 = list(uniques_values_2)

    with open(f"{model_dir}controlMidiInput.p", "wb") as control_input:
        pickle.dump(dataset[0][0], control_input)
    with open(f"{model_dir}controlMidiOutput.p", "wb") as control_output:
        pickle.dump(dataset[1][0], control_output)

    print("Initializing DataLoader")
    # TODO: Maybe don't shuffle to avoid messing with timestamp track
    dataloader = DataLoader(dataset, batch_size=batch_size) #, shuffle=True)
    cudaDevice = device('cuda' if cuda.is_available() else 'cpu')
    dataIter = iter(dataloader)

    midiSize = dataset.getMidiByteLength()

    print("Initializing Model")
    params = {"input_size": midiSize,
             "G_hidden_size_linear": layer_sizes[0],
             "G_hidden_size_attention": layer_sizes[1],
             "D_hidden_size": layer_sizes[2],
             "output_size": midiSize,
             "batch_size": batch_size,
             "padding_value": padding_value,
             "g_learning_rate": g_lr,
             "d_learning_rate": d_lr,
             "uniques_values_0": uniques_values_0,
             "uniques_values_1": uniques_values_1,
             "uniques_values_2": uniques_values_2}

    trainer = Trainer(params["input_size"],
                      params["G_hidden_size_linear"],
                      params["G_hidden_size_attention"],
                      params["D_hidden_size"],
                      params["output_size"],
                      params["batch_size"],
                      params["uniques_values_0"],
                      params["uniques_values_1"],
                      params["uniques_values_2"],
                      params["padding_value"],
                      params["g_learning_rate"],
                      params["d_learning_rate"])

    with open(f"{model_dir}params.json", "w") as json_file:
        json.dump(params, json_file)

    print("Beginning Training")
    trainer.D.train()
    trainer.G.train()
    step = 0
    for epoch in range(epochs):
        for idx, (input_chunk, output_chunk, _) in enumerate(dataIter):

            input_chunk = input_chunk[0].to(cudaDevice)
            output_chunk = output_chunk[0].to(cudaDevice)

            if step % d_step == 0:
                d_loss, d_pred_real, d_pred_fake = trainer.trainDiscriminator(output_chunk, input_chunk)

            g_loss = trainer.trainGenerator(input_chunk)

            step += 1

        if epoch % log_every == 0 or epoch == epochs-1:
            print(f"Epoch: {epoch+1}/{epochs}\n DLoss: {d_loss}\n GLoss: {g_loss}")

            folder_name = f"epoch_{epoch}"
            epoch_folder = os.path.join(model_dir, folder_name)
            if not os.path.exists(epoch_folder):
                os.makedirs(epoch_folder)

            torch.save(trainer.G.state_dict(), f"{str(epoch_folder)}/Generator.model")
            torch.save(trainer.D.state_dict(), f"{str(epoch_folder)}/Discriminator.model")

        dataIter = iter(dataloader)