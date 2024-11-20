import torch
from torch import device, cuda
from torch.utils.data import DataLoader
import pickle
import json
from tqdm import tqdm
import os
from helpers.MidiDataset import MidiDataSet
from helpers.Trainer import Trainer

def initializeMask(x, padding_value):
    return torch.where(x == padding_value, padding_value, 1.0)
    # return int(x != padding_value)

def randMidiGen(like, cudaDevice):
    randMidi = torch.rand_like(like) * 255
    randMidi = randMidi.to(cudaDevice)
    return randMidi

def trainLoop(midiDir, epochs, batch_size=32, limit=None, padding_value=-1, log_every=10, g_step=1):
    model_dir = "C:/Users/olive/Documents/NortheasternDocuments/2024Fall/Machine Learning and Data Mining 2/Gen AI Music/genAI_music/models/"

    print("Initializing DataSet")
    dataset = MidiDataSet(midiDir, limit=limit, padding_value=padding_value)

    with open(f"{model_dir}controlMidi.p", "wb") as control:
        pickle.dump(dataset[0][0], control)

    print("Initializing DataLoader")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cudaDevice = device('cuda' if cuda.is_available() else 'cpu')
    dataIter = iter(dataloader)
    midiSize = list(dataset[0][0].size())[1]

    print("Initializing Model")
    params = {"input_size": midiSize,
             "hidden_size_linear": int(midiSize / 2),
             "hidden_size_attention": int(midiSize / 2),
             "output_size": midiSize,
             "batch_size": batch_size,
             "padding_value": padding_value}

    trainer = Trainer(params["input_size"],
                      params["hidden_size_linear"],
                      params["hidden_size_attention"],
                      params["output_size"],
                      params["batch_size"],
                      params["padding_value"])

    with open(f"{model_dir}params.json", "w") as json_file:
        json.dump(params, json_file)

    print("Beginning Training")
    trainer.D.train()
    trainer.G.train()
    step = 0
    for epoch in range(epochs):
        for idx, (realMidis, _) in enumerate(dataIter):

            realMidis = realMidis.to(cudaDevice)
            fakeMidis = randMidiGen(realMidis, cudaDevice)

            attention_mask = initializeMask(realMidis, padding_value)

            d_loss, d_pred_real, d_pred_fake = trainer.trainDiscriminator(realMidis, fakeMidis, attention_mask)

            if step % g_step == 0:
                g_loss = trainer.trainGenerator(fakeMidis, attention_mask)

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