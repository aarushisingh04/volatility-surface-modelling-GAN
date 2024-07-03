import matplotlib.pyplot as plt
import torch

def plot_results(real_data, generated_data):
    plt.figure(figsize=(10, 5))
    plt.plot(real_data, label='Real Data')
    plt.plot(generated_data, label='Generated Data', linestyle='--')
    plt.legend()
    plt.show()

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
