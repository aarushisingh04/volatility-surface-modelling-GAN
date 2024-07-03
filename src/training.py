import torch
import torch.optim as optim
import torch.nn as nn
from volatility_surface import Generator, Discriminator
from data_preprocessing import load_data, preprocess_data

def train_gan(X_train, num_epochs=1000, batch_size=32, lr=0.0002):
    generator = Generator(input_dim=100)
    discriminator = Discriminator()
    
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        noise = torch.randn(batch_size, 100)
        fake_data = generator(noise)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{num_epochs}')

if __name__ == "__main__":
    file_path = 'data/vix_data.csv'
    data = load_data(file_path)
    X_train, _, _, _, _ = preprocess_data(data)
    train_gan(X_train)
