from data_preprocessing import load_data, preprocess_data
from training import train_gan, train_vae
from utils import generate_volatility_surface, evaluate_model
import matplotlib.pyplot as plt


data = load_data('data/vix_data.csv')
processed_data = preprocess_data(data)
gan_model = train_gan(processed_data)
vae_model = train_vae(processed_data)

volatility_surface = generate_volatility_surface(gan_model)



plt.figure(figsize=(10, 6))
plt.imshow(volatility_surface, cmap='viridis')
plt.colorbar(label='Volatility')
plt.title('Generated Volatility Surface')
plt.xlabel('Maturity')
plt.ylabel('Strike Price')
plt.show()


real_volatility_surface = load_data('data/real_volatility_surface.csv')
performance_metrics = evaluate_model(real_volatility_surface, volatility_surface)
print(performance_metrics)
