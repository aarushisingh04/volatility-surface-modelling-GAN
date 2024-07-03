import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['DATE'])
    data['DATE'] = pd.to_datetime(data['DATE'])
    return data

def preprocess_data(data):
    scaler = MinMaxScaler()
    data[['OPEN', 'HIGH', 'LOW', 'CLOSE']] = scaler.fit_transform(data[['OPEN', 'HIGH', 'LOW', 'CLOSE']])
    features = data[['OPEN', 'HIGH', 'LOW']]
    target = data['CLOSE']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    file_path = 'data/vix_data.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
