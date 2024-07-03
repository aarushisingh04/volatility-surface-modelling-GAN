import unittest
# from data_preprocessing import load_data
import pandas as pd

def load_data(file_path):
        data = pd.read_csv(file_path, parse_dates=['DATE'])
        data['DATE'] = pd.to_datetime(data['DATE'])
        return data

class TestDataPreprocessing(unittest.TestCase):

    def test_load_data(self):
        data = load_data('data/vix_data.csv')
        self.assertEqual(data.shape[0], 8791)

if __name__ == '__main__':
    unittest.main()
