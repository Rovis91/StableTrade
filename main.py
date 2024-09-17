from src.logger import setup_logger
from src.data_loader import DataLoader

if __name__ == "__main__":
    logger = setup_logger(__name__)
    data_loader = DataLoader(data_path=r'D:\StableTrade_dataset\EUTEUR_1m\EUTEUR_1m_final_merged.csv')
    data_loader.load_data()
