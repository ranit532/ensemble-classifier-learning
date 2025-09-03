from src.data_loader import load_data

def main():
    X, y = load_data()
    print('🚀 Data loaded with shape:', X.shape, y.shape)
    print('Now you can train classifiers using train_models.py')

if __name__ == "__main__":
    main()
