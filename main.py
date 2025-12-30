from src.preprocess import run_preprocessing
from src.train import train
from src.finetune import finetune_models

def main():
    X_train_scaled, y_train, X_test_scaled, y_test = run_preprocessing()

    if X_train_scaled is None:
        return

    train(
        X_train_scaled, y_train,
        X_test_scaled, y_test
    )

    finetune_models(
        X_train_scaled, y_train,
        X_test_scaled, y_test
    )

if __name__ == "__main__":
    main()
