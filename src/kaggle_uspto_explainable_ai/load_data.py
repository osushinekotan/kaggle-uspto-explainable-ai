import kaggle

from kaggle_uspto_explainable_ai.kaggle_utils.dataset import download_kaggle_competition_dataset

kaggle_client = kaggle.KaggleApi()
kaggle_client.authenticate()


def main() -> None:
    download_kaggle_competition_dataset(
        client=kaggle_client,
        competition="uspto-explainable-ai",
        out_dir="data/inputs",
        force=False,
    )


if __name__ == "__main__":
    main()