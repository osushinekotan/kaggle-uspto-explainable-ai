import polars as pl
import rootutils
from tqdm import tqdm

ROOT_DIR = rootutils.setup_root(".", cwd=True)

DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "inputs"
COMPETITION_DIR = INPUT_DIR / "uspto-explainable-ai"

SQLITE_PATH = f"sqlite:///{(COMPETITION_DIR / 'uspto.db').as_posix()}"


def main() -> None:
    patent_metadata_df = (
        pl.scan_parquet(COMPETITION_DIR / "patent_metadata.parquet")
        .filter(pl.col("publication_date") >= pl.date(1975, 1, 1))
        .select(["publication_number", "publication_date"])
        .with_columns(
            pl.col("publication_date").dt.year().alias("year"),
            pl.col("publication_date").dt.month().alias("month"),
        )
        .collect()
    )

    for (year, month), _ in tqdm(
        patent_metadata_df.group_by(["year", "month"]),
        total=patent_metadata_df.select(["year", "month"]).n_unique(),
    ):
        patent_path = COMPETITION_DIR / f"patent_data/{year}_{month}.parquet"
        patent_df = pl.scan_parquet(patent_path).select(pl.exclude(["claims", "description"]))

        # https://docs.pola.rs/api/python/dev/reference/api/polars.DataFrame.write_database.html
        patent_df.write_database(
            table_name="patent_data",
            connection=SQLITE_PATH,
            engine="abbc",
            if_table_exists="append",
        )


if __name__ == "__main__":
    main()
