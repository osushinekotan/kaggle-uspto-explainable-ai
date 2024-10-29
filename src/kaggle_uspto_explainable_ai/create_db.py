import sqlite3
from itertools import chain

import polars as pl
import rootutils
from loguru import logger
from tqdm import tqdm

ROOT_DIR = rootutils.setup_root(".", cwd=True)

DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "inputs"
COMPETITION_DIR = INPUT_DIR / "uspto-explainable-ai"

SQLITE_PATH = INPUT_DIR / "db" / "uspto.db"
SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

TABLE_NAME = "patent_data"


def main() -> None:
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        existing_publication_numbers = list(
            chain.from_iterable(set(conn.execute("SELECT DISTINCT publication_number FROM patent_data").fetchall()))
        )
        conn.close()

    except sqlite3.OperationalError:
        existing_publication_numbers = []
    logger.info(f"# of existing year-month pairs: {len(existing_publication_numbers)}")

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
        conn = sqlite3.connect(SQLITE_PATH)
        patent_path = COMPETITION_DIR / f"patent_data/{year}_{month}.parquet"
        _ = (
            pl.scan_parquet(patent_path)
            .filter(pl.col("publication_number").is_in(existing_publication_numbers).not_())
            .collect()
            .to_pandas()
            .to_sql(
                name=TABLE_NAME,
                con=conn,
                if_exists="append",
                index=False,
                chunksize=500,
            )
        )
        conn.close()


if __name__ == "__main__":
    main()
