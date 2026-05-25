import os
import sqlite3
import pandas as pd

def load_data(local_path: str="data/phishing.db") -> pd.DataFrame:
    """
    Loads the phishing dataset from a SQLite database.

    Args:
        local_path (str, optional): Path to the SQLite database file.

    Returns:
        pandas.DataFrame: The loaded dataset containing all features and labels.
    """

    # If the DB file doesn't exist locally
    # place it in the `data/` folder before running the script.
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            "Expected data/phishing.db. Please place phishing.db in the data/ folder."
        )

    # Connect and read tables
    conn = sqlite3.connect(local_path)
    query = """
        SELECT
            LineOfCode,
            LargestLineLength,
            NoOfURLRedirect,
            NoOfSelfRedirect,
            NoOfPopup,
            NoOfiFrame,
            NoOfImage,
            NoOfSelfRef,
            NoOfExternalRef,
            Robots,
            IsResponsive,
            Industry,
            HostingProvider,
            DomainAgeMonths,
            label
        FROM phishing_data
        """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # clip NoOfImage to be non-negative
    df["NoOfImage"] = df["NoOfImage"].clip(lower=0)

    return df
