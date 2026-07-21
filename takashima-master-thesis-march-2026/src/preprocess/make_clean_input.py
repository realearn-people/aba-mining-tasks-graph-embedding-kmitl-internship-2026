import pandas as pd

def make_clean_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["ID", "Assumption", "Proposition"]]
    return df

if __name__ == "__main__":
    input_file_path = "data/input/dirty_Silver_Staff_ContN_BodyP.csv"
    output_file_path = "data/input/Silver_Staff_ContN_BodyP.csv"

    df = pd.read_csv(input_file_path)
    df = make_clean_input(df)
    df.to_csv(output_file_path, index=False)