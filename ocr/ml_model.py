import pandas as pd


class DummyModel:
    def predict(self, X):
        return ["BUY" for _ in range(len(X))]


model = DummyModel()


def predict_from_matches_df(df_matches: pd.DataFrame) -> pd.DataFrame:
    """ Return new df with an added 'prediction' column.
    """
    # Remove lines without book_id
    df = df_matches.dropna(subset=["book_id"]).copy()


    X = df[["book_id"]]

    # Use model
    preds = model.predict(X)

    df["prediction"] = preds
    return df
