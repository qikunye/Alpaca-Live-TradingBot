import pandas as pd
from lumibot.backtesting import PandasDataBacktesting
from lumibot.data_sources.pandas_data import PandasData

class Asset:
    def __init__(self, symbol):
        self.symbol = symbol
    def __str__(self):
        return self.symbol
    def __repr__(self):
        return f"Asset('{self.symbol}')"
    def __eq__(self, other):
        if isinstance(other, Asset):
            return self.symbol == other.symbol
        return False
    def __hash__(self):
        return hash(self.symbol)

class CSVDataBacktesting(PandasDataBacktesting):
    def __init__(self, **kwargs):
        symbols = kwargs.pop("symbols", [Asset("SPY"), Asset("AAPL"), Asset("MSFT")])
        datetime_start = kwargs.pop("datetime_start", pd.Timestamp("2019-10-01"))
        datetime_end = kwargs.pop("datetime_end", pd.Timestamp("2024-12-31"))

        # Ensure datetime_start and datetime_end are tz-naive
        datetime_start = pd.to_datetime(datetime_start)
        if getattr(datetime_start, 'tzinfo', None) is not None:
            datetime_start = datetime_start.tz_localize(None)

        datetime_end = pd.to_datetime(datetime_end)
        if getattr(datetime_end, 'tzinfo', None) is not None:
            datetime_end = datetime_end.tz_localize(None)

        assets = {str(s): s if isinstance(s, Asset) else Asset(s) for s in symbols}
        data_dict = {}

        for symbol_str in assets:
            df = pd.read_csv(f"{symbol_str}_ohlcv.csv", skiprows=2)
            df.rename(columns={"Date": "datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])

            # Make datetime tz-naive if tz-aware
            if df["datetime"].dt.tz is not None:
                df["datetime"] = df["datetime"].dt.tz_localize(None)

            df = df[df["datetime"].between(datetime_start, datetime_end)]
            df.set_index("datetime", inplace=True)

            pd_data = PandasData(
                data=df,
                datetime_start=pd.to_datetime(df.index[0]),
                datetime_end=pd.to_datetime(df.index[-1]),
            )
            asset = assets[symbol_str]
            pd_data.asset = asset
            data_dict[asset] = pd_data

        asset_list = list(assets.values())
        kwargs.pop("pandas_data", None) 

        super().__init__(
    datetime_start=datetime_start,
    datetime_end=datetime_end,
    assets=asset_list,
    pandas_data=data_dict,
    **kwargs
)

