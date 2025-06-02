import alpaca_trade_api as tradeapi

api = tradeapi.REST("PKQPXKCSP3FXIZG65FVZ", "UVEDR5uXs1eSDdRLrhe8TCpZOCkbtLrYnCtmZ6dh", base_url="https://paper-api.alpaca.markets")

account = api.get_account()
print(account)


