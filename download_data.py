import gdown

print("Downloading Bitcoin Market Sentiment...")
gdown.download(id='1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf', output='sentiment_data.csv', quiet=False)

print("Downloading Historical Trader Data...")
gdown.download(id='1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs', output='trader_data.csv', quiet=False)

print("Done.")
