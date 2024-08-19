# %%
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="https://127.0.0.1")

# %%
all_experiments = client.search_experiments()

# %%
