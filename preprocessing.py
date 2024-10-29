# %%
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

pl.Config(tbl_cols=50)  # This is the equivalent of Pandas' number of columns extension
from settings import (
    TRANSACTIONS_FILE_PATH,
    REGIONS_FILE_PATH,
    NB_TRANSACTIONS_PER_MONTH,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
)
from datetime import datetime
import seaborn as sns

from sklearn.model_selection import cross_validate, KFold
import matplotlib.pyplot as plt

from data_processing_functions import (
    load_and_process_transactions,
    get_info_per_month_cities_enough_transactions,
    add_classification_target_to_transactions,
    remove_departments_with_few_transactions,
    remove_regions_with_few_transactions,
    load_regions_data,
)


# %%
filtered_transactions = load_and_process_transactions(
    file_path=TRANSACTIONS_FILE_PATH, lower_bound_date=datetime(2018, 1, 1)
)

original_columns = filtered_transactions.columns


# %%
average_per_month_per_city_enough_transactions, average_per_month_per_city = (
    get_info_per_month_cities_enough_transactions(
        filtered_transactions,
        grouping_cols=[
            "departement",
            "ville",
            "id_ville",
            "annee_transaction",
            "mois_transaction",
        ],
        verbose=False,
    )
)

# %%


# %%
"""
EXPLORATOIRE
Cette analyse montre que seules 260 villes ont plus de 2 transactions immobilières par mois
Utile dans le chapitre de la partie 4 pour montrer que le modèle ne peut fonctionner que dans des mois où on a assez de données
"""

min_nb_transacation_par_ville = average_per_month_per_city.group_by(
    ["departement", "ville", "id_ville"]
).agg(pl.min(NB_TRANSACTIONS_PER_MONTH))

min_nb_transacation_par_ville.filter(
    pl.col(NB_TRANSACTIONS_PER_MONTH) > pl.quantile(NB_TRANSACTIONS_PER_MONTH, 0.75)
).select(NB_TRANSACTIONS_PER_MONTH).describe()

"""
FIN EXPLORATOIRE
"""

# %%
filtered_transactions = filtered_transactions.join(
    average_per_month_per_city_enough_transactions,
    on=["departement", "ville", "id_ville", "annee_transaction", "mois_transaction"],
    how="inner",
)


# %%

# average_per_month_per_city_enough_transactions = average_per_month_per_city_enough_transactions.drop("ville")
filtered_transactions = add_classification_target_to_transactions(
    filtered_transactions, "en_dessous_du_marche", 0.1
)


# %%
filtered_transactions, departments_to_keep = remove_departments_with_few_transactions(
    filtered_transactions, threshold_percentile=0.25, verbose=False
)

# %%
filtered_transactions = filtered_transactions.with_columns(
    pl.col("annee_transaction").cast(pl.Int32),
    pl.col("mois_transaction").cast(pl.Int32),
    pl.col("departement").cast(pl.Int32),
    pl.col("vefa").cast(pl.Int32),
)

# %%


departements_regions = load_regions_data(REGIONS_FILE_PATH, departments_to_keep)


# %%
filtered_transactions = filtered_transactions.join(
    departements_regions, how="left", on="departement"
)


filtered_transactions = remove_regions_with_few_transactions(filtered_transactions)


# %%
filtered_transactions.columns

# %%

filtered_transactions.write_parquet("transactions_immobilieres.parquet")

# %%
average_per_month_per_city_enough_transactions.write_parquet(
    "transactions_par_ville.parquet"
)
