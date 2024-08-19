# %%
import shap

# %%
# TO DO : LOAD DATA AND MODEL

# %%
explainer = shap.Explainer(model)
shap_values = explainer(X)


# %%
shap.plots.waterfall(shap_values[0])

# %%
shap.plots.beeswarm(shap_values)

# %%

shap.plots.scatter(shap_values[:, "Latitude"], color=shap_values)
