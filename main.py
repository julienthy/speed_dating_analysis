import logging

from src.utils.io import load_data, load_config
from src.utils.basic_visualization import *

from src.visualization.quantitative import *

# Configuration globale du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = load_config("config.yaml")
raw_path = config["data"]["raw_path"]
encoding = config["data"]["encoding"]

figures_path = config["visualization"]["save_path"]

df_speed_dating = load_data(raw_path, encoding=encoding)

display_head(df_speed_dating)
display_info(df_speed_dating)
display_description(df_speed_dating)
plot_missing_values(df_speed_dating, figures_path)

# plot_temporal_histograms(df_speed_dating, "attr1_", ["1", "2", "3"], "gender", save_path=figures_path)
plot_temporal_histograms2(df_speed_dating, "attr1_", ["1", "2", "3"], "gender", save_path=figures_path)

# plot_scatter_comparison(df_speed_dating, "attr1_1", "attr", "gender", save_path=figures_path)

# plot_violin_comparison(df_speed_dating, ["attr3_1", "attr_o"], "gender", save_path=figures_path)

# plot_boxplots_by_decision(df_speed_dating, ["attr", "sinc", "intel", "fun", "amb", "shar", "like", "prob"], "dec", save_path=figures_path)

# plot_scatter_comparison(df_speed_dating, "int_corr", "like", "dec", save_path=figures_path)

# plot_correlation_heatmap(df_speed_dating, ["attr", "sinc", "intel", "fun", "amb", "shar", "like", "prob", "dec"], save_path=figures_path)