import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*[Aa]ll-NaN.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
