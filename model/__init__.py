import subprocess
import sys

# https://stackoverflow.com/questions/76448287/how-can-i-solve-importerror-using-the-trainer-with-pytorch-requires-accele
try:
    import transformers
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "accelerate"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", "transformers"]
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", "huggingface_hub[cli,torch]"]
    )
finally:
    from huggingface_hub import notebook_login

try:
    import datasets
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
finally:
    from datasets import Dataset

try:
    import evaluate
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "evaluate"])
finally:
    from datasets import Dataset

try:
    import sklearn
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", "scikit-learn"]
    )
finally:
    from sklearn.model_selection import train_test_split, GridSearchCV

try:
    import xgboost
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
finally:
    from .xgb_boost import xgb_train_new, xgb_plot_results


from .embed import embed_batch, embed_case

# from .xgb_boost import xgb_train_new, xgb_plot_results
from .eval import plot_confusion, print_metrics

# from sklearn.model_selection import train_test_split, GridSearchCV
from .legal_bert import (
    load_trained_model,
    get_new_trainer,
    predict_labels,
    tokenize_dataset,
)
from datasets import load_dataset, Dataset, load_metric, DatasetDict
