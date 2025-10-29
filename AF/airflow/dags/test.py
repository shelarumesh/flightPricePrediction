from datetime import datetime, timedelta

from Etl import extract_load_transform_save

import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder


etl = extract_load_transform_save()
load_data = etl.data_loader()
transform_data = etl.data_transformation()
encode_data = etl.data_encoding()
train_model = etl.model_traning()
