import importlib.util

# Define module name and file path
path = "D:\\AlmaBetter\\P01_travelPrice\\src\\airflow\\utils\\main.py"
mname = "main"

logger_path = "D:\\AlmaBetter\\P01_travelPrice\\src\\airflow\\utils\\logger_setup.py"
data_ingestion_path = "D:\\AlmaBetter\\P01_travelPrice\\src\\airflow\\utils\\data_ingestion.py"
data_transform_path = "D:\\AlmaBetter\\P01_travelPrice\\src\\airflow\\utils\\data_transformation.py"
model_training_path = "D:\\AlmaBetter\\P01_travelPrice\\src\\airflow\\utils\\model_training.py"

# Load module from specified file location
spec = importlib.util.spec_from_file_location(mname, path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Call the main function from the imported module
mod.main()