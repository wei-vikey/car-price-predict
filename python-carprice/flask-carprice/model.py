import joblib


class CarPriceModel:
	def __init__(self):
		self.onehot = None
		self.scaler = None
		self.predictor = None

	def load_models(self):
		print("load_models")
		model_dir = "./models"
		self.onehot = joblib.load(f"{model_dir}/oneHotEncoder.joblib")
		self.scaler = joblib.load(f"{model_dir}/standardScaler.joblib")
		self.predictor = joblib.load(f"{model_dir}/random_model.joblib")


carPriceModel = CarPriceModel()
carPriceModel.load_models()
