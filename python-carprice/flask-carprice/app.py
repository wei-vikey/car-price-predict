from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import model

app = Flask("carprice_predict")


def get_predict_result(df):
	# datajson = """{"year":{"6046":2017},"selling_price":{"6046":2100000},"km_driven":{"6046":48000},"fuel":{"6046":"Diesel"},"seller_type":{"6046":"Individual"},"transmission":{"6046":"Automatic"},"owner":{"6046":"Second Owner"},"mileage":{"6046":17.9},"engine":{"6046":2143.0},"max_power":{"6046":136.0},"torque":{"6046":3000.0},"seats":{"6046":5.0}}"""
	# df = pd.read_json(datajson)

	cat_columns = ["fuel", "seller_type", "transmission", "owner"]
	load_cat_fatures = model.carPriceModel.onehot.transform(df[cat_columns]).toarray()

	num_columns = ["mileage", "engine", "max_power", "torque", "seats"]
	load_num_fatures = model.carPriceModel.scaler.transform(df[num_columns])

	load_final_fatures = np.hstack([load_cat_fatures, load_num_fatures])
	result = model.carPriceModel.predictor.predict(load_final_fatures)

	print(result)
	return result


@app.route("/predict", methods=["get", "post"])
def predict():
	result = None
	if request.method == "POST":
		data = dict(request.form)
		df = pd.DataFrame([data.values()], columns=data.keys())
		result = str(get_predict_result(df))
	return render_template("predict.html", result=result)


app.run(host="0.0.0.0", port=5010)
