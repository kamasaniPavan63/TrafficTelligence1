import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# ✅ Updated paths after folder rename
model = joblib.load('Flask/model.pkl')
scale = joblib.load('Flask/encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    try:
        input_feature = [float(x) for x in request.form.values()]
        features_values = [np.array(input_feature)]

        names = ['holiday', 'temp', 'rain', 'snow', 'weather',
                 'year', 'month', 'day', 'hours', 'minutes', 'seconds']

        data = pd.DataFrame(features_values, columns=names)
        scaled_data = scale.transform(data)
        prediction = model.predict(scaled_data)[0]

        # 👉 Show result.html instead of index.html
        return render_template("result.html", prediction_text=f"Estimated Traffic Volume is: {int(prediction)}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {str(e)}")

# ✅ Run app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True, use_reloader=False)
