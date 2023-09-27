import joblib
model = joblib.load('concrete_strength_81.pkl')
tf_res = joblib.load('yjtransformer.pkl')

new_prediction = model.predict(tf_res.transform([[389.9, 189.0, 0.0, 145.9, 22, 944.7, 755.8, 91]]))
print("The compressive strength of the concrete structure with the given input features is:",float(new_prediction)) 