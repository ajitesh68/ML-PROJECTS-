import streamlit as st
import numpy as np
import pickle

#Load the train model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌸 Iris Flower Classifier 🌸")

sepal_length = st.number_input("Sepal Length(cm)",min_value=4.0,max_value=8.0,value=5.4 )
sepal_width = st.number_input("Sepal Width(cm)",min_value=2.0,max_value=4.5,value=3.0)
petal_length = st.number_input("Petal Length(cm)",min_value=1.0,max_value=7.0,value=3.5)
petal_width = st.number_input("Petal Width(cm)",min_value=0.1,max_value=2.5,value=1.3)


if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = ["Setosa","Versicolor","Virginica"]
    st.write(f"Predicted Flower Species: 🌼 {species[prediction[0]]}")
