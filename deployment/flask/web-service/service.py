import os
#Import Flask
from flask import Flask, request
from flask_cors import CORS
from keras.preprocessing import image
from ann_loader import cargarModelo
import numpy as np

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 5000
port = int(os.getenv('PORT', 5000))
print ("Port recognized: ", port)

#Initialize the application service
app = Flask(__name__)
CORS(app)
#global loaded_model,loaded_scaler, graph loaded_model,loaded_scaler, graph = cargarModelo()
global loaded_model,loaded_scaler, graph
loaded_model,loaded_scaler, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Modelo desplegado en la Nube!'

@app.route('/enfermedad/', methods=['GET','POST'])
def churn():
	return 'Modelo de Deteccion de Diabetes!'

@app.route('/enfermedad/paciente/', methods=['GET','POST'])
def default():
	# print (request.data)
	# print (request.args)
	# print (request.form)
	data = None
	if request.method == 'GET':
		print ("GET Method")
		data = request.args

	if request.method == 'POST':
		print ("POST Method")
		if (request.is_json):
			data = request.get_json()

	print("Data received:", data)

	# Obteniendo parametros
	Embarazos = data.get("Embarazos")
	Glucosa = data.get("Glucosa")
	Presion = data.get("Presion")
	EspesorPiel = data.get("EspesorPiel")
	Insulina = data.get("Insulina")
	IMC = data.get("IMC")
	DiabetesFamiliar = data.get("DiabetesFamiliar")
	Edad = data.get("Edad")

	print ("\nEmbarazos: ",Embarazos,
			"\nGlucosa: ", Glucosa,
			"\nPresion: ", Presion,
			"\nEspesorPiel: ", EspesorPiel,
			"\nInsulina: ", Insulina,
			"\nIMC: ", IMC,
			"\nDiabetesFamiliar: ", DiabetesFamiliar,
			"\nEdad: ", Edad)

	# Transformado/Escalando la data
	#[pais] = loaded_labelEncoderX1.transform([pais])
	#[genero] = loaded_labelEncoderX2.transform([genero])

	paciente = np.array([Embarazos,Glucosa,Presion,EspesorPiel,Insulina,IMC,DiabetesFamiliar,Edad])
	print("\npaciente: ", paciente)
	paciente = loaded_scaler.transform([paciente])
	print("cliente Norm: ", paciente)

	with graph.as_default():
		resultado = ""
		score = loaded_model.predict(paciente)
		print("\nFinal score: ", score)
		diabetico = (score > 0.5)
		if diabetico:
			resultado += "Diabetico"
		else:
		    resultado += "No Diabetico"
		return resultado + ', score: ' + str(score[0])

	# http://localhost:5000/abandono/cliente/?scoreCrediticio=3&pais=France&genero=Male&edad=36&tenencia=2&balance=1200.34&numDeProductos=3&tieneTarjetaCredito=1&esMiembroActivo=0&salarioEstimado=120000
	# http://localhost:5000/abandono/cliente/?scoreCrediticio=1&pais=Spain&genero=Female&edad=50&tenencia=2&balance=200.34&numDeProductos=1&tieneTarjetaCredito=0&esMiembroActivo=0&salarioEstimado=85000

# Run de application
app.run(host='0.0.0.0',port=port)
