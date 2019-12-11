# -----------------------------------------------------------
# Cargando modelo de disco
import tensorflow as tf
from keras.models import load_model
from sklearn.externals import joblib

def cargarModelo():

    FILENAME_MODEL_TO_LOAD = "model.h5"
    FILENAME_SCALER_TO_LOAD = "stdScaler.save"
    #FILENAME_LABELENCODER_X1_TO_LOAD = "labelEncoder_X_1.save"
    #FILENAME_LABELENCODER_X2_TO_LOAD = "labelEncoder_X_2.save"
    MODEL_PATH = "../../../models/diabetes"

    # Cargar la RNA desde disco
    loaded_model = load_model(MODEL_PATH + "/" + FILENAME_MODEL_TO_LOAD)
    print("Modelo cargado de disco << ", loaded_model)

    # Cargar los parametros usados
    loaded_scaler = joblib.load(MODEL_PATH + "/" + FILENAME_SCALER_TO_LOAD)
    #loaded_labelEncoderX1 = joblib.load(MODEL_PATH + "/" + FILENAME_LABELENCODER_X1_TO_LOAD)
    #loaded_labelEncoderX2 = joblib.load(MODEL_PATH + "/" + FILENAME_LABELENCODER_X2_TO_LOAD)

    graph = tf.get_default_graph()
    return loaded_model,loaded_scaler,graph
