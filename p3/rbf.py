#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:37:04 2016

@author: pagutierrez
"""

# TODO Incluir todos los import necesarios
# Librería para codificar y decodificar archivos para devolver la jerquía
# de clases entre objetos.
import pickle
# Este módulo proporciona una forma portátil de utilizar la funcionalidad
# dependiente del sistema operativo.
import os

# para solucionar conflictos entre versiones
import sys
sys.excepthook = sys.__excepthook__


# Poner para que es cada librería
import click  # Para crear interfaces de línea de comando.
import math  # Paquete de recursos matemáticos.
import numpy as np  # Paquete para la computación de matrices.
# Proporciona estructuras de datos y herramientas de análisis de datos de
# alto rendimiento.
import pandas as pd
import random as rd  # Para generar números aleatorios.
import warnings  # Para silenciar los warnings

# Mejor importar módulo a módulo (como el Cholo), para que ejecute más rápido.

# import sklearn #Módulo para Machine Learning en Python.
# Módulos scikit learn:
# De este paquete usamos el método para calcular el KMeans.
import sklearn.cluster
# Lo usamos porque incluye métodos para calcular el MSE y el CCR.
import sklearn.metrics
# Para cambiar los vectores de características sin procesar en una
# representación que sea más adecuada para los estimadores posteriores.
import sklearn.preprocessing
# Para calcular la matriz de distancia, para los Centroides.
import scipy.spatial.distance
import sklearn.linear_model  # Para la función de Regresión Logística.
# Para inicializar los centroides?
from sklearn.model_selection import train_test_split

# Para silenciar los warnigns de Python (son mu pesaos))
warnings.filterwarnings('ignore')


@click.command()
# TODO incluir el resto de parámetros...
@click.option('--train_file', '-t', default=None, required=True,
              help=u'Fichero con los datos de entrenamiento.')
@click.option('--test_file', '-T', default=None, required=False,
              help=u'Fichero con los datos de test.')
@click.option('--classification', '-c', is_flag=True, default=False, required=False,
              help=u'Activar para problemas de clasificacion.')
@click.option('--l2', '-l', is_flag=True, default=False, required=False,
              help=u'Activar para utilizar la regularización de L2 en lugar de la regularización L1 (regresión logística).')
@click.option('--eta', '-e', default=0.01, required=False, show_default=True,
              help=u'Valor del parametro de regularizacion para la regresión logistica.')
@click.option('--outputs', '-o', default=1, required=False, show_default=True,
              help=u'Numero de variables que se tomarán como salidas (siempre estan al final de la matriz).')
@click.option('--ratio_rbf', '-r', default=0.1, required=False, show_default=True,
              help=u'Ratio (en tanto por uno) de neuronas RBF con respecto al total de patrones.')
@click.option('--model_file', '-m', default="", show_default=True, required=False,
              help=u'Fichero en el que se guardará o desde el que se cargará el modelo (si existe el flag p).')  # KAGGLE
@click.option('--pred', '-p', is_flag=True, default=False, show_default=True, required=False,
              help=u'Activar el modo de predicción.')  # KAGGLE
# Función para entrenar de forma supervisada la RBF. Se le introducen los
# parámetros necesarios.
def entrenar_rbf_total(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model_file, pred):
    """ Modelo de aprendizaje supervisado mediante red neuronal de tipo RBF.
        Ejecución de 5 semillas.
    """
    # Si no es activado el modo de predicción.
    if not pred:

        # Si el conjunto de entrenamiento no se ha introducido, se muestra
        # mensaje de error por pantalla.
        if train_file is None:
            print("No se ha especificado el conjunto de entrenamiento (-t)")
            return

        # Inicializamos los vectores en los que guardamos los resultados
        # obtenidos con CCR y MSE para los conjuntos de Train y Test.
        train_mses = np.empty(5)
        train_ccrs = np.empty(5)
        test_mses = np.empty(5)
        test_ccrs = np.empty(5)

        # Leemos aquí y ahorramos timpo para la línea 91.
        train_inputs, train_outputs, test_inputs, test_outputs = lectura_datos(
            train_file, test_file, outputs)

        # Desde 1 hasta 5 incrementando de uno en uno? (preguntar chema)
        # Por defecto es 1 el incremento(último 1 del paréntesis es el
        # incremento).
        for s in range(1, 6, 1):
            # print("-----------")
            #print("Semilla: %d" % s)
            # print("-----------")

            # Se inicializan las semillas de forma aleatoria.
            np.random.seed(s)

            train_mses[s - 1], test_mses[s - 1], train_ccrs[s - 1], test_ccrs[s - 1] = \
                entrenar_rbf(train_inputs, train_outputs, test_inputs, test_outputs, classification, ratio_rbf, l2, eta, outputs,
                             model_file and "{}/{}.pickle".format(model_file, s) or "")

            # Chivatos del MSE y CCR obtenidos en cada iteración (semilla) para los conjuntos de entrenamiento y de test.
            #print("MSE de entrenamiento: %f" % train_mses[s-1])
            #print("MSE de test: %f" % test_mses[s-1])
            #print("CCR de entrenamiento: %.2f%%" % train_ccrs[s-1])
            #print("CCR de test: %.2f%%" % test_ccrs[s-1])

        # Chivato resumen de los datos obtenidos (se imprime solo el MSE a no
        # ser que sea para clasificación)
        print("*********************")
        print("Resumen de resultados")
        print("*********************")
        print("MSE de entrenamiento: %f +- %f" %
              (np.mean(train_mses), np.std(train_mses)))
        print("MSE de test: %f +- %f" %
              (np.mean(test_mses), np.std(test_mses)))
        # Para ver el CCR en el último apartado.
        #print("CCR de entrenamiento: %.2f%% +- %.2f%%" %
        #      (np.mean(train_ccrs), np.std(train_ccrs)))
        #print("CCR de test: %.2f%% +- %.2f%%" %
        #      (np.mean(test_ccrs), np.std(test_ccrs)))
        # Si es un problema de clasificación se imprime también el CCR
        if classification:
            print("CCR de entrenamiento: %.2f%% +- %.2f%%" %
                  (np.mean(train_ccrs), np.std(train_ccrs)))
            print("CCR de test: %.2f%% +- %.2f%%" %
                  (np.mean(test_ccrs), np.std(test_ccrs)))

    else:  # Si está activado el modo de predicción (se usa para el Kaggle)

        # KAGGLE
        if model_file is None:
            print("No se ha indicado un fichero que contenga el modelo (-m).")
            return

        # Obtener predicciones para el conjunto de test
        predictions = predict(test_file, model_file)

        # Imprimir las predicciones en formato csv
        print("Id,Category")
        for prediction, index in zip(predictions, range(len(predictions))):
            s = ""
            s += str(index)

            if isinstance(prediction, np.ndarray):
                for output in prediction:
                    s += ",{}".format(output)
            else:
                s += ",{}".format(int(prediction))

            print(s)

# Función para entrenar a la red RBF. ¿Diferencia entre esta función y la
# de arriba? (preguntar a chema)


def entrenar_rbf(train_inputs, train_outputs, test_inputs, test_outputs, classification, ratio_rbf, l2, eta, outputs, model_file=""):
    """ Modelo de aprendizaje supervisado mediante red neuronal de tipo RBF.
        Una única ejecución.
        Recibe los siguientes parámetros:
            - train_file: nombre del fichero de entrenamiento.
            - test_file: nombre del fichero de test.
            - classification: True si el problema es de clasificacion.
            - ratio_rbf: Ratio (en tanto por uno) de neuronas RBF con 
              respecto al total de patrones.
            - l2: True si queremos utilizar L2 para la Regresión Logística. 
              False si queremos usar L1 (para regresión logística).
            - eta: valor del parámetro de regularización para la Regresión 
              Logística.
            - outputs: número de variables que se tomarán como salidas 
              (todas al final de la matriz).
        Devuelve:
            - train_mse: Error de tipo Mean Squared Error en entrenamiento. 
              En el caso de clasificación, calcularemos el MSE de las 
              probabilidades predichas frente a las objetivo.
            - test_mse: Error de tipo Mean Squared Error en test. 
              En el caso de clasificación, calcularemos el MSE de las 
              probabilidades predichas frente a las objetivo.
            - train_ccr: Error de clasificación en entrenamiento. 
              En el caso de regresión, devolvemos un cero.
            - test_ccr: Error de clasificación en test. 
              En el caso de regresión, devolvemos un cero.
    """

    #train_inputs, train_outputs, test_inputs, test_outputs = lectura_datos(train_file, test_file, outputs)

    # TODO: Obtener num_rbf a partir de ratio_rbf
    # redondeamos el producto de ratio_rbf con el número de patrones de
    # entrada para obtener el número de neuronas de la red.
    num_rbf = round(ratio_rbf * len(train_inputs))

    #print("Número de RBFs utilizadas: %d" %(num_rbf))

    kmedias, distancias, centros = clustering(
        classification, train_inputs, train_outputs, num_rbf)

    radios = calcular_radios(centros, num_rbf)

    matriz_r = calcular_matriz_r(distancias, radios)

    # Si no es clasificación.
    if not classification:

        # Beta traspuesta = inversa R * Y
        # Matriz coeficientes = (salidas RBF ^ -1) * Salidas predecidas
        # R * Bt = Y
        coeficientes = invertir_matriz_regresion(matriz_r, train_outputs)
        train_predictions = np.matmul(matriz_r, coeficientes)
        train_mse = sklearn.metrics.mean_squared_error(
            train_predictions, train_outputs)
        # Para el último apartado:
        #train_ccr = 100 * \
        #    sklearn.metrics.accuracy_score(
        #        train_outputs, np.around(train_predictions))
        train_ccr = 0

    # Si lo es.
    else:
        logreg = logreg_clasificacion(matriz_r, train_outputs, eta, l2)
        predicciones = logreg.predict_proba(matriz_r)
        salidas_train = sklearn.preprocessing.OneHotEncoder(
            categories='auto').fit_transform(train_outputs).toarray()
        train_mse = sklearn.metrics.mean_squared_error(
            predicciones, salidas_train)
        train_ccr = 100 * logreg.score(matriz_r, train_outputs)

    """
    TODO: Calcular las distancias de los centroides a los patrones de test y la matriz R de test
    """

    distancias_test = kmedias.transform(test_inputs)
    matriz_r_test = calcular_matriz_r(distancias_test, radios)

    # # # # KAGGLE # # # #
    if model_file != "":
        save_obj = {
            'classification': classification,
            'radios': radios,
            'kmedias': kmedias
        }
        if not classification:
            save_obj['coeficientes'] = coeficientes
        else:
            save_obj['logreg'] = logreg

        dir = os.path.dirname(model_file)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        with open(model_file, 'wb') as f:
            pickle.dump(save_obj, f)

    # # # # # # # # # # #

    if not classification:
        """
        TODO: Obtener las predicciones de entrenamiento y de test y calcular el MSE
        """
        test_predictions = np.matmul(matriz_r_test, coeficientes)
        test_mse = sklearn.metrics.mean_squared_error(
            test_predictions, test_outputs)
        #test_ccr = 100 * \
        #    sklearn.metrics.accuracy_score(
        #        test_outputs, np.around(test_predictions))
        test_ccr = 0

    else:
        """
        TODO: Obtener las predicciones de entrenamiento y de test y calcular
              el CCR. Calcular también el MSE, comparando las probabilidades 
              obtenidas y las probabilidades objetivo
        """
        test_predictions = logreg.predict(matriz_r_test)
        predicciones = logreg.predict_proba(matriz_r_test)
        # OneHotEncoder para transformar las salidas en matriz de
        # categorización (ver apuntes).
        salidas_test = sklearn.preprocessing.OneHotEncoder(
            categories='auto').fit_transform(test_outputs).toarray()

        test_mse = sklearn.metrics.mean_squared_error(
            predicciones, salidas_test)
        test_ccr = 100 * logreg.score(matriz_r_test, test_outputs)
        print("Matríz de confusión del Test")
        print(sklearn.metrics.confusion_matrix(test_outputs, test_predictions))

    return train_mse, test_mse, train_ccr, test_ccr


def lectura_datos(fichero_train, fichero_test, outputs):
    """ Realiza la lectura de datos.
        Recibe los siguientes parámetros:
            - fichero_train: nombre del fichero de entrenamiento.
            - fichero_test: nombre del fichero de test.
            - outputs: número de variables que se tomarán como salidas 
              (todas al final de la matriz).
        Devuelve:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - test_inputs: matriz con las variables de entrada de 
              test.
            - test_outputs: matriz con las variables de salida de 
              test.
    """

    # TODO: Completar el código de la función

    train = pd.read_csv(fichero_train, header=None)
    train = np.array(train)
    train = train.astype(np.float64)
    train_inputs = train[:, :-outputs]
    train_outputs = train[:, -outputs:]

    test = pd.read_csv(fichero_test, header=None)
    test = np.array(test)
    test = test.astype(np.float64)
    test_inputs = test[:, :-outputs]
    test_outputs = test[:, -outputs:]

    return train_inputs, train_outputs, test_inputs, test_outputs


def inicializar_centroides_clas(train_inputs, train_outputs, num_rbf):
    """ Inicializa los centroides para el caso de clasificación.
        Debe elegir los patrones de forma estratificada, manteniendo
        la proporción de patrones por clase.
        Recibe los siguientes parámetros:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - centroides: matriz con todos los centroides iniciales
                          (num_rbf x num_entradas).
    """

    # TODO: Completar el código de la función
    x, centroides, y_train, y_test = train_test_split(
        train_inputs, train_outputs, stratify=train_outputs, test_size=num_rbf / len(train_inputs))

    return centroides


def clustering(clasificacion, train_inputs, train_outputs, num_rbf):
    """ Realiza el proceso de clustering. En el caso de la clasificación, se
        deben escoger los centroides usando inicializar_centroides_clas()
        En el caso de la regresión, se escogen aleatoriamente.
        Recibe los siguientes parámetros:
            - clasificacion: True si el problema es de clasificacion.
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - kmedias: objeto de tipo sklearn.cluster.KMeans ya entrenado.
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - centros: matriz (num_rbf x num_entradas) con los centroides 
              obtenidos tras el proceso de clustering.
    """

    # TODO: Completar el código de la función
    if(clasificacion):
        centroides = inicializar_centroides_clas(
            train_inputs, train_outputs, num_rbf)
        kmedias = sklearn.cluster.KMeans(
            len(centroides), centroides, 1, 500).fit(train_inputs, train_outputs)
        # cambiamos el algoritmo
        #kmedias = sklearn.cluster.KMeans(num_rbf, init='k-means++', n_init=1, max_iter=500).fit(train_inputs,train_outputs)
    else:
            #kmedias = sklearn.cluster.KMeans(num_rbf, init='k-means++', n_init=1, max_iter=500).fit(train_inputs,train_outputs)
        kmedias = sklearn.cluster.KMeans(
            num_rbf, init='random', n_init=1, max_iter=500).fit(train_inputs, train_outputs)

    centros = kmedias.cluster_centers_

    distancias = kmedias.transform(train_inputs)

    return kmedias, distancias, centros


def calcular_radios(centros, num_rbf):
    """ Calcula el valor de los radios tras el clustering.
        Recibe los siguientes parámetros:
            - centros: conjunto de centroides.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - radios: vector (num_rbf) con el radio de cada RBF.
    """

    # TODO: Completar el código de la función
    dist = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(centros))
    radios = np.array([], dtype=np.float64)

    for x in range(0, num_rbf):
        sumdist = 0
        sumdist = sum(dist[x])
        sumdist = sumdist / (2 * (num_rbf - 1))
        radios = np.append(radios, sumdist)

    return radios


def calcular_matriz_r(distancias, radios):
    """ Devuelve el valor de activación de cada neurona para cada patrón 
        (matriz R en la presentación)
        Recibe los siguientes parámetros:
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - radios: array (num_rbf) con el radio de cada RBF.
        Devuelve:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
    """

    # TODO: Completar el código de la función
    # Creamos una matriz vacia
    # Fórmula de la normal (función gaussiana, viene en los apuntes).
    matriz_r = np.empty([len(distancias), len(radios) + 1])
    matriz_r.astype(np.float64)
    for i in range(0, len(distancias)):
        for j in range(0, len(radios)):
            aux = math.exp(
                (distancias[i, j] * distancias[i, j]) / (-2 * radios[j] * radios[j]))
            matriz_r[i][j] = aux

    matriz_r[:, -1] = 1

    return matriz_r


def invertir_matriz_regresion(matriz_r, train_outputs):
    """ Devuelve el vector de coeficientes obtenidos para el caso de la 
        regresión (matriz beta en las diapositivas)
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
        Devuelve:
            - coeficientes: vector (num_rbf+1) con el valor del sesgo y del 
              coeficiente de salida para cada rbf.
    """

    # TODO: Completar el código de la función
    if len(matriz_r) == len(matriz_r[0]):
        coeficientes = np.matmul(np.linalg.inv(matriz_r), train_outputs)
    else:
        coeficientes = np.matmul(np.matmul(np.linalg.pinv(
            np.matmul(matriz_r.T, matriz_r)), matriz_r.T), train_outputs)

    return coeficientes


def logreg_clasificacion(matriz_r, train_outputs, eta, l2):
    """ Devuelve el objeto de tipo regresión logística obtenido a partir de la
        matriz R.
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - eta: valor del parámetro de regularización para la Regresión 
              Logística.
            - l2: True si queremos utilizar L2 para la Regresión Logística. 
              False si queremos usar L1.
        Devuelve:
            - logreg: objeto de tipo sklearn.linear_model.LogisticRegression ya
              entrenado.
    """

    # TODO: Completar el código de la función
    logreg = 0

    if l2:
        logreg = sklearn.linear_model.LogisticRegression(
            penalty='l2', C=1 / eta, solver='liblinear', multi_class='auto', max_iter=600)
    else:
        logreg = sklearn.linear_model.LogisticRegression(
            penalty='l1', C=1 / eta, solver='liblinear', multi_class='auto', max_iter=600)

    #print("Coeficiente: ", (1/eta))
    logreg.fit(matriz_r, train_outputs.ravel())

    return logreg


def predict(test_file, model_file):
    """ Calcula las predicciones para un conjunto de test que recibe como parámetro. Para ello, utiliza un fichero que
    contiene un modelo guardado.
    :param test_file: fichero csv (separado por comas) que contiene los datos de test.
    :param model_file: fichero de pickle que contiene el modelo guardado.
    :return: las predicciones para la variable de salida del conjunto de datos proporcionado.
    """

    test_df = pd.read_csv(test_file, header=None)
    test_inputs = test_df.values[:, :]

    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)

    radios = saved_data['radios']
    classification = saved_data['classification']
    kmedias = saved_data['kmedias']

    test_distancias = kmedias.transform(test_inputs)
    test_r = calcular_matriz_r(test_distancias, radios)

    if classification:
        logreg = saved_data['logreg']
        test_predictions = logreg.predict(test_r)
    else:
        coeficientes = saved_data['coeficientes']
        test_predictions = np.dot(test_r, coeficientes)

    return test_predictions


if __name__ == "__main__":
    entrenar_rbf_total()
