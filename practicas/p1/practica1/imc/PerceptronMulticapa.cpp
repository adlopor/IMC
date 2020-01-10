/*********************************************************************
* File : PerceptronMulticapa.cpp
* Date : 2018
*********************************************************************/

#include "PerceptronMulticapa.h"
#include "util.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <limits>
#include <math.h>

#include <algorithm>

using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// CONSTRUCTOR: Dar valor por defecto a todos los parámetros
PerceptronMulticapa::PerceptronMulticapa(){

	//Variables privadas:
	pCapas = NULL;
	nNumCapas = 3;
	
	//Variables públicas:
	dEta = 0.1;//Ponemos como valor por defecto para la tasa de aprendizaje un valor relativamente pequeño.
	dMu = 0.9; //Empezamos poniendo un momento (inercia) incial, relativamente grande, para que no se estanque al principio de la ejecución y puedo explorar bien.
	dValidacion = 0; //El conjunto de datos es todo para entrenamiento, por defecto.
	dDecremento = 1; //El factor de decremento por capas para la tasa de aprendizaje(dEta).
}

// ------------------------------
// Reservar memoria para las estructuras de datos
int PerceptronMulticapa::inicializar(int nl, int npl[]) {

	if(nl >= 3){//Si hay más de tres capas(1 de entrada, + de 1 oculta y 1 de salida)

		nNumCapas = nl;//Insertamos el número de capas del Perceptrón.
		pCapas = new Capa[nl];//Generamos el vector de capas del Perceptrón.

		for(int i=0; i<nNumCapas; i++){
			
			pCapas[i].nNumNeuronas = npl[i];//Se rellena la variable que contiene el número de neuronas de cada capa del Perceptrón. 
			pCapas[i].pNeuronas = new Neurona[npl[i]];//Se reserva memoria para el vector que almacena las neuronas de cada capa.

			for(int j=0; j<npl[i]; j++){

				Neurona *nuevaNeurona = new Neurona;//Se crea una neurona nueva y a continuación se rellenará.
				pCapas[i].pNeuronas[j] = *nuevaNeurona;//Se añade la neurona recién creada dentro de la capa actual.

				if(i > 0){//Si nos encontramos en una capa distinta a la de entrada (ya que en esta las neuronas no reciben pesos), se rellenarán los pesos.
					
					//Se reserva memoria para cada peso de la neurona(teniendo en cuenta que hay un peso por cada neurona de la capa anterior y además el sesgo).
					pCapas[i].pNeuronas[j].w = new double[pCapas[i-1].nNumNeuronas+1];
					pCapas[i].pNeuronas[j].wCopia = new double[pCapas[i-1].nNumNeuronas+1];
					pCapas[i].pNeuronas[j].deltaW = new double[pCapas[i-1].nNumNeuronas+1];
					pCapas[i].pNeuronas[j].ultimoDeltaW = new double[pCapas[i-1].nNumNeuronas+1];

				}
			}
		}
		return 0;//Inicializa correctamente.
	}

	else{//Si el número de capas no es el adecuado.
		printf("ERROR en PerceptronMulticapa.cpp : [FUNCION inicializar]. ERROR, el número de capas ha de ser mayor igual a 3 capas (una capa de entrada, una oculta y una de salida). Saliendo de la función...\n");
		return 1;//No inicializa bien.
	}

}


// ------------------------------
// DESTRUCTOR: liberar memoria
PerceptronMulticapa::~PerceptronMulticapa() {
	liberarMemoria();
}


// ------------------------------
// Liberar memoria para las estructuras de datos
void PerceptronMulticapa::liberarMemoria() {
	for(int i=0; i<nNumCapas; i++){
		
		for(int j=0; j<pCapas[i].nNumNeuronas; j++){
			
			if(i!=0){//No es la capa de entrada
			
				//Borramos la memoria reservada en la función inicializar para los pesos.
				delete pCapas[i].pNeuronas[j].deltaW;
				delete pCapas[i].pNeuronas[j].ultimoDeltaW;
				delete pCapas[i].pNeuronas[j].w;
				delete pCapas[i].pNeuronas[j].wCopia;
			
			}
		
		}
		
		//Borramos la memoria reservada para el vector de neuronas.
		delete pCapas[i].pNeuronas;
			
	}
	//Borramos la memoria reservada para el vector de capas.
	delete pCapas;
}

// ------------------------------
// Rellenar todos los pesos (w) aleatoriamente entre -1 y 1
void PerceptronMulticapa::pesosAleatorios() {

	for(int i=1; i<nNumCapas; i++){//Se recorre a partir de la primera capa oculta porque a la capa de entrada no le llegan pesos de una capa anterior, ya que es la primera de todas.

		for(int j=0; j<pCapas[i].nNumNeuronas; j++){//Se recorren todas las neuronas de cada capa, a partir de la capa 1.

			for(int k=0; k<pCapas[i-1].nNumNeuronas+1; k++){//Se generan todos los pesos que le llegan a cada neurona, que serían el número de neuronas de la capa anterior (ya que le llega un peso de cada neurona de la capa anterior) y el sesgo.

				double w = ((double)rand()/RAND_MAX)* pow(-1,rand());//El peso que se genera está siempre entre -1 y +1.
				//cout << "Pesos aleatorios w= "<<w<<"\tCapa: "<<i<<"\tNeurona: "<<j<<"\tPeso: "<<k<<endl;
				
				pCapas[i].pNeuronas[j].w[k] = w;//Se almacena el peso aleatorio generado.
				
			}
		}
	}
}


// ------------------------------
// Alimentar las neuronas de entrada de la red con un patrón pasado como argumento
void PerceptronMulticapa::alimentarEntradas(double* input) {

	for(int i=0; i < pCapas[0].nNumNeuronas; i++){//Rercorremos todas las neuronas de la Capa 0 (capa de entrada) del Perceptrón.

		pCapas[0].pNeuronas[i].x = input[i];//Se rellena la salida que le llega a cada neurona de la Capa de Entrada del Perceptrón, (out).(Ver diapositiva 8 de la práctica).
		//cout<<"pCapas[0].pNeuronas["<<i<<"].x: "<<pCapas[0].pNeuronas[i].x<<endl;
	}

}

// ------------------------------
// Recoger los valores predichos por la red (out de la capa de salida) y almacenarlos en el vector pasado como argumento
void PerceptronMulticapa::recogerSalidas(double* output){

	for(int i=0; i < pCapas[nNumCapas-1].nNumNeuronas; i++){//Recorremos cada neurona de la Capa N-ésima (Capa de Salida) del Perceptrón.

		output[i] = pCapas[nNumCapas -1].pNeuronas[i].x;//Guardamos en el vector de salidas pasado por argumento, las salidas obtenidas de cada neurona.
		//cout<<"output["<<i<<"]: "<<output[i]<<endl;
	}

}

// ------------------------------
// Hacer una copia de todos los pesos (copiar w en copiaW)
void PerceptronMulticapa::copiarPesos() {
	
	for(int i=1; i<nNumCapas; i++){//Recorremos el vector de capas del Perceptrón empezando desde la capa oculta, ya que la capa de entrada no tiene pesos.
	
		for(int j=0; j<pCapas[i].nNumNeuronas; j++){//Recorremos las neuronas de cada capa una a una para copiar los pesos que tiene cada una de las neuronas.
	
			for(int k=0; k<pCapas[i-1].nNumNeuronas+1; k++){//Recorremos los pesos de cada neurona que son equivalentes en número a las neuronas de la capa anterior más el sesgo. 
	
				pCapas[i].pNeuronas[j].wCopia[k] = pCapas[i].pNeuronas[j].w[k];//Se copian los pesos en la variable de copia.
			}
		}
	}
}

// ------------------------------
// Restaurar una copia de todos los pesos (copiar copiaW en w)
void PerceptronMulticapa::restaurarPesos() {

	for(int i=1; i<nNumCapas; i++){//Se recorren las capas del Perceptrón, (excepto la de entrada).
	
		for(int j=0; j<pCapas[i].nNumNeuronas; j++){//Se recorren las neuronas de cada capa.
	
			for(int k=0; k<pCapas[i-1].nNumNeuronas+1; k++){//Se recorren los pesos de cada neurona (numNeuronas de la capa anterior + 1(el sesgo).
	
				pCapas[i].pNeuronas[j].w[k] = pCapas[i].pNeuronas[j].wCopia[k];//Se almacena en la variable de los pesos, los pesos guardados en la variable de copia.
	
			}
		}
	}
}

// ------------------------------
// Calcular y propagar las salidas de las neuronas, desde la primera capa hasta la última
void PerceptronMulticapa::propagarEntradas() {//Ver pseudocódigo diapositiva 33. También diapositivas [8-10] se explica gráficamente.
	
	for(int i=1; i<nNumCapas; i++){//Se recorren las todas las capas del Perceptrón una a una, (excepto la de entrada).
	
		for(int j=0; j<pCapas[i].nNumNeuronas; j++){//Se recorren todas las neuronas de cada capa recorrida.
	
			double sumaSigmoide = pCapas[i].pNeuronas[j].w[0];//Se inicializa con el sesgo 
	
			for(int k=0; k<pCapas[i-1].nNumNeuronas+1; k++){//Se recorren los pesos de cada neurona (numNeuronas de la capa anterior + 1(el sesgo).
	
			   sumaSigmoide += pCapas[i].pNeuronas[j].w[k+1] * pCapas[i-1].pNeuronas[k].x;//Se hace el Sumatorio de los pesos(excepto sesgo)*entrada de cada neurona.
			}
	
			pCapas[i].pNeuronas[j].x = 1/(1+exp(-1*sumaSigmoide));//Se calcula el resto de la sigmoide y se guarda en la entrada de cada neurona.	
		}
	}
}

// ------------------------------
// Calcular el error de salida (MSE) del out de la capa de salida con respecto a un vector objetivo y devolverlo
double PerceptronMulticapa::calcularErrorSalida(double* objetivo) {

	double mse = 0.0;
		
		for(int i=0; i<pCapas[nNumCapas-1].nNumNeuronas; i++){//Se recorren todas las neuronas de cada capa excepto la capa de salida.
		
			mse += pow(objetivo[i]-pCapas[nNumCapas-1].pNeuronas[i].x,2);//Fórmula del error de salida (MSE).(falta la división del sumatorio, se hace fuera del bucle).
		
		}
	//Se hace casting a 'double' para que siga siendo de tipo 'double' después de la división entre un entero.	
	mse = (double)mse / pCapas[nNumCapas-1].nNumNeuronas;//Por último el error obtenido se divide entre el número de neuronas de la capa de salida.
	
	return mse;
}


// ------------------------------
// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
void PerceptronMulticapa::retropropagarError(double* objetivo) {//Pseudocódigo en la diapositiva 34. Se ve más clara la ecuación.
	
	for(int i=0; i<pCapas[nNumCapas-1].nNumNeuronas; i++){//Se recorren todas las neuronas de la capa de salida.
    
    	double out = pCapas[nNumCapas-1].pNeuronas[i].x;
    	
    	pCapas[nNumCapas-1].pNeuronas[i].dX = -(objetivo[i] - out) * (1 - out) * out;//Se calcula la derivada de la entrada para cada neurona de la capa de salida.
    	
    }
    
    for(int i=nNumCapas-2; i>=0; i--){//Se recorren todas las neuronas de todas las capas ocultas.
    
    	for(int j=0; j<pCapas[i].nNumNeuronas; j++){//Se recorren todas las neuronas de capa actual dentro del bucle anterior.
    	
    		double sum=0.0;
    	
    		for(int k=0; k<pCapas[i+1].nNumNeuronas; k++){
    			
    			sum += pCapas[i+1].pNeuronas[k].dX * pCapas[i+1].pNeuronas[k].w[j+1];//Se calcula el sumatorio.
    	
    		}
    		double out = pCapas[i].pNeuronas[j].x;
    		pCapas[i].pNeuronas[j].dX = sum * out * (1-out);//Se le añade el producto al sumatorio para finalizar la ecuación.
    	}
    }
}

// ------------------------------
// Acumular los cambios producidos por un patrón en deltaW
void PerceptronMulticapa::acumularCambio() {//Pseudocódigo en la diapositiva 35.

	for(int i=1; i<nNumCapas; i++){//Recorremos todas las capas ocultas y la de salida del Perceptrón.
	
		for(int j=0; j<pCapas[i].nNumNeuronas; j++){//Recorremos cada neurona de las capas recorridas
    
    		for(int k=1; k<pCapas[i-1].nNumNeuronas+1; k++){//Recorremos las neuronas de la capa anterior.
    
    			pCapas[i].pNeuronas[j].ultimoDeltaW[k] = pCapas[i].pNeuronas[j].deltaW[k];
    			pCapas[i].pNeuronas[j].deltaW[k] += pCapas[i].pNeuronas[j].dX * pCapas[i-1].pNeuronas[k-1].x;
    		}
			
			pCapas[i].pNeuronas[j].ultimoDeltaW[0] = pCapas[i].pNeuronas[j].deltaW[0];
    		pCapas[i].pNeuronas[j].deltaW[0] += pCapas[i].pNeuronas[j].dX;
		}
	}

}

// ------------------------------
// Actualizar los pesos de la red, desde la primera capa hasta la última
void PerceptronMulticapa::ajustarPesos() {//Pseudocódigo en diapositiva 36.

	double eta = dEta;
	
	for(int i=1; i<nNumCapas; i++){
	
		for(int j=0; j<pCapas[i].nNumNeuronas; j++){
    
    		for(int k=1; k<pCapas[i-1].nNumNeuronas+1; k++){
    
    			pCapas[i].pNeuronas[j].w[k] = pCapas[i].pNeuronas[j].w[k] - eta * pCapas[i].pNeuronas[j].deltaW[k] - dMu * (eta * pCapas[i].pNeuronas[j].ultimoDeltaW[k]);
    
    		}
	
			pCapas[i].pNeuronas[j].w[0] = pCapas[i].pNeuronas[j].w[0] - eta * pCapas[i].pNeuronas[j].deltaW[0] - dMu * (eta * pCapas[i].pNeuronas[j].ultimoDeltaW[0]);    		
		
		}
		
		eta = pow(dDecremento,-(nNumCapas-i)) * eta;
	}
}

// ------------------------------
// Imprimir la red, es decir, todas las matrices de pesos
void PerceptronMulticapa::imprimirRed() {

	for(int i=1; i<nNumCapas; i++){
	
		std::cout<<"Capa "<<i<<std::endl<<"________________"<<std::endl;
	
		for(int j=0; j<pCapas[i].nNumNeuronas; j++){

    		for(int k=0; k<pCapas[i-1].nNumNeuronas+1; k++){

    			std::cout<<pCapas[i].pNeuronas[j].w[k]<<"\t";

    		}
    		std::cout<<std::endl;
		}
	}
}

// ------------------------------
// Simular la red: propagar las entradas hacia delante, retropropagar el error y ajustar los pesos
// entrada es el vector de entradas del patrón y objetivo es el vector de salidas deseadas del patrón
void PerceptronMulticapa::simularRedOnline(double* entrada, double* objetivo) {//Ver diapositiva 29.

	for(int i=1; i<this->nNumCapas; i++){
	
		for(int j=0; j<this->pCapas[i].nNumNeuronas; j++){
		
			for(int k=0; k<this->pCapas[i-1].nNumNeuronas+1; k++){
			
				this->pCapas[i].pNeuronas[j].deltaW[k] = 0.0;
			}
		}
	}
	
	alimentarEntradas(entrada);
	propagarEntradas(); 
	retropropagarError(objetivo); 
	acumularCambio(); 
	ajustarPesos();
}

// ------------------------------
// Leer una matriz de datos a partir de un nombre de fichero y devolverla
Datos* PerceptronMulticapa::leerDatos(const char *archivo) {
	
	std::ifstream fichero (archivo);
	Datos* datos_fichero=new Datos;
	
	fichero>>datos_fichero->nNumEntradas>>datos_fichero->nNumSalidas>>datos_fichero->nNumPatrones;

	datos_fichero->entradas = new double*[datos_fichero->nNumPatrones];

	for(int i=0; i<datos_fichero->nNumPatrones; i++)
		datos_fichero->entradas[i] = new double[datos_fichero->nNumEntradas];

	datos_fichero->salidas = new double*[datos_fichero->nNumPatrones];
	
	for(int i=0; i<datos_fichero->nNumPatrones; i++)
		datos_fichero->salidas[i] = new double[datos_fichero->nNumSalidas];

	for(int i=0; i<datos_fichero->nNumPatrones; i++){
	
		for(int j=0; j<datos_fichero->nNumEntradas; j++){
			
			fichero>>datos_fichero->entradas[i][j];
		}
		
		for(int k=0; k<datos_fichero->nNumSalidas; k++){
			
			fichero>>datos_fichero->salidas[i][k];
		}
	}
	
	return datos_fichero;
}

// ------------------------------
// Entrenar la red on-line para un determinado fichero de datos
void PerceptronMulticapa::entrenarOnline(Datos* pDatosTrain) {
	
	for(int i=0; i<pDatosTrain->nNumPatrones; i++){
		simularRedOnline(pDatosTrain->entradas[i], pDatosTrain->salidas[i]);
	}
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el error MSE cometido
double PerceptronMulticapa::test(Datos* pDatosTest) {

	double mse=0.0;

	for(int i=0; i<pDatosTest->nNumPatrones; i++){
	
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		mse += calcularErrorSalida(pDatosTest->salidas[i]);
	}

	mse /= pDatosTest->nNumPatrones;
	
	return mse;
}

// OPCIONAL - KAGGLE
// Imprime las salidas predichas para un conjunto de datos.
// Utiliza el formato de Kaggle: dos columnas (Id y Predicted)
void PerceptronMulticapa::predecir(Datos* pDatosTest){

	int i;
	int j;
	int numSalidas = pCapas[nNumCapas-1].nNumNeuronas;
	double * salidas = new double[numSalidas];
	
	cout << "Id,Predicted" << endl;
	
	for (i=0; i<pDatosTest->nNumPatrones; i++){

		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(salidas);
		
		cout << i;

		for (j=0; j<numSalidas; j++)
			cout << "," << salidas[j];
		
		cout << endl;

	}
}

// ------------------------------
// Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
// Una vez terminado, probar como funciona la red en pDatosTest
// Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
void PerceptronMulticapa::ejecutarAlgoritmoOnline(Datos * pDatosTrain, Datos * pDatosTest, int maxiter, double *errorTrain, double *errorTest)
{

	int countTrain = 0;

	// Inicialización de pesos
	pesosAleatorios();

	double minTrainError = 0.0;
	int numSinMejorar;
	double testError = 0.0;
	
	double validationError = 0.0;
	
	double minValidationError = 0.0;
	double numSinMejorarValidacion = 0.0;
	Datos *pDatosValidacion=NULL;

	// Generar datos de validación
	if(dValidacion > 0 && dValidacion < 1){
		
		int* vectordeelegidos=vectorAleatoriosEnterosSinRepeticion(0,pDatosTrain->nNumPatrones-1,round(pDatosTrain->nNumPatrones*dValidacion));
		
		pDatosValidacion=new Datos;
		pDatosValidacion->nNumPatrones = round(pDatosTrain->nNumPatrones*dValidacion);
		pDatosValidacion->nNumEntradas = pDatosTrain->nNumEntradas;
		pDatosValidacion->nNumSalidas = pDatosTrain->nNumSalidas;


		//Reservamos memoria para las entradas de test
		pDatosValidacion->entradas=new double*[pDatosValidacion->nNumPatrones];
		
		for(int i = 0; i < pDatosValidacion->nNumPatrones; i++)
			pDatosValidacion->entradas[i] = new double[pDatosValidacion->nNumEntradas];
		
		
		//Reservamos memoria para las salidas de test
		pDatosValidacion->salidas=new double*[pDatosValidacion->nNumPatrones];
		
		for(int i = 0; i < pDatosValidacion->nNumPatrones; i++)
			pDatosValidacion->salidas[i] = new double[pDatosValidacion->nNumSalidas];


		//Reservamos memoria para entrTrain
		double** entrTrain=new double*[pDatosTrain->nNumPatrones-pDatosValidacion->nNumPatrones];
		
		for(int i = 0; i<pDatosTrain->nNumPatrones-pDatosValidacion->nNumPatrones; i++)
			entrTrain[i] = new double[pDatosTrain->nNumEntradas];

		//Reservamos memoria para saliTrain
		double** saliTrain=new double*[pDatosTrain->nNumPatrones-pDatosValidacion->nNumPatrones];
		for(int i=0; i<pDatosTrain->nNumPatrones-pDatosValidacion->nNumPatrones; i++)
			saliTrain[i] = new double[pDatosTrain->nNumSalidas];

		sort(vectordeelegidos, vectordeelegidos+pDatosValidacion->nNumPatrones);

		for(int i=0,j=0,k=0; i<pDatosTrain->nNumPatrones; i++){
			
			if(i==vectordeelegidos[j]){
				pDatosValidacion->entradas[j] = pDatosTrain->entradas[i];
				pDatosValidacion->salidas[j] = pDatosTrain->salidas[i];
				j++;
			}
			else{
				entrTrain[k] = pDatosTrain->entradas[i];
				saliTrain[i] = pDatosTrain->salidas[i];
				k++;
			}
		}
			
		pDatosTrain->nNumPatrones = pDatosTrain->nNumPatrones - pDatosValidacion->nNumPatrones;
		pDatosTrain->salidas = saliTrain;
		pDatosTrain->entradas = entrTrain;
	}


	// Aprendizaje del algoritmo
	do {

		entrenarOnline(pDatosTrain);
		double trainError = test(pDatosTrain);//Calculamos MSE del entrenamiento.
		
		if(dValidacion > 0 && dValidacion < 1){
			validationError = test(pDatosValidacion);//Calculamos MSE para validación.
			
			if(countTrain==0 || validationError < minValidationError){
			
				minValidationError = validationError;
				numSinMejorarValidacion = 0;
			}
			else if( (validationError-minValidationError) < 0.00001){
				numSinMejorarValidacion = 0;
			}
			else{numSinMejorarValidacion++;}
		}
		
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copiarPesos();
			numSinMejorar = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			numSinMejorar = 0;
		else
			numSinMejorar++;

		if(numSinMejorar==50){
			cout << "Salida porque no mejora el entrenamiento!!"<< endl;
			restaurarPesos();
			countTrain = maxiter;
		}
		
		if(numSinMejorarValidacion>=50){
			cout << "Salida porque no mejora el error de validación!!"<< endl;
			restaurarPesos();
			countTrain = maxiter;
		}
		
		countTrain++;

		// Comprobar condiciones de parada de validación y forzar
		// OJO: en este caso debemos guardar el error de validación anterior, no el mínimo
		// Por lo demás, la forma en que se debe comprobar la condición de parada es similar
		// a la que se ha aplicado más arriba para el error de entrenamiento

		cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError << "\t Error de validación: " << validationError << endl;

	} while ( countTrain<maxiter );

	cout << "PESOS DE LA RED" << endl;
	cout << "===============" << endl;
	imprimirRed();

	cout << "Salida Esperada Vs Salida Obtenida (test)" << endl;
	cout << "=========================================" << endl;
	
	for(int i=0; i<pDatosTest->nNumPatrones; i++){
		
		double* prediccion = new double[pDatosTest->nNumSalidas];

		// Cargamos las entradas y propagamos el valor
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(prediccion);
		
		for(int j=0; j<pDatosTest->nNumSalidas; j++)
			cout << pDatosTest->salidas[i][j] << " -- " << prediccion[j] << " ";
		
		cout << endl;
		delete[] prediccion;

	}

	testError = test(pDatosTest);
	*errorTest=testError;
	*errorTrain=minTrainError;

}


// OPCIONAL - KAGGLE
//Guardar los pesos del modelo en un fichero de texto.
bool PerceptronMulticapa::guardarPesos(const char * archivo)
{
	// Objeto de escritura de fichero
	ofstream f(archivo);

	if(!f.is_open())
		return false;

	// Escribir el numero de capas y el numero de neuronas en cada capa en la primera linea.
	f << nNumCapas;

	for(int i = 0; i < nNumCapas; i++)
		f << " " << pCapas[i].nNumNeuronas;
	f << endl;

	// Escribir los pesos de cada capa
	for(int i = 1; i < nNumCapas; i++)
		for(int j = 0; j < pCapas[i].nNumNeuronas; j++)
			for(int k = 0; k < pCapas[i-1].nNumNeuronas + 1; k++)
				f << pCapas[i].pNeuronas[j].w[k] << " ";

	f.close();

	return true;

}

// OPCIONAL - KAGGLE
//Cargar los pesos del modelo desde un fichero de texto.
bool PerceptronMulticapa::cargarPesos(const char * archivo)
{
	// Objeto de lectura de fichero
	ifstream f(archivo);

	if(!f.is_open())
		return false;

	// Número de capas y de neuronas por capa.
	int nl;
	int *npl;

	// Leer número de capas.
	f >> nl;

	npl = new int[nl];

	// Leer número de neuronas en cada capa.
	for(int i = 0; i < nl; i++)
		f >> npl[i];

	// Inicializar vectores y demás valores.
	inicializar(nl, npl);

	// Leer pesos.
	for(int i = 1; i < nNumCapas; i++)
		for(int j = 0; j < pCapas[i].nNumNeuronas; j++)
			for(int k = 0; k < pCapas[i-1].nNumNeuronas + 1; k++)
				f >> pCapas[i].pNeuronas[j].w[k];

	f.close();
	delete[] npl;

	return true;
}