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


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// CONSTRUCTOR: Dar valor por defecto a todos los parámetros
PerceptronMulticapa::PerceptronMulticapa(){

	dEta=0.1;//Ponemos como valor por defecto para la tasa de aprendizaje un valor relativamente pequeño.
	dMu=0.9; //Empezamos poniendo un momento (inercia) incial, relativamente grande, para que no se estanque al principio de la ejecución y puedo explorar bien.
	//dValidacion=0.0; //El conjunto de datos es todo para entrenamiento, por defecto.
	//dDecremento=1; //El factor de decremento por capas para la tasa de aprendizaje(dEta).
}

// ------------------------------
// Reservar memoria para las estructuras de datos
int PerceptronMulticapa::inicializar(int nl, int npl[]) {

	if(nl=>3){//Si hay más de tres capas(1 de entrada, + de 1 oculta y 1 de salida)

		nNumCapas=nl;//Insertamos el número de capas del Perceptrón.
		pCapas=new Capa[nl];//Generamos el vector de capas del Perceptrón.

		for(int i=0;i<nNumCapas;i++){

			Capa *nuevaCapa=new Capa;//Se crea una Capa nueva y a continuación se relenará.

			nuevaCapa->nNumNeuronas=npl[i];//Se añade el número de neuronas a cada capa.
			nuevaCapa->pNeuronas=new Neurona[npl[i]];//Se crea el vector con las neuronas de cada capa.

			for(int j=0;j<npl[i];j++){

				Neurona *nuevaNeurona= new Neurona;//Se crea una neurona nueva y a continuación se rellenará.
				nuevaCapa->pNeuronas[j]=*nuevaNeurona;//Se añade la neurona nueva dentro de la capa actual.

				if(i>0){//Si nos encontramos en una capa distinta a la de entrada (ya que en esta las neuronas no reciben pesos), se rellenarán los pesos.

					nuevaCapa->pNeuronas[j].w = new double[pCapas[i-1].nNumNeuronas+1];//Se genera memoria para cada peso de la neurona (teniendo en cuenta que hay un peso por cada neurona de la capa anterior y además el sesgo).
					nuevaCapa->pNeuronas[j].wCopia = new double[pCapas[i-1].nNumNeuronas+1];//Lo mismo que en línea 56.
					nuevaCapa->pNeuronas[j].deltaW = new double[pCapas[i-1].nNumNeuronas+1];//Lo mismo que en línea 56.
					nuevaCapa->pNeuronas[j].ultimoDeltaW = new double[pCapas[i-1].nNumNeuronas+1];//Lo mismo que en línea 56.

				}
			}

			pCapas[i]=*nuevaCapa;//Por último, se añade la capa generada en cada iteración al vector de capas que tiene el Perceptrón.

		}

		return 0;//Tras haber inicializado el perceptrón con éxtio, salimos de la función.
	}

	else{//Si el número de capas no es el adecuado.
		printf("ERROR en PerceptronMulticapa.cpp : [FUNCION inicializar]. ERROR, el número de capas ha de ser mayor igual a 3 capas (una capa de entrada, una oculta y una de salida). Saliendo de la función...\n")
		return 1;
	}



	/* HECHO CON MALLOC

//Se reserva la memoria para las capas
nNumCapas=nl;
pCapas=(Capa *)malloc(sizeof(Capa)*nl);

for(int i=0;i<nNumCapas;i++){
//Se reserva memoria para las neuronas
pCapas[i].pNeuronas=(Neurona *)malloc(npl[i]*sizeof(Neurona));
pCapas[i].nNumNeuronas=npl[i];

for(int j=0;j<pCapas[i].nNumNeuronas;j++){
pCapas[i].pNeuronas[j].x=1;
pCapas[i].pNeuronas[j].dX=1;

if(i!=0){ //No es la capa de entrada
int nEntradas=pCapas[i-1].nNumNeuronas+1;

//Se reserva memoria para los parametros de las neuronas
pCapas[i].pNeuronas[j].w=(double *)malloc(sizeof(double)*nEntradas);
pCapas[i].pNeuronas[j].deltaW=(double *)malloc(sizeof(double)*nEntradas);
pCapas[i].pNeuronas[j].ultimoDeltaW=(double *)malloc(sizeof(double)*nEntradas);
pCapas[i].pNeuronas[j].wCopia=(double *)malloc(sizeof(double)*nEntradas);

for(int k=0;k<nEntradas;k++){
//Se inicializan los parametros de las neuronas
pCapas[i].pNeuronas[j].w[k]=0.0;
pCapas[i].pNeuronas[j].deltaW[k]=0.0;
pCapas[i].pNeuronas[j].ultimoDeltaW[k]=0.0;
pCapas[i].pNeuronas[j].wCopia[k]=0.0;
}
}

else{ //Es la capa de entrada
//Se inicializan los parametros de las neuronas
pCapas[i].pNeuronas[j].w=NULL;
pCapas[i].pNeuronas[j].deltaW=NULL;
pCapas[i].pNeuronas[j].ultimoDeltaW=NULL;
pCapas[i].pNeuronas[j].wCopia=NULL;
}
}
}

return 1;
}
*/

}


// ------------------------------
// DESTRUCTOR: liberar memoria
PerceptronMulticapa::~PerceptronMulticapa() {
	liberarMemoria();
}


// ------------------------------
// Liberar memoria para las estructuras de datos
void PerceptronMulticapa::liberarMemoria() {
	for(int i=0;i<nNumCapas;i++){

		for(int j=0;j<pCapas[i].nNumNeuronas;j++){

			delete[] pCapas[i].pNeuronas[j].w;

		}

		delete[] pCapas[i].pNeuronas;

	}

	delete[] pCapas;
}

// ------------------------------
// Rellenar todos los pesos (w) aleatoriamente entre -1 y 1
void PerceptronMulticapa::pesosAleatorios() {

	for(int i=1;i<nNumCapas;i++){//Se recorre a partir de la primera capa oculta porque a la capa de entrada no le llegan pesos de una capa anterior, ya que es la primera de todas.

		for(int j=0;j<pCapas[i].nNumNeuronas;j++){//Se recorren todas las neuronas de cada capa, a partir de la capa 1.

			for(int k=0;k<pCapas[i-1].nNumNeuronas+1;k++){//Se generan todos los pesos que le llegan a cada neurona, que serían el número de neuronas de la capa anterior (ya que le llega un peso de cada neurona de la capa anterior) y el sesgo.

				int w= (rand()%3)-1;//El peso que se genera está siempre entre -1 y +1.

				pCapas[i].pNeuronas[j].w[k] = w;//Se almacena el peso aleatorio generado.
				pCapas[i].pNeuronas[j].wCopia[k] = pCapas[i].pNeuronas[j].w[k];//Se almacena también el vector_copia de los pesos.

			}
		}
	}

}

// ------------------------------
// Alimentar las neuronas de entrada de la red con un patrón pasado como argumento
void PerceptronMulticapa::alimentarEntradas(double* input) {

	for(int i=0; i < pCapas[0].nNumNeuronas; i++){//Rercorremos todas las neuronas de la Capa 0 (capa de entrada) del Perceptrón.

		pCapas[0].pNeuronas->x=input[i];//Se rellena la salida que le llega a cada neurona de la Capa de Entrada del Perceptrón, (out).(Ver diapositiva 8 de la práctica).

	}

}

// ------------------------------
// Recoger los valores predichos por la red (out de la capa de salida) y almacenarlos en el vector pasado como argumento
void PerceptronMulticapa::recogerSalidas(double* output){

	for(int i=0; i < pCapas[nNumCapas-1].nNumNeuronas;i++){//Recorremos cada neurona de la Capa N-ésima (Capa de Salida) del Perceptrón.

		output[i]=pCapas[0].pNeuronas->x;//Guardamos en el vector de salidas pasado por argumento las salidas obtenidas de cada nuerona.
	}

}

// ------------------------------
// Hacer una copia de todos los pesos (copiar w en copiaW)
void PerceptronMulticapa::copiarPesos() {

}

// ------------------------------
// Restaurar una copia de todos los pesos (copiar copiaW en w)
void PerceptronMulticapa::restaurarPesos() {

}

// ------------------------------
// Calcular y propagar las salidas de las neuronas, desde la primera capa hasta la última
void PerceptronMulticapa::propagarEntradas() {
	
}

// ------------------------------
// Calcular el error de salida (MSE) del out de la capa de salida con respecto a un vector objetivo y devolverlo
double PerceptronMulticapa::calcularErrorSalida(double* target) {
	return -1;
}


// ------------------------------
// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
void PerceptronMulticapa::retropropagarError(double* objetivo) {
	
}

// ------------------------------
// Acumular los cambios producidos por un patrón en deltaW
void PerceptronMulticapa::acumularCambio() {

}

// ------------------------------
// Actualizar los pesos de la red, desde la primera capa hasta la última
void PerceptronMulticapa::ajustarPesos() {

}

// ------------------------------
// Imprimir la red, es decir, todas las matrices de pesos
void PerceptronMulticapa::imprimirRed() {

}

// ------------------------------
// Simular la red: propagar las entradas hacia delante, retropropagar el error y ajustar los pesos
// entrada es el vector de entradas del patrón y objetivo es el vector de salidas deseadas del patrón
void PerceptronMulticapa::simularRedOnline(double* entrada, double* objetivo) {

}

// ------------------------------
// Leer una matriz de datos a partir de un nombre de fichero y devolverla
Datos* PerceptronMulticapa::leerDatos(const char *archivo) {


	return NULL;
}

// ------------------------------
// Entrenar la red on-line para un determinado fichero de datos
void PerceptronMulticapa::entrenarOnline(Datos* pDatosTrain) {
	int i;
	for(i=0; i<pDatosTrain->nNumPatrones; i++){
		simularRedOnline(pDatosTrain->entradas[i], pDatosTrain->salidas[i]);
	}
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el error MSE cometido
double PerceptronMulticapa::test(Datos* pDatosTest) {
	return -1.0;
}

// OPCIONAL - KAGGLE
// Imprime las salidas predichas para un conjunto de datos.
// Utiliza el formato de Kaggle: dos columnas (Id y Predicted)
void PerceptronMulticapa::predecir(Datos* pDatosTest)
{
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

		for (j = 0; j < numSalidas; j++)
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

	double minTrainError = 0;
	int numSinMejorar;
	double testError = 0;

	double validationError;

	// Generar datos de validación
	if(dValidacion > 0 && dValidacion < 1){
		// .......
	}


	// Aprendizaje del algoritmo
	do {

		entrenarOnline(pDatosTrain);
		double trainError = test(pDatosTrain);
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
