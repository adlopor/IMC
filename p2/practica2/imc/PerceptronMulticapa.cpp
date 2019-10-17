/*********************************************************************
 * File  : PerceptronMulticapa.cpp
 * Date  : 2018
 *********************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <limits>
#include <math.h>

#include "PerceptronMulticapa.h"
#include "util.h"

#include <algorithm>

using namespace imc;
using namespace std;
using namespace util;


// ------------------------------
// CONSTRUCTOR: Dar valor por defecto a todos los parámetros (dEta, dMu, dValidacion y dDecremento)
PerceptronMulticapa::PerceptronMulticapa(){

	//Variables privadas:
	pCapas = NULL;
	nNumCapas = 3;
	nNumPatronesTrain = 0;
	//Variables públicas:
	dEta = 0.7;//Ponemos como valor por defecto para la tasa de aprendizaje un valor relativamente pequeño.
	dMu = 1; //Empezamos poniendo un momento (inercia) incial, relativamente grande, para que no se estanque al principio de la ejecución y puedo explorar bien.
	bOnline = false; //El aprendizaje será off-line por defecto.
	dValidacion = 0; //El conjunto de datos es todo para entrenamiento, por defecto.
	dDecremento = 1; //El factor de decremento por capas para la tasa de aprendizaje(dEta).
	 
}

// ------------------------------
// Reservar memoria para las estructuras de datos
// nl tiene el numero de capas y npl es un vector que contiene el número de neuronas por cada una de las capas
// tipo contiene el tipo de cada capa (0 => sigmoide, 1 => softmax)
// Rellenar vector Capa* pCapas
int PerceptronMulticapa::inicializar(int nl, int npl[], int tipo[]) {
	
	if(nl >= 3){//Si hay más de tres capas(1 de entrada, + de 1 oculta y 1 de salida)

		nNumCapas = nl;//Insertamos el número de capas del Perceptrón.
		pCapas = new Capa[nNumCapas];//Generamos el vector de capas del Perceptrón.

		for(int i=0; i<nNumCapas; i++){
			
			pCapas[i].nNumNeuronas = npl[i];//Se rellena la variable que contiene el número de neuronas de cada capa del Perceptrón. 
			pCapas[i].pNeuronas = new Neurona[npl[i]];//Se reserva memoria para el vector que almacena las neuronas de cada capa.
			pCapas[0].tipo = tipo[0];

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
void PerceptronMulticapa::propagarEntradas() {

	//Calculamos la salida de todos los nodos (neuronas), que sean de tipo Sigmoide.
	for(int i=1; i<nNumCapas; i++){//Se recorren las todas las capas del Perceptrón una a una, (excepto la de entrada).
		if(pCapas[i].tipo==0){
		
			for(int j=0; j<pCapas[i].nNumNeuronas; j++){//Se recorren todas las neuronas de cada capa recorrida.
	
				double sumaSigmoide = pCapas[i].pNeuronas[j].w[0];//Se inicializa con el sesgo 
	
				for(int k=0; k<pCapas[i-1].nNumNeuronas+1; k++){//Se recorren los pesos de cada neurona (numNeuronas de la capa anterior + 1(el sesgo).
	
			   		sumaSigmoide += pCapas[i].pNeuronas[j].w[k+1] * pCapas[i-1].pNeuronas[k].x;//Se hace el Sumatorio de los pesos(excepto sesgo)*entrada de cada neurona.
				}
	
				pCapas[i].pNeuronas[j].x = 1/(1+exp(-1*sumaSigmoide));//Se calcula el resto de la sigmoide y se guarda en la entrada de cada neurona.	
			}
		}
		else{
		
			double sumaSoftMax = 0;
			
			for(int j=0; j<pCapas[i].nNumNeuronas; j++){//Se recorren todas las neuronas de cada capa recorrida.
			
				double sumaSigmoide = pCapas[i].pNeuronas[j].w[0];//Se inicializa con el sesgo
				
				for(int k=0;k<pCapas[i-1].nNumNeuronas+1;k++){//Se recorren los pesos de cada neurona (numNeuronas de la capa anterior + 1(el sesgo).
				
				   sumaSigmoide += pCapas[i].pNeuronas[j].w[k+1]*pCapas[i-1].pNeuronas[k].x;//Se hace el Sumatorio de los pesos(excepto sesgo)*entrada de cada neurona.
				
				}
				
				pCapas[i].pNeuronas[j].x = exp(sumaSigmoide);
				
				sumaSoftMax += pCapas[i].pNeuronas[j].x;
			}
			
			for(int j=0; j<pCapas[i].nNumNeuronas; j++)
				pCapas[i].pNeuronas[j].x = pCapas[i].pNeuronas[j].x / sumaSoftMax;
			
		}
	}
	
}

// ------------------------------
// Calcular el error de salida del out de la capa de salida con respecto a un vector objetivo y devolverlo
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double PerceptronMulticapa::calcularErrorSalida(double* objetivo, int funcionError) {

	double error=0.0;
	
	if(funcionError == 0){//La función de error será la entropía cruzada.
		for(int i=0; i<pCapas[nNumCapas-1]. nNumNeuronas; i++)
			error += pow(objetivo[i]-pCapas[nNumCapas-1].pNeuronas[i].x,2);
			
		error = (double)error / pCapas[nNumCapas-1].nNumNeuronas;
	}
	else{//La función de error será el MSE.
		for(int i=0; i<pCapas[nNumCapas-1].nNumNeuronas; i++)		
			error += -1 * log(pCapas[nNumCapas-1].pNeuronas[i].x) * objetivo[i];
		
		error = (double)error/pCapas[nNumCapas-1].nNumNeuronas;
	}
	return error;
}


// ------------------------------
// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::retropropagarError(double* objetivo, int funcionError) {//Ver diferencia en las funciones de error respecto a si es SotfMax o Sigmoide. 

	if(pCapas[nNumCapas-1].tipo == 0){//Comprobamos que el tipo de nodo de la capa de salida sea Sigmoide.
	
		if(funcionError == 0){//Si la función de error es MSE
	
			for(int i=0; i<pCapas[nNumCapas-1].nNumNeuronas; i++){//Recorremos todas la neuronas de la capa de salida.
				
				double out = pCapas[nNumCapas-1].pNeuronas[i].x;
				pCapas[nNumCapas-1].pNeuronas[i].dX = -1 * (objetivo[i] - out) * (1 - out) * out;
			}
		}
		
		else{//Si la funcion de error es EntropiaCruzada.
			for(int i=0; i<pCapas[nNumCapas-1].nNumNeuronas; i++){
				
					double out = pCapas[nNumCapas-1].pNeuronas[i].x;
					pCapas[nNumCapas-1].pNeuronas[i].dX = -1 * (objetivo[i] / out) * (1 - out) * out;
				}
			}
	}
	else{//Si el tipo de nodo de la capa de salida es SoftMax.
	
			if(funcionError == 0){//Y la funcion error es MSE.
			
				for(int j=0; j<pCapas[nNumCapas-1].nNumNeuronas; j++){
					
					double sumaError=0.0;
					
					for(int i=0; i<pCapas[nNumCapas-1].nNumNeuronas; i++){
						
						double outj = pCapas[nNumCapas-1].pNeuronas[j].x;
						double outi = pCapas[nNumCapas-1].pNeuronas[i].x;
						
						if(i!=j){
							sumaError += -1 * (objetivo[i] - outi) * (-outi) * outj;
						}
						else{
							sumaError += -1 * (objetivo[i] - outj) * (1 - outj) * outj;
						}
					}
					
					pCapas[nNumCapas-1].pNeuronas[j].dX = sumaError;
				
				}
			}
			else{//Si la funcion de error es EntropiaCruzada.
			
				for(int j=0; j<pCapas[nNumCapas-1].nNumNeuronas; j++){
				
					double sumaError=0.0;
				
					for(int i=0; i<pCapas[nNumCapas-1].nNumNeuronas; i++){
					
						double outj = pCapas[nNumCapas-1].pNeuronas[j].x;
						double outi = pCapas[nNumCapas-1].pNeuronas[i].x;
						
						if(i!=j){
							sumaError += -1 * (objetivo[i] / outi) * (-outi) * outj;
						}
						else{
							sumaError += -1 *(objetivo[i] / outj) * (1 - outj) * outj;
						}
					}
					
					pCapas[nNumCapas-1].pNeuronas[j].dX = sumaError;
				}
			}
	}
    
    for(int j=nNumCapas-2; j>=1; j--){//retropropagamos el error a partir de la última capa oculta hacia atrás.
    
    	for(int k=0; k<pCapas[j].nNumNeuronas; k++){
    	
    		double suma = 0.0;
    		
    		for(int l=0;l<pCapas[j+1].nNumNeuronas;l++){
    		
    			suma += pCapas[j+1].pNeuronas[l].dX*pCapas[j+1].pNeuronas[l].w[k+1];
    		}
    		
    		double out = pCapas[j].pNeuronas[k].x;
    		pCapas[j].pNeuronas[k].dX = suma * out * (1 - out);
    	}
    }
}

// ------------------------------
// Acumular los cambios producidos por un patrón en deltaW
void PerceptronMulticapa::acumularCambio() {

	for(int i=1; i<nNumCapas; i++){
	
		for(int j=0; j<pCapas[i].nNumNeuronas; j++){
    		for(int k=1; k<pCapas[i-1].nNumNeuronas+1; k++){
    		
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
void PerceptronMulticapa::ajustarPesos() {

	double eta = dEta;
	
	for(int i=1; i<nNumCapas; i++){
	
		for(int j=0; j<pCapas[i].nNumNeuronas; j++){
		
    		for(int k=1; k<pCapas[i-1].nNumNeuronas+1; k++){
    		
    			if(!bOnline){
    				pCapas[i].pNeuronas[j].w[k]=pCapas[i].pNeuronas[j].w[k]-((eta*pCapas[i].pNeuronas[j].deltaW[k])/nNumPatronesTrain)-((dMu*(eta*pCapas[i].pNeuronas[j].ultimoDeltaW[k]))/nNumPatronesTrain);
    			}
    			else{
    				pCapas[i].pNeuronas[j].w[k]=pCapas[i].pNeuronas[j].w[k]-(eta*pCapas[i].pNeuronas[j].deltaW[k])-(dMu*(eta*pCapas[i].pNeuronas[j].ultimoDeltaW[k]));
				}
    		}
    		
			if(!bOnline){
				pCapas[i].pNeuronas[j].w[0]=pCapas[i].pNeuronas[j].w[0]-((eta*pCapas[i].pNeuronas[j].deltaW[0])/nNumPatronesTrain)-((dMu*(eta*pCapas[i].pNeuronas[j].ultimoDeltaW[0]))/nNumPatronesTrain);
			}
			else{
				pCapas[i].pNeuronas[j].w[0]=pCapas[i].pNeuronas[j].w[0]-(eta*pCapas[i].pNeuronas[j].deltaW[0])-(dMu*(eta*pCapas[i].pNeuronas[j].ultimoDeltaW[0]));
			}
		}
		eta = pow(dDecremento,-(nNumCapas - i)) * eta;
	}
}

// ------------------------------
// Imprimir la red, es decir, todas las matrices de pesos
void PerceptronMulticapa::imprimirRed() {

	for(int i=1; i<nNumCapas; i++){
	
		cout<<"Capa "<<i<<endl<<"========"<<endl;
		
		for(int j=0; j<pCapas[i].nNumNeuronas; j++){
		
    		for(int k=0; k<pCapas[i-1].nNumNeuronas+1; k++){
    		
    			cout<<pCapas[i].pNeuronas[j].w[k]<<"\t";
    		}
    		cout<<endl;
		}
	}
}

// ------------------------------
// Simular la red: propragar las entradas hacia delante, computar el error, retropropagar el error y ajustar los pesos
// entrada es el vector de entradas del patrón, objetivo es el vector de salidas deseadas del patrón.
// El paso de ajustar pesos solo deberá hacerse si el algoritmo es on-line
// Si no lo es, el ajuste de pesos hay que hacerlo en la función "entrenar"
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::simularRed(double* entrada, double* objetivo, int funcionError) {//Ver diapositivas XX práctica 1.
	
	alimentarEntradas(entrada);
	propagarEntradas();
	retropropagarError(objetivo,funcionError);
	acumularCambio();
	if(bOnline){
		ajustarPesos();
	}
}

// ------------------------------
// Leer una matriz de datos a partir de un nombre de fichero y devolverla
Datos* PerceptronMulticapa::leerDatos(const char *archivo) {
	
	Datos * pDatos = new Datos;
	ifstream fichero (archivo);
	
	fichero>>pDatos->nNumEntradas>>pDatos->nNumSalidas>>pDatos->nNumPatrones;

	pDatos->entradas=new double*[pDatos->nNumPatrones];
	
	for(int i=0; i<pDatos->nNumPatrones; i++)
		pDatos->entradas[i] = new double[pDatos->nNumEntradas];
		
	pDatos->salidas=new double*[pDatos->nNumPatrones];
	
	for(int i=0; i<pDatos->nNumPatrones; i++)
		pDatos->salidas[i] = new double[pDatos->nNumSalidas];

	for(int i=0; i<pDatos->nNumPatrones; i++){
		for(int j=0;j<pDatos->nNumEntradas;j++){
			
			fichero>>pDatos->entradas[i][j];
		}
	
		for(int k=0; k<pDatos->nNumSalidas; k++)
			fichero>>pDatos->salidas[i][k];

	}

	return pDatos;
}


// ------------------------------
// Entrenar la red para un determinado fichero de datos (pasar una vez por todos los patrones)
void PerceptronMulticapa::entrenar(Datos* pDatosTrain, int funcionError) {

	if(bOnline){//Si se entrena on-line
		for(int r=0; r<pDatosTrain->nNumPatrones; r++){
			for(int i=1; i<this->nNumCapas; i++){
				for(int j=0; j this->pCapas[i].nNumNeuronas; j++){
					for(int k=0; k<this->pCapas[i-1].nNumNeuronas+1; k++){
			
						this->pCapas[i].pNeuronas[j].deltaW[k] = 0.0;
					}
				}
			}
			simularRed(pDatosTrain->entradas[r],pDatosTrain->salidas[r],funcionError);
		}
	}
	else{//Si se entrena off-line
		for(int i=1; i<this->nNumCapas; i++){
			for(int j=0; j<this->pCapas[i].nNumNeuronas; j++){
				for(int k=0; k<this->pCapas[i-1].nNumNeuronas+1; k++){
			
					this->pCapas[i].pNeuronas[j].deltaW[k] = 0.0;
				}
			}
		}
		for(int r=0; r<pDatosTrain->nNumPatrones; r++)
			simularRed(pDatosTrain->entradas[r],pDatosTrain->salidas[r],funcionError);
		
		ajustarPesos();

	}
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el error cometido
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double PerceptronMulticapa::test(Datos* pDatosTest, int funcionError) {
	
	double mse=0.0;
	for(int i=0; i<pDatosTest->nNumPatrones; i++){

		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		mse += calcularErrorSalida(pDatosTest->salidas[i],funcionError);
	}
	
	return (mse / pDatosTest->nNumPatrones);
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

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;
		
		cout << i << "," << maxIndex << endl;

	}
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el CCR
double PerceptronMulticapa::testClassification(Datos* pDatosTest) {

	double ccr = 0.0;
	
	int numSalidas = pCapas[nNumCapas-1].nNumNeuronas;

	if(matrizConf != NULL){
	
		for(int i=0; i<numSalidas; i++)
			for(int k=0; k<numSalidas; k++)
				matrizConf[i][k] = 0;
	}
	
	double * salidas = new double[numSalidas];
	for(int i=0; i<pDatosTest->nNumPatrones; i++){
		
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(salidas);

		double mayor = salidas[0];
		int indiceMayor = 0;
		
		for(int j=1; j<numSalidas; j++){
			
			if(mayor < salidas[j]){
				mayor = salidas[j];
				indiceMayor = j;
			}
		}
		if(pDatosTest->salidas[i][indiceMayor] == 1){
			ccr++;
		}
		if(matrizConf != NULL){
			
			int j=0;
			for(int k=0; k<numSalidas; k++){
				if(pDatosTest->salidas[i][k] == 1)
					j=k;
			}
			matrizConf[indiceMayor][j]++;
		}
	}
	
	return ((ccr / pDatosTest->nNumPatrones) * 100);
}

// ------------------------------
// Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
// Una vez terminado, probar como funciona la red en pDatosTest
// Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::ejecutarAlgoritmo(Datos * pDatosTrain, Datos * pDatosTest, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int funcionError)
{
	int countTrain = 0;

	// Inicialización de pesos
	pesosAleatorios();

	double minTrainError = 0;
	int numSinMejorar = 0;
	double testError = 0;
	nNumPatronesTrain = pDatosTrain->nNumPatrones;

	Datos * pDatosValidacion = NULL;
	double validationError = 0, previousValidationError = 0;
	int numSinMejorarValidacion = 0;

	// Generar datos de validación
	if(dValidacion > 0 && dValidacion < 1){

	}

	// Aprendizaje del algoritmo
	do {

		entrenar(pDatosTrain,funcionError);

		double trainError = test(pDatosTrain,funcionError);
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

		testError = test(pDatosTest,funcionError);
		countTrain++;

		// Comprobar condiciones de parada de validación y forzar

		cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError << "\t Error de test: " << testError << "\t Error de validacion: " << validationError << endl;

	} while ( countTrain<maxiter );

	if ( (numSinMejorarValidacion!=50) && (numSinMejorar!=50))
		restaurarPesos();

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
			cout << pDatosTest->salidas[i][j] << " -- " << prediccion[j] << " \\\\ " ;
		cout << endl;
		delete[] prediccion;

	}

	*errorTest=test(pDatosTest,funcionError);;
	*errorTrain=minTrainError;
	*ccrTest = testClassification(pDatosTest);
	*ccrTrain = testClassification(pDatosTrain);

}

// OPCIONAL - KAGGLE
//Guardar los pesos del modelo en un fichero de texto.
bool PerceptronMulticapa::guardarPesos(const char * archivo)
{
	// Objeto de escritura de fichero
	ofstream f(archivo);

	if(!f.is_open())
		return false;

	// Escribir el numero de capas, el numero de neuronas en cada capa y el tipo de capa en la primera linea.
	f << nNumCapas;

	for(int i = 0; i < nNumCapas; i++)
	{
		f << " " << pCapas[i].nNumNeuronas;
		f << " " << pCapas[i].tipo;
	}
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
	int *tipos;

	// Leer número de capas.
	f >> nl;

	npl = new int[nl];
	tipos = new int[nl];

	// Leer número de neuronas en cada capa y tipo de capa.
	for(int i = 0; i < nl; i++)
	{
		f >> npl[i];
		f >> tipos[i];
	}

	// Inicializar vectores y demás valores.
	inicializar(nl, npl, tipos);

	// Leer pesos.
	for(int i = 1; i < nNumCapas; i++)
		for(int j = 0; j < pCapas[i].nNumNeuronas; j++)
			for(int k = 0; k < pCapas[i-1].nNumNeuronas + 1; k++)
				f >> pCapas[i].pNeuronas[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
