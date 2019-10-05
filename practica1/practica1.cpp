//============================================================================
// Introducción a los Modelos Computacionales
// Name        : practica1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // Para coger la hora time()
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <string.h>
#include <math.h>
#include "imc/PerceptronMulticapa.h"


using namespace imc;
using namespace std;

int main(int argc, char **argv) {

	//Procesar los argumentos de la línea de comandos
	bool tflag=false,Tflag=false, iflag=false, lflag=false, hflag=false, eflag=false, mflag=false, vflag=false, dflag=false, wflag=false, pflag=false;
	char *tvalue=NULL,* Tvalue=NULL,* ivalue,* lvalue,* hvalue,* evalue,* mvalue,* vvalue,* dvalue,* wvalue=NULL;
	int c;

	opterr=0;

	//a: opción que requiere un argumento
	//a:: el argumento requerido es opcional
	
	while((c=getopt(argc,argv,"t:T:i:l:h:e:m:v:d:w:p"))!=-1){
	
		//Se han añadido los parámetros necesarios para usar el modo opcional de predicción (kaggle).
		//Añadir el resto de parámetros que sean necesarios para la parte básica de las prácticas.
		switch(c){
	
			case 't':
				tflag=true;
				tvalue=optarg;

				break;

			case 'T':
				Tflag=true;
				Tvalue=optarg;

				break;

			case 'i':
				iflag=true;
				ivalue=optarg;

				break;

			case 'l':
				lflag=true;
				lvalue=optarg;

				break;

			case 'h':
				hflag=true;
				hvalue=optarg;

				break;

			case 'e':
				eflag=true;
				evalue=optarg;

				break;

			case 'm':
				mflag=true;
				mvalue=optarg;

				break;

			case 'v':
				vflag=true;
				vvalue=optarg;

				break;

			case 'd':
				dflag=true;
				dvalue=optarg;

			case 'w':
				wflag=true;
				wvalue=optarg;

				break;

			case 'p':
				pflag=true;

				break;

			case '?':
				
				if(optopt=='T' || optopt=='w' || optopt=='p' || optopt=='t' || optopt=='i' || optopt=='l' || optopt=='h' || optopt=='e' || optopt=='m' || optopt=='v' || optopt=='d'){
					fprintf(stderr,"La opción -%c requiere un argumento.\n",optopt);
				}

				else if(isprint(optopt)){
					fprintf(stderr,"Opción desconocida `-%c'.\n",optopt);
				}

				else{
					fprintf(stderr,"Caracter de opción desconocido `\\x%x'.\n",optopt);
				}

				return EXIT_FAILURE;

			default:
				return EXIT_FAILURE;
		}
	}

	if(tflag==false){
		cout << "Error: Opción t es obligatoria." << endl;

		return EXIT_FAILURE;
	}

	if(!pflag){
		////////////////////////////////////////
		// MODO DE ENTRENAMIENTO Y EVALUACIÓN //
		///////////////////////////////////////

		//Objeto perceptrón multicapa
		PerceptronMulticapa mlp;

		//Parámetros del mlp. Por ejemplo, mlp.dEta = valorQueSea;
		int iteraciones;
		int capas;
		int neuronas;

		if(iflag==true){
			iteraciones=atoi(ivalue);
		}

		else{
			iteraciones=1000;
		}

		if(lflag==true){
			capas=atoi(lvalue);
		}

		else{
			capas=1;
		}

		if(hflag==true){
			neuronas=atoi(hvalue);
		}

		else{
			neuronas=5;
		}

		if(eflag==true){
			mlp.dEta=atof(evalue);
		}

		else{
			mlp.dEta=0.1;
		}

		if(mflag==true){
			mlp.dMu=atof(mvalue);
		}

		else{
			mlp.dMu=0.9;
		}

		if(vflag==true){
			mlp.dValidacion=atof(vvalue);
		}

		else{
			mlp.dValidacion=0;
		}

		if(dflag==true){
			mlp.dDecremento=atof(dvalue);
		}

		else{
			mlp.dDecremento=1;
		}

		//Lectura de datos de entrenamiento y test: llamar a mlp.leerDatos(...)
		Datos * pDatosTrain=mlp.leerDatos(tvalue);
		Datos * pDatosTest=mlp.leerDatos(Tvalue);

		if(pDatosTrain==NULL or pDatosTest==NULL){
				return EXIT_FAILURE;
		}

		//Inicializar vector topología
		int * topologia=new int[capas+2];

		topologia[0] = pDatosTrain->nNumEntradas;

		for(int i=1; i<(capas+2-1); i++){
			topologia[i] = neuronas;
		}

		topologia[capas+2-1]=pDatosTrain->nNumSalidas;

		//Inicializar red con vector de topología
		mlp.inicializar(capas+2,topologia);

		//Semilla de los números aleatorios
		int semillas[]={1,2,3,4,5};
		double * erroresTest=new double[5];
		double * erroresTrain=new double[5];
		double mejorErrorTest=1.0;
		for(int i=0; i<5; i++){
			cout << "**********" << endl;
			cout << "SEMILLA " << semillas[i] << endl;
			cout << "**********" << endl;

			srand(semillas[i]);

			mlp.ejecutarAlgoritmoOnline(pDatosTrain,pDatosTest,iteraciones,&(erroresTrain[i]),&(erroresTest[i]));

			cout << "Finalizamos => Error de test final: " << erroresTest[i] << endl;

			//(Opcional - Kaggle) Guardamos los pesos cada vez que encontremos un modelo mejor.
			if(wflag && erroresTest[i] <= mejorErrorTest){
				mlp.guardarPesos(wvalue);
				mejorErrorTest=erroresTest[i];
			}
		}

		cout << "HEMOS TERMINADO TODAS LAS SEMILLAS" << endl;

		double mediaErrorTest = 0,desviacionTipicaErrorTest = 0;
		double mediaErrorTrain = 0,desviacionTipicaErrorTrain = 0;
        
		//Calcular medias y desviaciones típicas de entrenamiento y test
		for(int i=0; i<5; i++){
			mediaErrorTrain += erroresTrain[i];
			mediaErrorTest += erroresTest[i];
		}

		mediaErrorTest/=5;
		mediaErrorTrain/=5;

		double auxTest=0;
		double auxTrain=0;

		for(int i=0;i<5;i++){
			auxTest += pow(erroresTest[i]-mediaErrorTest,2);
			auxTrain += pow(erroresTrain[i]-mediaErrorTrain,2);
		}

		desviacionTipicaErrorTest = sqrt(0.25*auxTest);
		desviacionTipicaErrorTrain = sqrt(0.25*auxTrain);

		cout << "INFORME FINAL" << endl;
		cout << "*************" << endl;
		cout << "Error de entrenamiento (Media +- DT): " << mediaErrorTrain << " +- " << desviacionTipicaErrorTrain << endl;
		cout << "Error de test (Media +- DT):          " << mediaErrorTest << " +- " << desviacionTipicaErrorTest << endl;

		return EXIT_SUCCESS;
	}

	else{

		/////////////////////////////////
		// MODO DE PREDICCIÓN (KAGGLE) //
		////////////////////////////////

		//Desde aquí hasta el final del fichero no es necesario modificar nada.
        
		//Objeto perceptrón multicapa
		PerceptronMulticapa mlp;

		//Inicializar red con vector de topología
		if(!wflag || !mlp.cargarPesos(wvalue)){
			cerr << "Error al cargar los pesos. No se puede continuar." << endl;

			exit(-1);
		}

		//Lectura de datos de entrenamiento y test: llamar a mlp.leerDatos(...)
		Datos * pDatosTest;
		pDatosTest=mlp.leerDatos(Tvalue);
		if(pDatosTest==NULL){
			cerr << "El conjunto de datos de test no es válido. No se puede continuar." << endl;

			exit(-1);
		}

		mlp.predecir(pDatosTest);

		return EXIT_SUCCESS;
	}
}