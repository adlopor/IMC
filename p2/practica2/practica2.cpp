//============================================================================
// Introducción a los Modelos Computacionales
// Name        : practica2.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // Para cojer la hora time()
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <string.h>
#include <math.h>
#include "imc/PerceptronMulticapa.h"

using namespace imc;
using namespace std;

int main(int argc, char **argv) {
	/*
	    int c;
	*/
	// Procesar los argumentos de la línea de comandos
    bool tflag = 0, Tflag = 0, iflag = 0, lflag =0, hflag = 0, eflag = 0, mflag = 0, vflag = 0, dflag = 0, oflag=0, fflag = 0, sflag = 0, wflag = 0, pflag = 0;
	char *tvalue = NULL, *Tvalue = NULL, *ivalue = NULL, *lvalue = NULL, *hvalue = NULL, *evalue = NULL, *mvalue = NULL, *vvalue = NULL, *dvalue = NULL, *fvalue = NULL, *wvalue = NULL;    
    int c;

    opterr = 0;

    // a: opción que requiere un argumento
    // a:: el argumento requerido es opcional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:v:d:of:sw:p")) != -1)
    {
        // Se han añadido los parámetros necesarios para usar el modo opcional de predicción (kaggle).
        // Añadir el resto de parámetros que sean necesarios para la parte básica de las prácticas.
        switch(c){
        
        	case 't':
                tflag = true;
                tvalue = optarg;
                break;
                
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
                
			case 'i':
                iflag = true;
                ivalue = optarg;
                break;
                
            case 'l':
                lflag = true;
                lvalue = optarg;
                break;
            
            case 'h':
                hflag = true;
                hvalue = optarg;
                break;
                
            case 'e':
                eflag = true;
                evalue = optarg;
                break;
            
            case 'm':
                mflag = true;
                mvalue = optarg;
                break;
                
            case 'v':
                vflag = true;
                vvalue = optarg;
                break;
                
            case 'd':
                dflag = true;
                dvalue = optarg;
                break;
            
            case 'o':
                oflag = true;
                break;
                
            case 'f':
                fflag = true;
                fvalue = optarg;
                break;
                
            case 's':
                sflag = true;
                break;
            
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            
            case 'p':
                pflag = true;
                break;
            
            case '?':
                if (optopt == 't' || optopt == 'T' || optopt == 'i' || optopt == 'l' ||optopt == 'h' || optopt == 'e' || optopt == 'm' || optopt == 'v' || optopt == 'd' || optopt == 'f' || optopt == 'w')
                    fprintf (stderr, "La opción -%c requiere un argumento.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Opción desconocida `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Caracter de opción desconocido `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

    if (!pflag) {

        ////////////////////////////////////////
        // MODO DE ENTRENAMIENTO Y EVALUACIÓN //
        ///////////////////////////////////////

    	// Objeto perceptrón multicapa
    	PerceptronMulticapa mlp;

		

    	
    	

		if (!tflag){
              fprintf (stderr, "La opción -t es necesaria para la ejecución.\n");
              return EXIT_FAILURE;
        }
        
		//Parámetros del mlp. Por ejemplo, mlp.dEta = valorQueSea;
		int iteraciones, capas, neuronas, error;
		
		if(iflag==true){
			iteraciones = atoi(ivalue);
		}

		else{
			iteraciones = 1000;
		}

		if(lflag==true){
			capas = atoi(lvalue);
		}

		else{
			capas = 1;
		}

		if(hflag==true){
			neuronas = atoi(hvalue);
		}

		else{
			neuronas = 4;
		}

		if(eflag==true){
			mlp.dEta = atof(evalue);
		}

		else{
			mlp.dEta = 0.7;
		}

		if(mflag==true){
			mlp.dMu = atof(mvalue);
		}

		else{
			mlp.dMu = 1.0;
		}

		if(vflag==true){
			mlp.dValidacion = atof(vvalue);
		}

		else{
			mlp.dValidacion = 0;
		}

		if(dflag==true){
			mlp.dDecremento = atof(dvalue);
		}

		else{
			mlp.dDecremento=1;
		}
		if(fflag==true){
			error = atof(fvalue);
		}

		else{
			error = 0;
		}
		
		mlp.bOnline = oflag;
		
		// Lectura de datos de entrenamiento y test: llamar a mlp.leerDatos(...)
		Datos * pDatosTrain = mlp.leerDatos(tvalue);
		Datos * pDatosTest = NULL;
		
		if(Tflag==true){
			pDatosTest = mlp.leerDatos(Tvalue);
		}

		else{
			pDatosTest = mlp.leerDatos(tvalue);
		}
		
		if(pDatosTrain==NULL or pDatosTest==NULL){
				return EXIT_FAILURE;
		}
		
		// Inicializar vector topología
    	int *topologia = new int[capas+2];
    	int *tipoCapas = new int[capas+2];
    	
    	//Capa de entrada
    	topologia[0] = pDatosTrain->nNumEntradas;
    	tipoCapas[0] = 0;
    	
    	//Capas ocultas
    	for(int i=1; i<(capas+2-1); i++){
    		
    		tipoCapas[i] = 0;
    	  	topologia[i] = neuronas;
    	}
    	
    	//Capa de salida
    	topologia[capas+2-1] = pDatosTrain->nNumSalidas;

		if(sflag){
        	tipoCapas[capas+2-1]=1;
        }
        else{
        	tipoCapas[capas+2-1]=0;
		}
		
		// Inicializar red con vector de topología
    	mlp.inicializar(capas+2, topologia, tipoCapas);
		
		//Reservamos memoria para la matriz de Confusión (CCR).
		int **matrizConf = new int* [pDatosTrain->nNumSalidas];
		
		for(int i=0; i<pDatosTrain->nNumSalidas; i++)
        	matrizConf[i] = new int[pDatosTrain->nNumSalidas];
        	
        // Semilla de los números aleatorios
        int semillas[] = {1};
        //double *errores = new double[5];
        double *erroresTrain = new double[5];
        double *erroresTest = new double[5];
        double *ccrs = new double[5];
        double *ccrsTrain = new double[5];
        double mejorErrorTest = 1.0;
        
        for(int i=0; i<1; i++){
        	//cout << "**********" << endl;
        	//cout << "SEMILLA " << semillas[i] << endl;
        	//cout << "**********" << endl;
    		srand(semillas[i]);
    		mlp.ejecutarAlgoritmo(pDatosTrain, pDatosTest, iteraciones, &(erroresTrain[i]), &(erroresTest[i]), &(ccrsTrain[i]), &(ccrs[i]), error, matrizConf);
    		//cout << "Finalizamos => CCR de test final: " << ccrs[i] << endl;

            // (Opcional - Kaggle) Guardamos los pesos cada vez que encontremos un modelo mejor.
            if(wflag && erroresTest[i] <= mejorErrorTest)
            {
                mlp.guardarPesos(wvalue);
                mejorErrorTest = erroresTest[i];
            }

        }

        double mediaErrorTrain = 0, desviacionTipicaErrorTrain = 0;
        double mediaErrorTest = 0, desviacionTipicaErrorTest = 0;
        double mediaCCR = 0, desviacionTipicaCCR = 0;
        double mediaCCRTrain = 0, desviacionTipicaCCRTrain = 0;

		/*-----------CHIVATO CCRS---------------*/
		/*cout << "Chivato de los CCRs:" << endl;
		for (int i=0; i<5; i++){
			
			cout << "CCRTest[ " << i << " ]: " << ccrs[i] << endl;
			cout << "CCRTrain[ " << i << " ]: " << ccrsTrain[i] << endl;
			
		}*/
		
		
        // Calcular medias y desviaciones típicas de entrenamiento y test
		for(int i=0; i<1; i++){
		
			mediaCCR += ccrs[i];
			mediaCCRTrain += ccrsTrain[i];
			mediaErrorTrain += erroresTrain[i];
			mediaErrorTest += erroresTest[i];
		
		}
		
		mediaCCRTrain /= 5;
		mediaCCR /= 5;
		mediaErrorTest /= 5;
		mediaErrorTrain /= 5;

		double auxTest = 0;
		double auxTrain = 0;
		double auxCCRTest = 0;
		double auxCCRTrain = 0;
		
		for(int i=0; i<5; i++){
		
			auxCCRTest += pow(ccrs[i]-mediaCCR,2);
			auxCCRTrain += pow(ccrsTrain[i]-mediaCCRTrain,2);
			auxTest += pow(erroresTest[i]-mediaErrorTest,2);
			auxTrain += pow(erroresTrain[i]-mediaErrorTrain,2);
		
		}
		
		desviacionTipicaCCRTrain= sqrt(auxCCRTrain/4);
		desviacionTipicaCCR = sqrt(auxCCRTest/4);
		desviacionTipicaErrorTest = sqrt(auxTest/4);
		desviacionTipicaErrorTrain = sqrt(auxTrain/4);

        //cout << "HEMOS TERMINADO TODAS LAS SEMILLAS" << endl;

    	//cout << "INFORME FINAL" << endl;
    	//cout << "*************" << endl;
        //cout << "Error de entrenamiento (Media +- DT): " << mediaErrorTrain << " +- " << desviacionTipicaErrorTrain << endl;
        //cout << "Error de test (Media +- DT): " << mediaErrorTest << " +- " << desviacionTipicaErrorTest << endl;
        //cout << "CCR de entrenamiento (Media +- DT): " << mediaCCRTrain << " +- " << desviacionTipicaCCRTrain << endl;
        //cout << "CCR de test (Media +- DT): " << mediaCCR << " +- " << desviacionTipicaCCR << endl;
        
        /*
        cout << mediaErrorTrain << ";" << desviacionTipicaErrorTrain << ";" << mediaErrorTest << ";" << desviacionTipicaErrorTest << ";" << mediaCCRTrain << ";" << desviacionTipicaCCRTrain << ";" << mediaCCR << ";" << desviacionTipicaCCR << endl;
        */
      	
        //Añadimos a la salida la matriz de confusion obtenida.
        
       /* cout << "Matriz de confusión" << endl;
        for(int i=0; i<pDatosTrain->nNumSalidas; i++){
        
        	cout << "|";
        
        	for(int j=0; j<pDatosTrain->nNumSalidas; j++){
        		cout << " " << matrizConf[j][i];
        	}
        	
        	cout << " |" << endl;
        }
        */
        
    	return EXIT_SUCCESS;
    }
    else{

        /////////////////////////////////
        // MODO DE PREDICCIÓN (KAGGLE) //
        ////////////////////////////////

        // Desde aquí hasta el final del fichero no es necesario modificar nada.
        
        // Objeto perceptrón multicapa
        PerceptronMulticapa mlp;
	
	    // Inicializar red con vector de topología
        if(!wflag || !mlp.cargarPesos(wvalue))
        {
            cerr << "Error al cargar los pesos. No se puede continuar." << endl;
            exit(-1);
        }

        // Lectura de datos de entrenamiento y test: llamar a mlp.leerDatos(...)
        Datos *pDatosTest;
        pDatosTest = mlp.leerDatos(Tvalue);
        if(pDatosTest == NULL)
        {
            cerr << "El conjunto de datos de test no es válido. No se puede continuar." << endl;
            exit(-1);
        }

        mlp.predecir(pDatosTest);

        return EXIT_SUCCESS;

    }
}

