#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define eps 1e-6
#define N 10000

void solveEquation(double *A, double *b, double *x, double *y, 
	double *tmp);

void matrixOnVectorMultiply(double *result, double *matrix, double *vector);

void vectorDif(double *result, double *vector1, double *vector2);

double scalarMultiply(double *vector1, double *vector2);

double norm(double *vector);

void multiplyOnScalar(double *vector, double scalar);

void printVector(double *vector);

int main() {

	//Allocating memory for vectors and matrix	
	double *x = (double*)calloc(N, sizeof(double)), *y = (double*)malloc(sizeof(double) * N), 
		*b = (double*)malloc(sizeof(double) * N), *A = (double*)malloc(sizeof(double) * N * N),
		*tmp = (double*)malloc(sizeof(double) * N);
	if (!(A && b && x && y)) {

			fprintf(stderr, "Memory allocation error\n");

	}

	//Initialziing vector of answers and matrix
	for (size_t i = 0;i < N;++i) {

		for (size_t j = 0; j < N; ++j) {
		
			A[i * N + j] = (i == j)? 2 : 1;
		}

		b[i] = i;

	}
	
	//Counting number of threads and their rank
	#pragma omp parallel 
	{
		
		//Solving
		solveEquation(A, b, x, y, tmp);
		
	}	
	
	//Deallocating memory
	free(A);
	free(b);
	free(x);
	free(y);
	free(tmp);
	
	return 0;	
}


void solveEquation(double *A, double *b, double *x, double *y,
	double *tmp) {

	double start = omp_get_wtime();

	//Counting Ax;
	matrixOnVectorMultiply(tmp, A, x);
        //Counting y = Ax - b
	vectorDif(y, tmp, b);

	//Counting acc = ||y|| / ||b||
	double normOfB = norm(b);
	double accuracy = norm(y) / normOfB, tau;

	//Main algorithm
	while (eps < accuracy) { 

		//Counting Ay
		matrixOnVectorMultiply(tmp, A, y);

		//Counting tau = (y, Ay) / (Ay, Ay)
		tau = scalarMultiply(y, tmp) / scalarMultiply(tmp, tmp);

		//Counting y = tau * y		
		multiplyOnScalar(y, tau);
	
		//Counting x = x - tau * y
		vectorDif(x, x, y);

		//Counting Ax
		matrixOnVectorMultiply(tmp, A, x);

		//Counting y = Ax - b
		vectorDif(y, tmp, b);
		
		accuracy = norm(y) / normOfB;

	}
	double end = omp_get_wtime();

	#pragma omp master 
	{

		printVector(x);
		printf("Time taken: %lf s\n", end - start);

	}

}


void matrixOnVectorMultiply(double *result, double *matrix, double *vector) {

	double *line = (double *)malloc(sizeof(double) * N);

	#pragma omp for
	for (size_t i = 0; i < N; ++i) {

		for (size_t j = 0; j < N; ++j) {

			line[j] = matrix[i * N + j];
		}

		result[i] = scalarMultiply(line, vector);
	}

	free(line);

}


void vectorDif(double *result, double *vector1, double *vector2) {

	#pragma omp for
	for (size_t i = 0; i < N; ++i) {

		result[i] = vector1[i] - vector2[i];
	
	}

}


double scalarMultiply(double *vector1, double *vector2) {
	
	double mul = 0;

	#pragma for shared(mul) reduction(+:mul)
	for (size_t i = 0; i < N; ++i) {

		mul += vector1[i] * vector2[i];

	}

	return mul;

}


double norm(double *vector) {

	return sqrt(scalarMultiply(vector, vector));

}


void multiplyOnScalar(double *vector, double scalar) {

	#pragma omp for
	for (size_t i = 0; i < N; ++i) {

		vector[i] *= scalar;

	}

} 

void printVector(double *vector) {


	for (size_t i = 0; i < N; ++i) {
	
		printf("%lf\n", vector[i]);
	}

}
