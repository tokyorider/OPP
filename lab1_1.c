#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define eps 1e-6
#define N 10000

void solveEquation(double *A, double *b, double *x, double *y, 
	int width, int numProcs, int rank);

double* matrixOnVectorMultiply(double *matrix, double *vector, int width);

double* vectorDif(double *vector1, double *vector2, int begin1, int end, int begin2);

double scalarMultiply(double *vector1, double *vector2, int width);

double norm(double *vector, int width);

void multiplyOnScalar(double *vector, double scalar, int width);

void printVector(double *vector);

int main(int argc, char *argv[]) {
	
	//Starting work with MPI
	MPI_Init(&argc, &argv);

	//Counting number of proccesses and their ranks
	int numProcs, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//Allocating memory for partial matrix and vector of answers		
	int M = N / numProcs + ((rank < N % numProcs) ? 1 : 0);
	double *A = (double*)malloc(sizeof(double) * N * M),
		*b = (double*)malloc(sizeof(double) * N);
	if (!A || !b) {

		fprintf(stderr, "Memory allocation error\n");
		return 2;
	}

	//Initializing partial matrix
	for (size_t i = 0; i < M; ++i) {

		for (size_t j = 0; j < N; ++j) {

			A[i * N + j] = ((M * rank + i) == j)? 2 : 1; 
		}
	}

	//Initialziing vector of answers
	for (size_t i = 0;i < N;++i) {

		b[i] = i;
	}

	//Allocating memory for variable vectors
	double *x = (double*)calloc(N, sizeof(double)),
	*y = (double*)malloc(sizeof(double) * N);
	if (!x || !y) {

		fprintf(stderr, "Memory allocation error\n");
		return 2;
	}
	
	//Solving
	solveEquation(A, b, x, y, M, numProcs, rank);
	
	//Deallocating memory
	free(A);
	free(b);
	free(x);
	free(y);
	
	//Ending work with MPI
	MPI_Finalize();	
	return 0;	
}


void solveEquation(double *A, double *b, double *x, double *y,
	 int width, int numProcs, int rank) {
	
	double start = MPI_Wtime();

	//Filling arguments for Allgatherv
	int *recvCounts = (int*)malloc(sizeof(int) * numProcs);
	MPI_Allgather(&width, 1, MPI_INT, recvCounts, 1, MPI_INT,
		MPI_COMM_WORLD);

	int *offsets = (int*)malloc(sizeof(int) * numProcs);
	int offset = (rank < N % numProcs) ? width * rank : 
		(N % numProcs) * (width + 1) + (rank - N % numProcs) * width;
	
	MPI_Allgather(&offset, 1, MPI_INT, offsets, 1, MPI_INT, 
		MPI_COMM_WORLD);
	
	 //Counting partial Ax;
        double *partialTmp = matrixOnVectorMultiply(A, x, width);

        //Counting partial Ax - b
        double *partialY = vectorDif(partialTmp, b, 0, width, offsets[rank]);

        free(partialTmp);

	//Counting acc = ||y|| / ||b||
	MPI_Allgatherv(partialY, width, MPI_DOUBLE, y, recvCounts,
		 offsets, MPI_DOUBLE, MPI_COMM_WORLD);

	double normOfB = norm(b, N);
	double accuracy = norm(y, N) / normOfB, tau;

	double *partialX;
	size_t countIters = 0;
	//Main algorithm
	while (eps < accuracy) {

		countIters++;
		//Counting partial Ay
		partialTmp = matrixOnVectorMultiply(A, y, width);
		
		//Counting partial (y, Ay) and partial (Ay, Ay)
		double partialSM1 = scalarMultiply(partialY, partialTmp, width),
			partialSM2 = scalarMultiply(partialTmp, partialTmp, width),
			sm1, sm2;;
		free(partialTmp);

		//Counting tau = (y, Ay) / (Ay, Ay)
		MPI_Allreduce(&partialSM1, &sm1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&partialSM2, &sm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		tau = sm1 / sm2;
			
		//Counting y = tau * y		
		multiplyOnScalar(partialY, tau, width);
		
		//Counting x = x - tau * y
		partialX = vectorDif(x, partialY, offsets[rank], 
			(rank == numProcs - 1) ? N : offsets[rank + 1], 0);
		free(partialY);
		MPI_Allgatherv(partialX, width, MPI_DOUBLE, x, recvCounts, 
			offsets, MPI_DOUBLE, MPI_COMM_WORLD);
		free(partialX);

		//Counting partial Ax
		partialTmp = matrixOnVectorMultiply(A, x, width);
		
		//Counting partialY = Ax - b
		partialY = vectorDif(partialTmp, b, 0, width, offsets[rank]);
		
		free(partialTmp);
		
		//Collecting parts of y
		MPI_Allgatherv(partialY, width, MPI_DOUBLE, y, recvCounts,
			offsets, MPI_DOUBLE, MPI_COMM_WORLD);
		accuracy = norm(y, N) / normOfB;

	}
	
	double end = MPI_Wtime();
	
	free(recvCounts);
	free(offsets);
	free(partialY);	

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
	
		printf("Number of iterations: %zd\n", countIters);
		//printVector(x);
		printf("Time taken: %lf s\n", end - start);
	}

}

double* matrixOnVectorMultiply(double *matrix, double *vector, int width) {

	double *mul = (double *)malloc(sizeof(double) * width),
	*line = (double *)malloc(sizeof(double) * N);

	for (size_t i = 0; i < width; ++i) {

		for (size_t j = 0; j < N; ++j) {

			line[j] = matrix[i * N + j];
		}

		mul[i] = scalarMultiply(line, vector, N);
	}

	free(line);
	return mul;

}


double* vectorDif(double *vector1, double *vector2, int begin1, int end, int begin2) {

	double *dif = (double*)malloc(sizeof(double) * (end - begin1));
	for (size_t i = begin2, j = begin1, k = 0; j < end; ++i, ++j, ++k) {

		dif[k] = vector1[j] - vector2[i];
	}

	return dif;

}


double scalarMultiply(double *vector1, double *vector2, int width) {

	double mul = 0;
	for (size_t i = 0; i < width; ++i) {

		mul += vector1[i] * vector2[i];
	}

	return mul;

}


double norm(double *vector, int width) {

	return sqrt(scalarMultiply(vector, vector, width));

}


void multiplyOnScalar(double *vector, double scalar, int width) {

	for (size_t i = 0; i < width; ++i) {

		vector[i] *= scalar;
	}

} 

void printVector(double *vector) {


	for (size_t i = 0; i < N; ++i) {
	
		printf("%lf\n", vector[i]);
	}

}
