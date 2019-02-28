#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define eps 1e-6
#define N 100

void solveEquation(double *A, double *b, double *x, double *y, 
	int width, int numProcs, int rank);

double* ring(double *matrix, double *vector, int width, int numProcs, int rank);

double* matrixOnVectorMultiply(double *matrix, double *vector, int width1, int width2, int begin);

void vectorDif(double *result, double *vector1, double *vector2, int width);

double scalarMultiply(double *vector1, double *vector2, int width);

double norm(double *vector, int width);

void multiplyOnScalar(double *vector, double scalar, int width);

void printVector(double *vector, int width);

int main(int argc, char *argv[]) {
	
	//Starting work w+ ith MPI
	MPI_Init(&argc, &argv);

	//Counting number of proccesses and their ranks
	int numProcs, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		
	
	//Allocating memory for partial matrix and partial vectors
	int M = N / numProcs + ((rank < N % numProcs) ? 1 : 0);
	double *A = (double*)malloc(sizeof(double) * N * M), 
		*b = (double*)malloc(sizeof(double) * (N / numProcs + 1)),
		*x = (double*)calloc(sizeof(double), (N / numProcs + 1)),
		*y = (double*)malloc(sizeof(double) * (N / numProcs + 1));
		
	if (!(A && b && x && y)) {
	
		fprintf(stderr, "Memory allocation error\n");
		MPI_Finalize();
	}	

	//Initializing partial matrix
	for (size_t i = 0; i < M; ++i) {
			
		for (size_t j = 0; j < N; ++j) {
		
			A[i * N + j] = ((M * rank + i) == j)? 2 : 1; 
		}
	}
	
	//Filling partial b
	for (size_t i = 0;i < M;++i) {
	
		b[i] = i;;
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
	
	 //Counting partial Ax;
	double *tmp = ring(A, x, width, numProcs, rank);
		
        //Counting partial Ax - b
        vectorDif(y, tmp, b, width);

        free(tmp);

	//Counting acc = ||y|| / ||b||
	double partialSquareNormOfB = norm(b, width), partialSquareNormOfY = norm(y, width);
	partialSquareNormOfB *= partialSquareNormOfB;
	partialSquareNormOfY *= partialSquareNormOfY;
	double squareNormOfB, squareNormOfY;
	MPI_Allreduce(&partialSquareNormOfB, &squareNormOfB, 1, MPI_DOUBLE, MPI_SUM, 
			MPI_COMM_WORLD);
	MPI_Allreduce(&partialSquareNormOfY, &squareNormOfY, 1, MPI_DOUBLE, MPI_SUM, 
			MPI_COMM_WORLD);
	
	double accuracy = sqrt(squareNormOfY) / sqrt(squareNormOfB), tau;
	
	size_t countIters = 0;
	//Main algorithm
	while (eps < accuracy) {
	
		countIters++;
		//Counting partial Ay
		tmp = ring(A, y, width, numProcs, rank);
		
		//Counting partial (y, Ay) and partial (Ay, Ay)
		double partialSM1 = scalarMultiply(y, tmp, width),
			partialSM2 = scalarMultiply(tmp, tmp, width),
			sm1, sm2;

		//Counting tau = (y, Ay) / (Ay, Ay)
		MPI_Allreduce(&partialSM1, &sm1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&partialSM2, &sm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		free(tmp);	
	
		tau = sm1 / sm2;
	
		//Counting y = tau * y		
		multiplyOnScalar(y, tau, width);
		
		//Counting x = x - tau * y
		vectorDif(x, x, y, width);

		//Counting partial Ax
		tmp = ring(A, x, width, numProcs, rank);

		//Counting partialY = Ax - b
		vectorDif(y, tmp, b, width);
		free(tmp);

		//Counting accuracy
		partialSquareNormOfB = norm(b, width);
		partialSquareNormOfY = norm(y, width);
		partialSquareNormOfB *= partialSquareNormOfB;
		partialSquareNormOfY *= partialSquareNormOfY;
		MPI_Allreduce(&partialSquareNormOfB, &squareNormOfB, 1, MPI_DOUBLE, MPI_SUM, 
				MPI_COMM_WORLD);
		MPI_Allreduce(&partialSquareNormOfY, &squareNormOfY, 1, MPI_DOUBLE, MPI_SUM, 
				MPI_COMM_WORLD);

		accuracy = sqrt(squareNormOfY) / sqrt(squareNormOfB);

	}
	
	double end = MPI_Wtime();

	MPI_Barrier(MPI_COMM_WORLD);
	for (size_t i = 0;i < numProcs;++i) {

		if (rank == i) {
	
			//printVector(x, width);
		}
		
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		
		printf("Number of iterations: %zd\n", countIters);
		printf("Time taken: %lf s\n", end - start);
	}

}


double* ring(double *matrix, double *vector, int width, int numProcs, int rank) {

	double *tmp = (double*)calloc(sizeof(double), width);
	int newWidth = width, nextRank = (rank + 1) % numProcs, 
		prevRank = (rank) ? rank - 1 : numProcs - 1, 
	offset = (rank < N % numProcs) ? width * rank : (width + 1) * (N % numProcs) + width * (rank - N % numProcs);

	for (int i = 0;i < numProcs;++i) {
	
        	double *partialTmp = matrixOnVectorMultiply(matrix, vector, width, newWidth, 
	offset);

		for (size_t j = 0;j < width;++j) {
			
			tmp[j] += partialTmp[j];
		}
		free(partialTmp);

		if (numProcs == 1) {
			
			return tmp;
		}

		MPI_Sendrecv(vector, newWidth, MPI_DOUBLE, nextRank, 0, vector,
				N / numProcs + 1, MPI_DOUBLE, prevRank, 0 ,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);	

		offset = (offset + newWidth) % N;
		int prev = (rank - i - 1 >= 0) ? rank - i - 1 : numProcs - abs(rank - i - 1);
		newWidth = (prev < N % numProcs) ? N / numProcs + 1 : N / numProcs;

	}

	return tmp;
	
}

double* matrixOnVectorMultiply(double *matrix, double *vector, int width1, int width2, int begin) {

	double *mul = (double *)malloc(sizeof(double) * width1),
	*line = (double *)malloc(sizeof(double) * width2);

	for (size_t i = 0; i < width1; ++i) {

		for (size_t j = begin, k = 0; k < width2; ++j, ++k) {

			line[k] = matrix[i * N + j];
		}

		mul[i] = scalarMultiply(line, vector, width2);
	}

	free(line);
	return mul;

}


void vectorDif(double *result, double *vector1, double *vector2, int width) {

	for (size_t i = 0; i < width; ++i) {

		result[i] = vector1[i] - vector2[i];
	}

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


void printVector(double *vector, int width) {


	for (size_t i = 0; i < width; ++i) {
	
		printf("%lf\n", vector[i]);
	}

}
