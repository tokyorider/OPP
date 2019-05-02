#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define Nx 16
#define Ny 16
#define Nz 16
#define hx (Dx / (Nx - 1))
#define hy (Dy / (Ny - 1))
#define hz (Dz / (Nz - 1))
#define x0 -1.0
#define y0 -1.0
#define z0 -1.0
#define Dx 2.0
#define Dy 2.0
#define Dz 2.0
#define a 1e5
#define eps 1e-8


void delete(double **arr, size_t size) {

	for (size_t i = 0; i < size; ++i) {
		free(arr[i]);
	}
	free(arr);

}


double **allocate(size_t size, size_t subSize) {

	int flag = 0;	
	double **arr = (double **)malloc(size * sizeof(double*));
	if (!arr) {
		flag = 1;
	}

	for (size_t i = 0; i < size; ++i) {
	
		arr[i] = (double *)malloc(subSize * sizeof(double));
		if (!arr[i]) {

			delete(arr, i);
			flag = 1;
			break;
		}
	}
	if (flag) {

		fprintf(stderr, "Memory allocation error\n");
		return NULL;
	}
	return arr;

}



double f(double x, double y, double z) {

	return x * x + y * y + z * z;

}


double r(double x, double y, double z) {

	return 6 - a * f(x, y, z);

}


double expression(double **arr, size_t i, size_t j, size_t k, double x, double y, double z) {

	static double expr1 = 1 / (2 * (1 / (hx * hx) + 1 / (hy * hy) + 1 / (hz * hz)) + a);
	double expr2 = (arr[i][(j + 1) * Ny + k] + arr[i][(j - 1) * Ny + k]) / (hx * hx),
		expr3 = (arr[i][j * Ny + k + 1] + arr[i][j * Ny + k - 1]) / (hy * hy),
		expr4 = (arr[i + 1][j * Ny + k] + arr[i - 1][j * Ny + k]) / (hz * hz),
		expr5 = r(x, y, z);\

	return expr1 * (expr2 + expr3 + expr4 - expr5);
	
}



void fill(double **oldPart, double **newPart, double value, size_t partNz, int numProcs, int rank) {

	const size_t zOffset = (rank) ? (rank * Nz / numProcs - 1) : 0 ;
	for (size_t i = 0; i < partNz; ++i) {

		for (size_t j = 0; j < Nx; ++j) {

			for (size_t k = 0; k < Ny; ++k) {

				//detecting border
				if (i == 0 && rank == 0 || i == partNz - 1 && rank == numProcs - 1 ||	
					 j == 0 || k == 0 || j == Nx - 1 || k == Ny - 1) {
					
					double x = x0 + j * hx, y = y0 + k * hy, 
						z = z0 + zOffset * hz + i * hz;
					newPart[i][j * Ny + k] = oldPart[i][j * Ny + k] = f(x, y, z);
				}
				else {
					newPart[i][j * Ny + k] = oldPart[i][j * Ny + k] = value;
				}
			}
		}
	}

}


void solve(double **oldPart, double **newPart, size_t partNz, int numProcs, int rank) {

	const size_t zOffset = rank * (Dz / numProcs), lowerBound = (rank == 0) ? 0 : 1, 
			shift = (rank % 2) ? -1 : 1, begin = (rank % 2) ? partNz - 2 : 1, end = (rank % 2) ? 0 : partNz - 1;
	double maxDif = 0, maxMaxDif = 0;
	MPI_Request leftSendReq, rightSendReq, leftRecvReq, rightRecvReq, reduceReq;
	reduceReq = leftSendReq = rightSendReq = leftRecvReq = rightRecvReq = MPI_REQUEST_NULL;

	//Main algorythm
	double start = MPI_Wtime();
	do { 

		for (size_t i = begin; i != end; i += shift) {

			for (size_t j = 1; j < Nx - 1; ++j) {

				for (size_t k = 1; k < Ny - 1; ++k) {
			
					double x = x0 + j * hx, y = y0 + k * hy, 
						z = z0 + zOffset + (i - lowerBound) * hz;
					newPart[i][j * Ny + k] = expression(oldPart, i, j, k, x, y, z);

					//Finding maximum of difs between old and new
					if (abs(newPart[i][j * Ny + k] - oldPart[i][j * Ny + k]) > maxDif) {
						maxDif = abs(newPart[i][j * Ny + k] - oldPart[i][j * Ny + k]);
					}
				}
			}

			if (i == begin && rank != 0) {
					
					MPI_Wait(&leftSendReq, MPI_STATUS_IGNORE);	
					MPI_Wait(&leftRecvReq, MPI_STATUS_IGNORE);
					MPI_Isend(newPart[1], Nx * Ny, MPI_DOUBLE, rank - 1,  1, MPI_COMM_WORLD, &leftSendReq);
					MPI_Irecv(newPart[0], Nx * Ny, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &leftRecvReq);
			}
			if (i == begin && rank != numProcs - 1) {

					MPI_Wait(&rightSendReq, MPI_STATUS_IGNORE);	
					MPI_Wait(&rightRecvReq, MPI_STATUS_IGNORE);
					MPI_Isend(newPart[partNz - 2], Nx * Ny, MPI_DOUBLE, rank + 1,  2, MPI_COMM_WORLD, &rightSendReq);
					MPI_Irecv(newPart[partNz - 1], Nx * Ny, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &rightRecvReq);
			}
		}
		//Finding max of difs from all processes
		MPI_Wait(&reduceReq, MPI_STATUS_IGNORE);
		MPI_Iallreduce(&maxDif, &maxMaxDif, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, &reduceReq);
		maxDif = 0;

		double **tmp = oldPart;
		oldPart = newPart;
		newPart = tmp;
	} while(eps < maxMaxDif);
	double finish = MPI_Wtime();

	if (rank == 0) {
		printf("Time taken: %lf s\n", finish - start);
	}
}


void print(double **arr, int numProcs, int rank) {

	size_t lowerBound = ((rank == 0) ? 0 : 1), upperBound = lowerBound + Nz / numProcs;
	for (size_t n = 0; n < numProcs; ++n) {

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == n) {

			for (size_t i = lowerBound; i < upperBound; ++i) {

				for (size_t j = 0; j < Nx; ++j) {
	
					for (size_t k = 0; k < Ny; ++k) {
						printf("%lf ", arr[i][j * Ny + k]);
					}
					printf("\n");
				}
			}
			printf("\n\n\n");
		}
	}
}


int main(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);
	
	int numProcs, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//Size of part;+1 or +2 is for the edge points
	size_t partNz = Nz / numProcs + ((rank == 0 || rank == numProcs - 1) ? 1 : 2), subSize = Nx * Ny;
	double **oldPart = allocate(partNz, subSize), **newPart = allocate(partNz, subSize);
	if (!oldPart || !newPart) {
		return 1;
	}

	double f0 = 0;
	fill(oldPart, newPart, f0, partNz, numProcs, rank);
	solve(oldPart, newPart, partNz, numProcs, rank);

	//Printing result	
	//print(oldPart, numProcs, rank);

	delete(oldPart, partNz);
	delete(newPart, partNz);
	MPI_Finalize();
	return 0;

}
