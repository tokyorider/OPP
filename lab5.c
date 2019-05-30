#include <iostream>
#include <math.h>
#include <mpi.h>
#include <pthread.h>
#include <queue>
#include <unistd.h>

using Task = int;
using TaskList = std::queue<Task>;
TaskList taskList = TaskList();
inline size_t NUM_ITERS = 5, NUM_TASKS_PER_ITER = 20, MODE = 1;

size_t getListSize(const int size, const int rank) {
	switch (MODE) {
		case 1 : {
			size_t rest = (rank < NUM_TASKS_PER_ITER % size) ? 1 : 0;
			return NUM_TASKS_PER_ITER / size + rest;
		}
		case 2 : {
			const size_t half = NUM_TASKS_PER_ITER / std::min(size, 2);
			if (rank == 0) {
				return half;
			}

			size_t rest = (rank - 1 < (NUM_TASKS_PER_ITER - half) % (size - 1)) ? 1 : 0;
			return (NUM_TASKS_PER_ITER - half) / (size - 1) + rest;
		}
		case 3 : {
			return (rank == 0) ? NUM_TASKS_PER_ITER : 0;
		}			
	}
}


void initTaskList(const int size, const int rank, const int iter) {
	static const size_t listSize = getListSize(size, rank);
	Task task = fabs(2 * rank - iter % size) + 1;
	for (size_t i = 0; i < listSize; ++i) {
		taskList.push(task);
	}
}


void completeTask(const Task &task, const int rank) {
	sleep(task);
	fprintf(stderr, "Process %d has slept for %d secs\n", rank, task);
}


void* completeTasks(void *arg) {
	const int size = ((int *) arg)[0], rank = ((int *) arg)[1];
	pthread_mutex_t mutex;
	pthread_mutex_init(&mutex, NULL);

	for (size_t iter = 0; iter < NUM_ITERS; ++iter) {
		MPI_Barrier(MPI_COMM_WORLD);
		initTaskList(size, rank, iter);
		fprintf(stderr, "Process with number %d has got taskList with length %zd\n", rank, 
			taskList.size());

		size_t i = 1;
		do {
			Task task;			
			while (!taskList.empty()) {
				pthread_mutex_lock(&mutex);
				task = taskList.front();
				taskList.pop();
				pthread_mutex_unlock(&mutex);
				completeTask(task, rank);
			}
			
			int neighbour = (rank + i) % size;
			MPI_Send(&rank, 1, MPI_INT, neighbour, 1, MPI_COMM_WORLD);
			MPI_Recv(&task, 1, MPI_INT, neighbour, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (task != 0) {
				taskList.push(task);
			}
			else {
				++i;
			}
		} while (i < size);
	}
	int quit = -1;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Send(&quit, 1, MPI_INT, rank, 1, MPI_COMM_WORLD);

	pthread_mutex_destroy(&mutex);
}


void* shareTasks(void *ptr) {
	int rank;
	pthread_mutex_t mutex;

	while (true) {
		MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (rank == -1) {
			break;
		}		
		
		Task task = 0;
		if (!taskList.empty()) {
			pthread_mutex_lock(&mutex);
			task = taskList.front();
			taskList.pop();
			pthread_mutex_unlock(&mutex);
		}
		MPI_Send(&task, 1, MPI_INT, rank, 2, MPI_COMM_WORLD);
	}

	pthread_mutex_destroy(&mutex);
}


int getJoinedAttributes(pthread_attr_t *attrs) {
	if (pthread_attr_init(attrs) != 0) {
		perror("Cannot init thread attributes");
		return 1;
	}

	if (pthread_attr_setdetachstate(attrs, PTHREAD_CREATE_JOINABLE) != 0) {
		perror("Cannot set attributes joined");
		return 2;
	}

	return 0;
}


int *initParams() {
	int rank, numProcs;
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int *params = (int *)malloc(2 * sizeof(int));
	params[0] = numProcs;
	params[1] = rank;
	return params;
}


int createThread(pthread_t *thread, const pthread_attr_t *attrs, void *(*func) (void *), void *arg) {
	if (pthread_create(thread, attrs, func, arg) != 0) {
		perror("Cannot create thread");
		return 1;
	}
	
	return 0;
}	


int joinThread(pthread_t thread, void **returnValue) {
	if (pthread_join(thread, returnValue) != 0) {
		perror("Cannot create thread");
		return 1;
	}
	
	return 0;
}	


int main(int argc, char *argv[]) {
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, NULL);

	pthread_attr_t attrs;
	if (getJoinedAttributes(&attrs) != 0) {
		return 1;
	}

	pthread_t taskThread, shareThread;
	int *params = initParams();
	if (createThread(&taskThread, &attrs, completeTasks, params) != 0 ||
		createThread(&shareThread, &attrs, shareTasks, NULL) != 0) 
	{
		return 1;
	}
	pthread_attr_destroy(&attrs);

	if (joinThread(taskThread, NULL) != 0 || joinThread(shareThread, NULL) != 0) {
		return 1;
	}

	MPI_Finalize();	
	return 0;
}
