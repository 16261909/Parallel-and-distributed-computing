#pragma GCC optimzer("O3")
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define MIN(a,b)  ((a)<(b)?(a):(b))
int main (int argc, char *argv[]) {
	int    count;        /* Local prime count */
	double elapsed_time; /* Parallel execution time */
	int    first;        /* Index of first multiple */
	int    global_count; /* Global prime count */
	int    high_value;   /* Highest value on this proc */
	int    i;
	int    id;           /* Process ID number */
	int    index;        /* Index of current prime */
	int    low_value;    /* Lowest value on this proc */
	char  *marked;       /* Portion of 2,...,'n' */
	int    n;            /* Sieving from 2, ..., 'n' */
	int    p;            /* Number of processes */
	int    proc0_size;   /* Size of proc 0's subarray */
	int    prime;        /* Current prime */
	int    size;         /* Elements in 'marked' */
	MPI_Init (&argc, &argv);
	/* Start the timer */
	MPI_Comm_rank (MPI_COMM_WORLD, &id);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = -MPI_Wtime();
	if (argc != 2) {
		if (!id) printf ("Command line: %s <m>\n", argv[0]);
		MPI_Finalize();
		exit (1);
	}
	n = atoi(argv[1]);
	/* Figure out this process's share of the array, as
	   well as the integers represented by the first and
	   last array elements */
	low_value = 3 + id*(n-2)/p;
	high_value = 2 + (id+1)*(n-2)/p;
	size = high_value - low_value + 1;
	/* Bail out if all the primes used for sieving are
	   not all held by process 0 */
	proc0_size = (n-2)/p;
	if ((2 + proc0_size) < (int) sqrt((double) n)) {
		if (!id) printf ("Too many processes\n");
		MPI_Finalize();
		exit (1);
	}
	/* Allocate this process's share of the array. */
	marked = (char *) malloc (size);
	if (marked == NULL) {
		printf ("Cannot allocate enough memory\n");
		MPI_Finalize();
		exit (1);
	}
	for (i = 0; i < size; i++) marked[i] = 0;
	if (!id) index = 0;
	prime = 3;
	do {
		if (prime * prime > low_value)
			first = prime * prime - low_value;
		else {
//			if (low_value % prime == 0) first = 0;
//			else first = prime - (low_value % prime);
			if (!(low_value % prime)) first = ((low_value / prime) & 1) ? 0 : prime;
			else first = prime - (low_value % prime) + (((low_value / prime) & 1) ? prime : 0);
//			printf("%d %d %d\n",low_value,prime,first);
		}
//		printf("%d %d %d %d\n",low_value,high_value,prime,first);
		for (i = first; i < size; i += prime << 1) marked[i] = 1;
		if (!id) {
			do {
				index += 2;
			} while (marked[index]);
			prime = index + 3;
			//printf("%d\n",prime);
		}
		MPI_Bcast (&prime,  1, MPI_INT, 0, MPI_COMM_WORLD);
	} while (prime * prime <= n);
	count = 0;
	for (i = ( low_value & 1 ) ? 0 : 1 ; i < size; i+=2)
		if (!marked[i]) count++;
	MPI_Reduce (&count, &global_count, 1, MPI_INT, MPI_SUM,
		                       0, MPI_COMM_WORLD);
	/* Stop the timer */
	elapsed_time += MPI_Wtime();
	/* Print the results */
	if (!id) {
		printf ("There are %d primes less than or equal to %d\n",
		        global_count+1, n);
		printf ("SIEVE (%d) %10.6f\n", p, elapsed_time);
	}
	MPI_Finalize ();
	return 0;
}
