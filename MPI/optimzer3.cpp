#include "mpi.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;
#define MIN(a,b)  ((a)<(b)?(a):(b))
#define ID(x) (((x)-2)/2)
#define RID(x) (2*(x)+3)
int main (int argc, char *argv[]) {
	int    count;        /* Local prime count */
	double elapsed_time; /* Parallel execution time */
	int    first;        /* Index of first multiple */
	int    global_count; /* Global prime count */
	int    high_value;   /* Highest value on this proc */
	int    i,j;
	int    id;           /* Process ID number */
	int    index;        /* Index of current prime */
	int    low_value;    /* Lowest value on this proc */
	bool   *marked,*flag;/* Portion of 2,...,'n' */
	int    n;            /* Sieving from 2, ..., 'n' */
	int    p;            /* Number of processes */
	int    proc0_size;   /* Size of proc 0's subarray */
	int    prime;        /* Current prime */
	int    size;         /* Elements in 'marked' */
	int    id_low_value;
	int    id_high_value;
	int	   low_value0;
	int    high_value0;
	int    size0;
	vector<int> pri;
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
	low_value += !(low_value & 1);
	high_value = 2 + (id+1)*(n-2)/p;
	high_value -= !(high_value & 1);
	id_low_value = ID(low_value);
	id_high_value = ID(high_value);
	size = id_high_value - id_low_value + 1;
	low_value0 = 0;
	high_value0 = (int) sqrt((double) n);
	high_value0 -= !(high_value0 & 1);
	size0 = high_value0 - low_value0 + 1;
	/* Bail out if all the primes used for sieving are
	   not all held by process 0 */
	proc0_size = (n-2)/p;
	if ((2 + proc0_size) < (int) sqrt((double) n)) {
		if (!id) printf ("Too many processes\n");
		MPI_Finalize();
		exit (1);
	}
	/* Allocate this process's share of the array. */
	marked = (bool *) malloc (size);
	flag = (bool *)malloc (size0);
	if (marked == NULL || flag == NULL) {
		printf ("Cannot allocate enough memory\n");
		MPI_Finalize();
		exit (1);
	}
	memset(marked,0,size);
	memset(flag,0,size0);
	for(i = 2; i <= high_value0; i++)
	{
		if(marked[i]) continue;
		pri.push_back(i);
		for(j = i * i; j < high_value0; j += i) marked[j] = 1; 
	}
	for(int j = 1; j < pri.size(); j++)
	{
		prime = pri[j];
		if (prime * prime > low_value)
			first = prime * prime - low_value;
		else {
			if (!(low_value % prime)) first = ((low_value / prime) & 1) ? 0 : prime;
			else first = prime - (low_value % prime) + (((low_value / prime) & 1) ? prime : 0);
		}
//		printf("%d %d %d %d %d %d\n",low_value,high_value,id_low_value,id_high_value,prime,first);
		for (i = ID(first+low_value); i <= id_high_value; i += prime) marked[i-id_low_value] = 1;// printf("%d %d\n",id,i-id_low_value);
	}
	count = 0;
	for (i = 0; i < size; i++)
		if (!marked[i]) count++;//,printf("%d\n",RID(id_low_value+i));
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
