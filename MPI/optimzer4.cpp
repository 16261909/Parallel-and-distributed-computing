#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
inline int MIN(int a,int b)
{
	if(a<b)return a;
	return b;
}
inline int MAX(int a,int b)
{
	if(a<b)return b;
	return a;
}
#define ID(x) (((x)-2)>>1)	      /* Logical address to physical address */
#define RID(x) (((x)<<1)+3)	      /* Physical address to Logical address */
int main (int argc, char *argv[]) {
	int    			count;
	double 			elapsed_time; /* Parallel execution time */
	int    			global_count; /* Global prime count */
	register int    high_value;   /* Highest value on this proc */
	register int    i,j,k;	      /* Iterative variable */
	register int 	L,tL;	      /* Left bound of current thread  */
	int    			id;           /* Process ID number */
	int    			index;        /* Index of current prime */
	register int    low_value;    /* Lowest value on this proc */
	bool   			*marked;      /* Portion of 2,...,'n' */
	int 			pri[33000];   /* Save prime numbers */
	int 			top;	      /* Size of pri[] */
	register int    n;            /* Sieving from 2, ..., 'n' */
	register int    p;            /* Number of processes */
	register int	r,d;          /* Reminder, quotient */
	register int    prime,p2;     /* Current prime, Square of prime*/
	int    			size;         /* Elements in 'marked' */
	register int    id_low_value; /* Mapped low value */
	register int    id_high_value;/* Mapped high value */
	int    			low_value0;   /* Low value for Sieving */
	int    			high_value0;  /* High value for Sieving */
//	int				id_low_value0;
//	register int	id_high_value0;
	int    			size0;        /* Mapped size */
	int    			sizep;	      /* Chunk size, used for cache acceleration */
	MPI_Init (&argc, &argv);
	/* Start the timer */
	MPI_Comm_rank (MPI_COMM_WORLD, &id);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = -MPI_Wtime();
	if (argc < 2) {
		if (!id) printf ("Command line: %s <m>\n", argv[0]);
		MPI_Finalize();
		exit (1);
	}
	n = atoi(argv[1]);
	/* Figure out this process's share of the array, as
	   well as the integers represented by the first and
	   last array elements */
	low_value = 3 + 1ll * id * (n-2) / p;
	low_value += !(low_value & 1);
	high_value = 2 + 1ll* (id + 1) * (n - 2) / p;
	high_value -= !(high_value & 1);
	id_low_value = ID(low_value);
	id_high_value = ID(high_value);
	high_value0 = (int) sqrt((double)n)+1;
	high_value0 -= !(high_value0 & 1);
//	id_low_value0 = ID(low_value0);
//	id_high_value0 = ID(high_value0);
	size = MAX(high_value0, id_high_value - id_low_value + 1);
	sizep = 210000;
	/* Allocate this process's share of the array. */
	marked = (bool *) malloc (size);
	//pri = (int *) malloc (high_value0 * sizeof(int));
	//static int pri[high_value0];
	if (marked == NULL) {
		printf ("Cannot allocate enough memory\n");
		MPI_Finalize();
		exit (1);
	}
	memset(marked, 0, high_value0);
	/*delete even numbers
	for(i = id_low_value0; i < id_high_value0; i++)
	{
		if(marked[i])continue;
		pri[top++] = prime = RID(i);
		for(j = prime * prime; j < high_value0; j += prime << 1)marked[ID(j)] = 1;
	}
	*/
	for(i = 2; i < high_value0; i++)
	{
		/*
		if(!marked[i])pri[top++];
		for(j = 0; j < top && pri[j] * i < high_value0; j++)
		{
			marked[i * pri[j]] = 1;
			if(!(i % pri[j])) break;
		}
		*/
		///*
		if(marked[i]) continue;
		pri[top++] = i;
		for(j = i * i; j < high_value0; j += i) marked[j] = 1; 
		//*/
	}
	memset(marked, 0, size0);
//	printf("%10.6f\n",elapsed_time+MPI_Wtime());
	//return 0;
	//for(i = 0;i < top; i++)printf("%d\n",pri[i]);
	for(L = ID(low_value); L <= id_high_value; L += sizep)
	{
		tL = RID(L);
		//printf("[%d %d]\n",RID(L),RID(L+sizep -1));
		for(j = 1; j < top; j++)
		{
			prime = pri[j];
			p2 = prime * prime;
			int first = 0;
			if (p2 > tL)
				first = p2 - tL;
			else {
				r = tL % prime, d = tL / prime;
				if (!r) first = (1 - (d & 1)) * prime;
				else first = prime - r + (d & 1) * prime;
			}
			//printf("%d\n",first);
			//printf("%d %d\n",MAX(ID(first+tL),L),MIN(L+sizep-1,id_high_value));
			r = MIN(L + sizep - 1, id_high_value) - id_low_value;
			for (i = ID(first + tL) - id_low_value; i <= r; i += prime) marked[i] = 1;//printf("%d %d\n",id,RID(i));
			//printf("[%d %d][%d %d]\n",L,R,tL,tR);
		}
		//printf("[%d %d][%d %d]\n",id_low_value,id_high_value,L,R);
	}
	count = 0;
	size = id_high_value - id_low_value +1;
	for (i = 0; i < size; i++)
		if (!marked[i]) count++;//printf("%d\n",RID(id_low_value+i));
	//printf("!%10.6f\n",elapsed_time+MPI_Wtime());
	MPI_Reduce (&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
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
