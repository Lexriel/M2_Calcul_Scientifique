# include <mpi.h>
# include <string.h>
# include "date.h"

int main()
{
  int rank, target, nb;
  double begin, end;
  char date[100], date_received[100];
  time_t timestamp;
  struct tm *t;
  MPI_Status status;
  MPI_Request request;

  MPI_Init(NULL,NULL);
  begin = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nb);


/* ======================== Processus of rank 0 ============================ */

  if (rank == 0)
    {
      timestamp = time(NULL);
      t = localtime(&timestamp);

/* According to the date, we define the date differently (because of the
   suffix for the number of day).
   The function "sprintf" add the string in second argument into the first
   one, like a printf but for printing in the variable 'date', and not at the 
   screen. */
  if (t->tm_mday % 20 == 1)
     sprintf(date, "%s, %s the %d%s %d", day[t->tm_wday], month[t->tm_mon], t->tm_mday, "st", 1900 + t->tm_year);
  if (t->tm_mday == 31)
     sprintf(date, "%s, %s the %d%s %d", day[t->tm_wday], month[t->tm_mon], t->tm_mday, "st", 1900 + t->tm_year);
  if (t->tm_mday % 20 == 2)
     sprintf(date, "%s, %s the %d%s %d", day[t->tm_wday], month[t->tm_mon], t->tm_mday, "nd", 1900 + t->tm_year);
  if (t->tm_mday % 20 == 3)
     sprintf(date, "%s, %s the %d%s %d", day[t->tm_wday], month[t->tm_mon], t->tm_mday, "rd", 1900 + t->tm_year);
  else
     sprintf(date, "%s, %s the %d%s %d", day[t->tm_wday], month[t->tm_mon], t->tm_mday, "th", 1900 + t->tm_year);

// Sends the date to all the processes, except the one of rank 0 :
      for (target=1; target<nb; target++)
	MPI_Isend(date, 100, MPI_CHAR, target, 1, MPI_COMM_WORLD, &request);
    }

/* ============================================================================== */



/* ================ All the processes of rank different from 0 ================== */

  else
    {
// Reception of the date :
      MPI_Recv(date_received, 100, MPI_CHAR, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);

// Display of the date and execution time for each process different from 0 :
      printf("The processor 0 sent the date \"%s\" to the processor %d.\n", date_received, rank);
      end = MPI_Wtime();
      printf("Execution time of the processor %d = %f.\n", rank, end-begin);
    }

/* ============================================================================== */

  MPI_Finalize();

  return 0;
}
