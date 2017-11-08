# include <stdio.h>
# include <mpi.h>
# include <string.h>

int main()
{
  int i, rank, nb, nb_received, position;
  double begin, end;
  MPI_Status status;
  MPI_Request request;
  char t[1000];
  char letters[100];
  char letters_received[100];
  

  MPI_Init(NULL,NULL);
  begin = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nb);

/* ================ All the processes of rank different from 0 ================== */

  if (rank != 0)
    {

// According to the rank of the process, the rank in letters should be different :
      if (rank == 1)
         strcpy(letters, "\"one\"");
      if (rank == 2)
         strcpy(letters, "\"two\"");
      if (rank == 3)
         strcpy(letters, "\"three\"");
      if (rank == 4)
         strcpy(letters, "\"four\"");
      if (rank == 5)
         strcpy(letters, "\"five\"");
      if (rank == 6)
         strcpy(letters, "\"six\"");
      if (rank == 7)
         strcpy(letters, "\"seven\"");
      if (rank == 8)
         strcpy(letters, "\"height\"");
      if (rank == 9)
         strcpy(letters, "\"nine\"");
      if (rank == 10)
         strcpy(letters, "\"ten\"");
      if (rank == 11)
         strcpy(letters, "\"eleven\"");
      if (rank == 12)
         strcpy(letters, "\"twelve\"");
      if (rank == 13)
         strcpy(letters, "\"thirteen\"");
      if (rank == 14)
         strcpy(letters, "\"fourteen\"");
      if (rank == 15)
         strcpy(letters, "\"fifteen\"");
      if (rank > 15)
         strcpy(letters, "\"Sorry, I just know how to count until 15 !\"\n.");


// Creates a package to send containing different types of variable :
      position = 0;
      rank;
      MPI_Pack(&rank, 1, MPI_INT, t, 1000, &position, MPI_COMM_WORLD);
      MPI_Pack(&letters, 100, MPI_CHAR, t, 1000, &position, MPI_COMM_WORLD);


// Sends data in the pack previously made to the process 0 :
      MPI_Isend(t, position, MPI_PACKED, 0, 12, MPI_COMM_WORLD, &request);
      end = MPI_Wtime();
    }

/* ============================================================================== */



/* =========================== Process of rank 0 ================================ */

  else
    {

// Receives the packs, opens them and displays what they contain :
  for(i=1; i<nb; i++)
      {
        position = 0;
        MPI_Recv(t, 1000, MPI_PACKED, MPI_ANY_SOURCE, 12, MPI_COMM_WORLD, &status);	
        MPI_Unpack(t, 1000, &position, &nb_received, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(t, 1000, &position, &letters_received, 100, MPI_CHAR, MPI_COMM_WORLD);
        printf("The processor %d sent to the processor 0 the rank %d in number and %s in letters.\n", nb_received, nb_received, letters_received);
      }

// Displays the execution time of the program :
      end = MPI_Wtime();
      printf("Execution time of this program : %f.\n", end-begin);
    }

/* ============================================================================== */

  MPI_Finalize();

  return 0;
}
