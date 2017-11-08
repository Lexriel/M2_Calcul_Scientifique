# include "conf.h"
# include "fonctions.h"



int main(int argc, char* argv[])
{
// Message d'erreur s'il manque un paramètre en entrée
  if (argc < 4)
    {
      printf("Entrez au moins trois paramètres : 2 noms de fichier et la taille k des k-mots.\n");
      exit(1);
    }

  unsigned long *index_kmers1;
  unsigned long *index_kmers2;
  unsigned long *kmers_communs_GPU;
  unsigned long size1 = taille_fichier(argv[1]); // fonction donnant la taille d'un fichier
  unsigned long size2 = taille_fichier(argv[2]);
  unsigned long i, m, m_GPU;
  unsigned long k = atoi(argv[3]);
  unsigned long nb_kmers = power(4, k);
  int valeur = 0;
  char mot[k];
  char* tab_fichier1;
  char* tab_fichier2;

  srand(time(NULL)); // pour initialiser la fonction random_number

  index_kmers1 = (unsigned long*) calloc(nb_kmers, sizeof(unsigned long));
  index_kmers2 = (unsigned long*) calloc(nb_kmers, sizeof(unsigned long));
  kmers_communs_GPU = (unsigned long*) calloc(nb_kmers, sizeof(unsigned long));

// Stocke les fichiers rentrés en paramètres dans des tableaux
  stocker(argv[1], size1, tab_fichier1);
  stocker(argv[2], size2, tab_fichier2);

// Création de l'index de la séquence de référence (donnée en argv[1])
  for (i=0; i<k; i++)
    mot[i] = tab_fichier1[i];

  for (i=k; i<size1; i++)
    {
      valeur = code(mot, k);
      index_kmers1[valeur]++;
      decalage(mot, tab_fichier1[i], k);
    }

// Création de l'index de la séquence requête (donnée en argv[2])
  for (i=0; i<k; i++)
    mot[i] = tab_fichier2[i];

  for (i=k; i<size2; i++)
    {
      valeur = code(mot, k);
      index_kmers2[valeur]++;
      decalage(mot, tab_fichier2[i], k);
    }

// Affiche l'indexation de la séquence de référence en ne prenant pas les éléments apparaissant peu
//  display_array(index_kmers1, nb_kmers);
  m = find_kmers_CPU(index_kmers1, index_kmers2, nb_kmers);


// GPU ------------------------------------------------------
    int err;                            // error code returned from api calls
      
    size_t global;                      // global domain size for our calculation
    size_t max_local;
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context

    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input1, input2;              // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
    time_start();

    // Connect to a compute device
    // Create a compute context
    setupCL(&device_id, &context);

    // Create the compute program from the source buffer
    // Build the program executable
    program = create_and_buildCL("TP.kernel.c",     0, device_id, context, ""); // From source
    time_stop();
    time_start();
    
      
    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    handleCL(err, "clCreateCommandQueue");
 
 
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "find_kmers_GPU", &err);
    handleCL(err, "clCreateKernel");
 
    // Create the input and output arrays in device memory for our calculation
    input1 = clCreateBuffer(context, CL_MEM_READ_WRITE, nb_kmers*sizeof(unsigned long), NULL, &err);
    handleCL(err, "clCreateBuffer");
    input2 = clCreateBuffer(context, CL_MEM_READ_WRITE, nb_kmers*sizeof(unsigned long), NULL, &err);
    handleCL(err, "clCreateBuffer");
    output = clCreateBuffer(context, CL_MEM_READ_WRITE, nb_kmers*sizeof(unsigned long), NULL, &err);
    handleCL(err, "clCreateBuffer");

    // Events for profiling
    cl_event event_write, event_kernel, event_read;

    // Write our data set into the input arrays in device memory 
    err = clEnqueueWriteBuffer(commands, input1, CL_TRUE, 0, nb_kmers*sizeof(unsigned long), tab_fichier1, 0, NULL, &event_write);
    err = clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, nb_kmers*sizeof(unsigned long), tab_fichier2, 0, NULL, &event_write);
    handleCL(err, "clEnqueueWriteBuffer");

    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
    handleCL(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
    handleCL(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    handleCL(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 3, sizeof(unsigned long), &nb_kmers);
    handleCL(err, "clSetKernelArg");

    // Set the total number of work-items 
    global = nb_kmers ;

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_local), &max_local, NULL);
    handleCL(err, "clGetKernelWorkGroupInfo");
    
    // Set the number of work-items inside a work-group
    local = max_local ;


    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    printf("== Launch kernel (global: %ld, local: %ld)\n", (unsigned long) global, (unsigned long) local);
    printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = %d\n", (int) CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS); 
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &event_kernel);

    handleCL(err, "clEnqueueNDRangeKernel");
 
    // Wait for the command commands to get serviced before reading back results
    clFinish(commands);
 
    // Read back the results from the device to verify the output
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, nb_kmers*sizeof(unsigned long), kmers_communs_GPU, 0, NULL, &event_read);
    handleCL(err, "clEnqueueReadBuffer");

    m_GPU = sum(kmers_communs_GPU, nb_kmers);
    
    time_stop();
  
    // Profiling
    printf("Time for write:   %9.3f ms\n", profileCL(&event_write));
    printf("Time for kernel:  %9.3f ms\n", profileCL(&event_kernel));
    printf("Time for read:    %9.3f ms\n", profileCL(&event_read));

    // Shutdown and cleanup
    clReleaseMemObject(input1);
    clReleaseMemObject(input2);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

// ----------------------------------------------------------



// Affichage de données
  printf("taille du fichier 1 = %ld\n", size1);
  printf("taille du fichier 2 = %ld\n", size2);
  printf("nb_kmers = %ld\n", nb_kmers);
  printf("nombre de k_mers en communs entre le fichier 1 et le fichier 2 (CPU) : %ld.\n", m);
  printf("nombre de k_mers en communs entre le fichier 1 et le fichier 2 (GPU) : %ld.\n", m_GPU);
  printf("Pourcentage de correspondance entre les deux fichiers : %f.\n", (float) m / (float) maximum(size1-k, size2-k));

// Calcul et affichage des 10 k-mers les plus fréquents
  les10meilleurs(index_kmers1, nb_kmers, k);

  char zzzz[4];
  traduction(228, zzzz, 4);
  for (i=0; i<4; i++)
    printf("%c", zzzz[i]);

// Désallocation de tableaux
  free(tab_fichier1);
  free(tab_fichier2);
  free(index_kmers1);
  free(index_kmers2);

  return 0;
}
