
////////////////////////////////////////////////////////////////////////////////

 
void setupCL(cl_device_id *device_id, cl_context *context)
{
  cl_device_type device_type;

#ifdef TYPE_ALL
  device_type = CL_DEVICE_TYPE_ALL;
#else
  device_type = CL_DEVICE_TYPE_GPU;
#endif



  char pbuf[1024];
  cl_int status = 0;

  /*
   * Have a look at the available platforms 
   */

  cl_uint nb_platforms;
  cl_platform_id platform = NULL;
  status = clGetPlatformIDs(0, NULL, &nb_platforms);
  handleCL(status, "clGetPlatformIDs");
  printf("==== %d platforms\n", nb_platforms);

  if (nb_platforms == 0)
    exit(1);

  cl_platform_id* platforms =  (cl_platform_id*) malloc(sizeof(cl_platform_id) * nb_platforms);
  status = clGetPlatformIDs(nb_platforms, platforms, NULL);
  handleCL(status, "clGetPlatformIDs");

  unsigned p ;
  for (p = 0; p < nb_platforms; p++) 
    {
      platform = platforms[p];
      printf("==== platform %d: ", p);

      unsigned j ;
      for (j = 0; j<5; j++)
	{	      
	  int info ;
	  switch (j)
	    {
	    case 0: info = CL_PLATFORM_NAME ; break ;
	    case 1: info = CL_PLATFORM_VENDOR ; break ;
	    case 2: info = CL_PLATFORM_PROFILE ; break ;
	    case 3: info = CL_PLATFORM_VERSION ; break ;
	    case 4: info = CL_PLATFORM_EXTENSIONS ; break ;
	    }
	  status = clGetPlatformInfo(platform, info,
				     sizeof(pbuf), pbuf,
				     NULL);
	  handleCL(status, "clGetPlaformInfo");
	  printf("%s ", pbuf);
	}
      
      printf("\n");

      // if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) break ;


      // Number of devices
      cl_uint nb_devices ;
      status = clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 0, NULL, &nb_devices);
      handleCL(status, "clGetDeviceIDs");

      printf("     platform with %d devices\n\n", nb_devices);
      cl_device_id* devices = (cl_device_id*) malloc(sizeof(cl_device_id) * nb_devices);

      // List of devices
      //   int gpu = 1;gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU
      status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, nb_devices, devices, NULL);
      handleCL(status, "clGetDeviceIDs");

      unsigned i;
      for(i = 0; i < nb_devices; ++i ) 
	{  
	  cl_ulong ret ;
	  printf("== Device %d (id: 0x%lx)\n", i, (long unsigned) devices[i]); 
	  
	  // See OpenCL reference 4.2
	  clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(pbuf), &pbuf, NULL);
	  printf("       %s ", pbuf);
	  
	  clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(pbuf), &pbuf, NULL);
	  printf("(%s), ", pbuf);

	  clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(pbuf), &pbuf, NULL);
	  printf("%s\n", pbuf);
	  
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ret), &ret, NULL);
	  printf("       max_compute_units \t %ld\n", (long) ret);

	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(ret), &ret, NULL);
	  printf("       max_workgroup_size \t %ld\n", (long) ret);
	  
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(ret), &ret, NULL);
	  printf("       max_clock_freq    \t %ld MHz\n", (long) ret);

	  clGetDeviceInfo(devices[i], CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(ret), &ret, NULL);
	  printf("       mem alignment     \t %ld b, ", (long) ret);
	  clGetDeviceInfo(devices[i], CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(ret), &ret, NULL);
	  printf("%ld B\n", (long) ret);


	  clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ret), &ret, NULL);
	  printf("       mem    global     \t %ld MB ", (long) ret / (1 << 20));

	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(ret), &ret, NULL);
	  printf("(max alloc: %ld MB)\n", (long) ret / (1 << 20));


	  clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(ret), &ret, NULL);
	  printf("              cache      \t %ld kB ", (long) ret / (1 << 10));

	  if (ret)
	    {
	      clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(ret), &ret, NULL);
	      printf("(line: %ld B)\n", (long) ret);
	    }
	  else
	    printf("\n");

	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(ret), &ret, NULL);
	  printf("              constant    \t %ld kB ", (long) ret / (1 << 10));
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(ret), &ret, NULL);
	  printf("(max args: %ld)\n", (long) ret);

	  clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ret), &ret, NULL);
	  printf("              local     \t %ld kB \n", (long) ret / (1 << 10));

	  // segfault ?
	  //cl_device_local_mem_type r;
	  //clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_TYPE, sizeof(r), &r, NULL);
	  //printf("(%s)\n", (r == CL_LOCAL) ? : "local", "global");

	  clGetDeviceInfo(devices[i], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(ret), &ret, NULL);
	  printf("       profiling timer res \t %ld ns\n", (long) ret);

	  clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, sizeof(pbuf), &pbuf, NULL);
	  printf("       %s\n\n", pbuf);

	  //clGetDeviceInfo(devices[i], CL_DEVICE_COMPILER_AVAILABLE, sizeof(ret), &ret, NULL);
	  //printf("       compiler         \t%d\n\n", ret);
	}
    }
  // Select the platform
  platform = platforms[0];

  /* from ATI
   * If we could find our platform, use it. Otherwise pass a NULL and get whatever the
   * implementation thinks we should be using.
   */
  cl_context_properties cps[3] = 
	{
	  CL_CONTEXT_PLATFORM, 
	  (cl_context_properties)platform, 
	  0
	};

  /* Use NULL for backward compatibility */
  cl_context_properties* cprops = (NULL == platform) ? NULL : cps;


  // Select this device (FIXME: does not work ?)
  // device_id = &devices[0] ; 


  ///// Or create directly 1 context from type
  // *context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
  // handleCL(status, "clCreateContextFromType");
  // return ;

  ///// Or select directly 1 GPU device:  (works on apple)
  status = clGetDeviceIDs(platform, device_type, 1, device_id, NULL);
  handleCL(status, "clGetDeviceIDs");
  

  // Creating the context
  printf("== Creating context on device 0x%lx\n", (long unsigned) *device_id);  
  *context = clCreateContext(cprops, 1, device_id, NULL, NULL, &status);
  handleCL(status, "clCreateContext");

  printf("\n");
}


float profileCL(cl_event* event) // microseconds
{
  cl_int status = 0;
  cl_ulong start, end;
  
  status = clWaitForEvents(1, event);
  handleCL(status, "clWaitForEvents");

  status = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
				   &start, NULL);
  handleCL(status, "clGetEventProfilingInfo -- START");

  status = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END,sizeof(cl_ulong), 
					  &end,NULL);
  handleCL(status, "clGetEventProfilingInfo -- END");

  // clReleaseEvent(*event);

  return (float) (end - start) / 1E6 ;
}




//////////////////////

#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
	

// suppose one context with only one device...
cl_program create_and_buildCL(const char *kernel_filename, int is_binary, cl_device_id device_id,
				cl_context context, const char *compile_flags)
{
  printf("== Creating and building the program...\n");
  int err ;
  int save_binary = 1 ;


  char kernel_binary_filename[100] ;
  sprintf(kernel_binary_filename, "%s.bin", kernel_filename);

  // Check binary date/time, too see if we need compilation
  // TODO: update with compile_flags: CL_PROGRAM_BUILD_OPTIONS
  struct stat buf;		

  if (!is_binary && !stat(kernel_binary_filename, &buf))
    {	
      time_t sec_kernel_binary = buf.st_mtime ; // .tv_sec  ;

      if (!stat(kernel_filename, &buf))
	{
	  time_t sec_diff = sec_kernel_binary - buf.st_mtime ; //.tv_sec ;
	  
	  if (sec_diff >= 0)
	    {
	      printf("    <== Using binary (%lds newer than source '%s')\n", sec_diff, kernel_filename);
	      is_binary = 1 ;
	      save_binary = 0 ;
	      kernel_filename = kernel_binary_filename ;
	    }
	}
    }
  
  // Open the file
  printf("    <== kernel: %s\n", kernel_filename);
  FILE *f = fopen(kernel_filename, "r");

  if (!f)
    {
      printf("    ### Error opening the file %s \n", kernel_filename);
      exit(1) ;
    }

  // Get the file size
  fseek(f, 0L, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0L, SEEK_SET);
  printf("    <== kernel: %ld bytes\n", file_size);

  // Read the file
  char *kernel = (char *) malloc (file_size * sizeof (char));
  fread(kernel, sizeof(char), file_size, f);
  
  // printf("Kernel source code: \n %s\n", kernel);



  // Create the program
  cl_program program ;

//  if (!is_binary)
    {
      // From source
      program = clCreateProgramWithSource(context, 1, (const char **) &kernel, NULL, &err);
      handleCL(err, "clCreateProgramWithSource");
    }
/*  else
    {
      // From binary
      int binary_status;
      program = clCreateProgramWithBinary (context, 1, &device_id, 
					   &file_size, (const unsigned char **) &kernel, 
					   &binary_status, &err);
      handleCL(err, "clCreateProgramWithBinary");
      handleCL(binary_status, "clCreateProgramWithBinary / binary_status");
    }
*/
  // Build the program executable
  err = clBuildProgram(program, 0, NULL, compile_flags, NULL, NULL);


  if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[100008];
      
      statusCL(err, "clBuildProgram");

      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

      printf("\n\n====> %s\n", buffer);
      handleCL(err, "clBuildInfo");
      exit(1);
    }

  // Saving the binary
  size_t binary_size ;
  err = clGetProgramInfo (program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
  handleCL(err, "clGetProgramInfo / CL_PROGRAM_BINARY_SIZES");

  printf("    ==> binary: %d bytes\n", (int) binary_size);
#ifndef ATI
  if (save_binary)
    {
      printf("    ==> binary: %s \n", kernel_binary_filename);
      char *kernel_binary = (char*) malloc(binary_size);
      err = clGetProgramInfo (program, CL_PROGRAM_BINARIES, binary_size, &kernel_binary, NULL);
      handleCL(err, "clGetProgramInfo / CL_PROGRAM_BINARIES");
      
      {
	FILE *f = fopen(kernel_binary_filename, "w");
	fwrite(kernel_binary, 1, binary_size, f);
	fclose(f);
      }
    }
#endif
 
  // ok

  printf("\n");
  return program ;
}




///////// time

//time_taken_milis = (int)((clock()-milistart)*1E3/CLOCKS_PER_SEC);
