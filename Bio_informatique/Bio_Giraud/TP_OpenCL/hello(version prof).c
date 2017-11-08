// OpenCL Tutorial
 
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef OPENCL_H
#include <OpenCL/opencl.h> // Apple
#else
#include <CL/cl.h>  // ATI, nvidia
#endif

#include "handle.c"
#include "setup.c"
#include "time_tools.c"

#define DATA_SIZE (256*1024)
 

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
      
    float data[DATA_SIZE];              // original data set given to device
    float results[DATA_SIZE];           // results returned from device
    unsigned int correct;               // number of correct results returned
 
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context

    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
    time_start();

    // Connect to a compute device
    // Create a compute context
    setupCL(&device_id, &context);

    // Create the compute program from the source buffer
    // Build the program executable
    program = create_and_buildCL("hello.kernel.c",     0, device_id, context, ""); // From source
    //program = create_and_buildCL("hello.kernel.c.bin", 1, device_id, context, ""); // From binary
    time_stop();
    time_start();
    
    // Fill our data set with random float values
    //
    int i = 0;
    unsigned int count = DATA_SIZE;
    for(i = 0; i < count; i++)
        data[i] = rand() / (float)RAND_MAX;
    
      
    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    handleCL(err, "clCreateCommandQueue");
 
 
    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "square", &err);
    handleCL(err, "clCreateKernel");
 
    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
    handleCL(err, "clCreateBuffer");
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
    handleCL(err, "clCreateBuffer");

    // Events for profiling
    cl_event event_write, event_kernel, event_read;

    
    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data,
			       0, NULL, &event_write);
    handleCL(err, "clEnqueueWriteBuffer");

 
    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    handleCL(err, "clSetKernelArg");

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    handleCL(err, "clGetKernelWorkGroupInfo");
 
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 
				 0, NULL, &event_kernel);
    handleCL(err, "clEnqueueNDRangeKernel");
 
    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);
 
    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * count, results, 
			      0, NULL, &event_read);    
    handleCL(err, "clEnqueueReadBuffer");

    
    time_stop();
    // Validate our results
    //
    correct = 0;
    for(i = 0; i < count; i++)
    {
        if(results[i] == data[i] * data[i])
            correct++;
    }
    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values!\n", correct, count);
    if (correct < count)
      printf("   >>> WRONG OUTPUT <<<\n");
    
    // Profiling
    printf("Time for write:   %9.3f ms\n", profileCL(&event_write));
    printf("Time for kernel:  %9.3f ms\n", profileCL(&event_kernel));
    printf("Time for read:    %9.3f ms\n", profileCL(&event_read));

    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
 
    return 0;
}

