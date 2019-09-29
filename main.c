#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <mpi.h>
#include "libs/bitmap.h"

// Convolutional Kernel Examples, each with dimension 3,
// gaussian kernel with dimension 5
// If you apply another kernel, remember not only to exchange
// the kernel but also the kernelFactor and the correct dimension.

int const sobelYKernel[] = {-1, -2, -1,
                             0,  0,  0,
                             1,  2,  1};
float const sobelYKernelFactor = (float) 1.0;

int const sobelXKernel[] = {-1, -0, -1,
                            -2,  0, -2,
                            -1,  0, -1 , 0};
float const sobelXKernelFactor = (float) 1.0;


int const laplacian1Kernel[] = {  -1,  -4,  -1,
                                 -4,  20,  -4,
                                 -1,  -4,  -1};
float const laplacian1KernelFactor = (float) 1.0;

int const laplacian2Kernel[] = { 0,  1,  0,
                                 1, -4,  1,
                                 0,  1,  0};
float const laplacian2KernelFactor = (float) 1.0;

int const laplacian3Kernel[] = { -1,  -1,  -1,
                                  -1,   8,  -1,
                                  -1,  -1,  -1};
float const laplacian3KernelFactor = (float) 1.0;


//Bonus Kernel:

int const gaussianKernel[] = { 1,  4,  6,  4, 1,
                               4, 16, 24, 16, 4,
                               6, 24, 36, 24, 6,
                               4, 16, 24, 16, 4,
                               1,  4,  6,  4, 1 };
float const gaussianKernelFactor = (float) 1.0 / 256.0;

// Helper function to swap bmpImageChannel pointers
void swapImageChannel(bmpImageChannel **one, bmpImageChannel **two) {
  bmpImageChannel *helper = *two;
  *two = *one;
  *one = helper;
}

// Apply convolutional kernel on image data
void applyKernel(unsigned char **out, unsigned char **in, unsigned int width, unsigned int height, int *kernel, unsigned int kernelDim, float kernelFactor) {
    unsigned int const kernelCenter = (kernelDim / 2);
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            int aggregate = 0;
            for (unsigned int ky = 0; ky < kernelDim; ky++) {
                int nky = kernelDim - 1 - ky;
                for (unsigned int kx = 0; kx < kernelDim; kx++) {
                    int nkx = kernelDim - 1 - kx;

                    int yy = y + (ky - kernelCenter);
                    int xx = x + (kx - kernelCenter);
                    if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
                        aggregate += in[yy][xx] * kernel[nky * kernelDim + nkx];
                }
            }
            aggregate *= kernelFactor;
            if (aggregate > 0) {
                out[y][x] = (aggregate > 255) ? 255 : aggregate;
            } else {
                out[y][x] = 0;
            }
        }
    }
}

void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;
    if (opt != 0) {
        out = stderr;
        if (optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s in.bmp out.bmp -i 10000\n", exec);
}

int main(int argc, char **argv) {
    /*
    Parameter parsing, don't change this!
   */
    unsigned int iterations = 1;
    char *output = NULL;
    char *input = NULL;
    int ret = 0;

    static struct option const long_options[] =  {
            {"help",       no_argument,       0, 'h'},
            {"iterations", required_argument, 0, 'i'},
            {0, 0, 0, 0}
    };

    static char const * short_options = "hi:";
    {
        char *endptr;
        int c;
        int option_index = 0;
        while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
            switch (c) {
                case 'h':
                    help(argv[0],0, NULL);
                    goto graceful_exit;
                case 'i':
                    iterations = strtol(optarg, &endptr, 10);
                    if (endptr == optarg) {
                        help(argv[0], c, optarg);
                        goto error_exit;
                    }
                    break;
                default:
                    abort();
            }
        }
    }

    if (argc <= (optind+1)) {
        help(argv[0],' ',"Not enough arugments");
        goto error_exit;
    }
    input = calloc(strlen(argv[optind]) + 1, sizeof(char));
    strncpy(input, argv[optind], strlen(argv[optind]));
    optind++;

    output = calloc(strlen(argv[optind]) + 1, sizeof(char));
    strncpy(output, argv[optind], strlen(argv[optind]));
    optind++;


    /*
     * End of Parameter parsing!
     */

    /*
     * ---Edited code under this line---
     */

    int comm_sz, my_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    bmpImage *image;
    bmpImageChannel *imageChannel;

    int *send_counts, *recv_counts,  *recv_displs, *send_displs;
    int im_XSZ, im_YSZ;

    int kernelDim = 3;
    int kernelRadi;

    // The image will only be opened in root rank.
    // It will be distributed to the at the very end of this if-block.
    if (my_rank == 0) {
        /*
        Create the BMP image and load it from disk.
        */
        image = newBmpImage(0,0);

        if (image == NULL) {
            fprintf(stderr, "Could not allocate new image!\n");
        }

        if (loadBmpImage(image, input) != 0) {
            fprintf(stderr, "Could not load bmp image '%s'!\n", input);
            freeBmpImage(image);
            goto error_exit;
        }


        // Create a single color channel image. It is easier to work just with one color
        imageChannel = newBmpImageChannel(image->width, image->height);
        if (imageChannel == NULL) {
            fprintf(stderr, "Could not allocate new image channel!\n");
            freeBmpImage(image);
            goto error_exit;
        }

        // Extract from the loaded image an average over all colors - nothing else than
        // a black and white representation
        // extractImageChannel and mapImageChannel need the images to be in the exact
        // same dimensions!
        // Other prepared extraction functions are extractRed, extractGreen, extractBlue
        if(extractImageChannel(imageChannel, image, extractAverage) != 0) {
            fprintf(stderr, "Could not extract image channel!\n");
            freeBmpImage(image);
            freeBmpImageChannel(imageChannel);
            goto error_exit;
        }
        im_YSZ = imageChannel->height;
        im_XSZ = imageChannel->width;

        kernelRadi = kernelDim/2;

        recv_counts = malloc(sizeof(int) * comm_sz);
        recv_displs = malloc(sizeof(int) * comm_sz);
        send_counts = malloc(sizeof(int) * comm_sz);
        send_displs = malloc(sizeof(int) * comm_sz);

        int send_sum = 0;
        int recv_sum = 0;

        for (int i = 0; i < comm_sz; i++) {
            recv_counts[i] = (im_YSZ / comm_sz)  * im_XSZ;

            // Every processed local image data is shifted by (+ kernelradius * width)
            // before it they are sent to root process.
            recv_displs[i] = kernelRadi * im_XSZ;
            send_displs[i] = send_sum;

            // First rank gets the remainder rows if they don't divide evenly.
            if (i == 0) {
                recv_counts[i] += (im_YSZ % comm_sz) * im_XSZ;

                // The local image data that will be processed is shifted by (- kernelradius * width).
                send_sum -= kernelRadi * im_XSZ;
                recv_displs[i] = 0;
            }

            send_sum += recv_counts[i];

            // Add additional rows beyond the local image, based on kernel radius.
            // Here we also take the top and bottom part of the image into consideration,
            // since they are the "real" edges of the image.
            if (i == 0 || i == comm_sz - 1) {
                send_counts[i] = recv_counts[i] + kernelRadi * im_XSZ;
            }
            else {
                send_counts[i] = recv_counts[i] + 2 * kernelRadi * im_XSZ;
            }

            printf("%d SEND\t counts:\t%d \tdisps:%d\n", i, send_counts[i]/im_XSZ, send_displs[i]/im_XSZ);
            printf("%d RECV\t counts:\t%d \tdisps:%d\n", i, recv_counts[i]/im_XSZ, recv_displs[i]/im_XSZ);

            // Send data to the respective processes, so that the relevant info to be used
            // in processing and sending the local image is readily at hand.
            MPI_Send(&recv_counts[i], 1, MPI_INT, i, 0, comm);
            MPI_Send(&recv_displs[i], 1, MPI_INT, i, 1, comm);
            MPI_Send(&send_counts[i], 1, MPI_INT, i, 2, comm);

            // Distribute the actual image data to the processes.
            int tag = 0;
            for (int k = recv_sum; k < recv_sum + recv_counts[i]; k++) {
                MPI_Send(&imageChannel->rawdata[k], 1, MPI_UNSIGNED_CHAR, i, tag, comm);
                tag++;
            }

            recv_sum += recv_counts[i];
        }
    }

    MPI_Bcast(&im_XSZ, 1, MPI_INT, 0, comm);
    MPI_Bcast(&im_YSZ, 1, MPI_INT, 0, comm);

    // Only to be used locally in each process.
    int local_n, local_displs, recv_n;
    MPI_Recv(&recv_n, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
    MPI_Recv(&local_displs, 1, MPI_INT, 0, 1, comm, MPI_STATUS_IGNORE);
    MPI_Recv(&local_n, 1, MPI_INT, 0, 2, comm, MPI_STATUS_IGNORE);
    int local_YSZ = local_n / im_XSZ;

    bmpImageChannel* local_imChannel = newBmpImageChannel(im_XSZ, local_YSZ);
    bmpImageChannel* local_procImChannel = newBmpImageChannel(im_XSZ, local_YSZ);

    // Receive the image data distributed by root proc.
    int tag = 0;
    int ydisp = local_displs / im_XSZ;
    int yrecv = recv_n / im_XSZ;
    printf("yrecv: %d,\tlocal_YSZ:%d\n", yrecv, local_YSZ);
    for (int y = ydisp; y < ydisp + yrecv; y++) {
        for (int x = 0; x < im_XSZ; x++) {
            MPI_Recv(&local_imChannel->data[y][x], local_n, MPI_UNSIGNED_CHAR, 0, tag, comm, MPI_STATUS_IGNORE);
            tag++;
        }
    }
    local_imChannel->rawdata = &(local_imChannel->data[0][0]);

    if (my_rank == 0) {
        for (int x = 0; x < im_XSZ; x++) {
            MPI_Send(&local_imChannel->data[local_YSZ - 1][], 1, MPI_UNSIGNED_CHAR, i, tag, comm);
        }
    }

    //Here we do the actual computation!
    printf("Proc %d: Here I go...\n", my_rank);
    for (unsigned int i = 0; i < iterations; i ++) {
        applyKernel(local_procImChannel->data,
                    local_imChannel->data,
                    im_XSZ,
                    local_YSZ,
                    (int *)laplacian1Kernel, kernelDim, laplacian1KernelFactor
//                    (int *)laplacian2Kernel, 3, laplacian2KernelFactor
//                    (int *)laplacian3Kernel, 3, laplacian3KernelFactor
//                    (int *)gaussianKernel, 5, gaussianKernelFactor
        );
        swapImageChannel(&local_procImChannel, &local_imChannel);
    }
    printf("Proc %d: ... And I'm done!\n", my_rank);

    freeBmpImageChannel(local_procImChannel);

    // Each process sends their own part of the processed image to the root proc.
    tag = 0;
    for (int k = local_displs; k < local_displs + recv_n; k++) {
        MPI_Send(&local_imChannel->rawdata[k], 1, MPI_UNSIGNED_CHAR, 0, tag, comm);
        tag++;
    }


    if (my_rank == 0) {
        // The root proc. receives it here.
        int sum = 0;
        for (int rank = 0; rank < comm_sz; rank++) {
            int tag = 0;
            for (int y = sum; y < sum + recv_counts[rank] / im_XSZ; y++) {
                for (int x = 0; x < im_XSZ; x++) {
                    MPI_Recv(&imageChannel->data[y][x], 1, MPI_UNSIGNED_CHAR, rank, tag, comm, MPI_STATUS_IGNORE);
                    tag++;
                }
            }
            sum += recv_counts[rank]/im_XSZ;
        }
        freeBmpImageChannel(local_imChannel);

        // Free arrays
        free(recv_counts);
        free(recv_displs);
        free(send_counts);
        free(send_displs);

        /*
        * ---Edited code over this line---
        */

        /*
        * ---The code below this line has mostly remained the same---
        */

        for (unsigned int i = 0; i < imageChannel->height; i++) {
            imageChannel->data[i] = &(imageChannel->rawdata[i * imageChannel->width]);
        }
        // Map our single color image back to a normal BMP image with 3 color channels
        // mapEqual puts the color value on all three channels the same way
        // other mapping functions are mapRed, mapGreen, mapBlue
        if (mapImageChannel(image, imageChannel, mapEqual) != 0) {
            fprintf(stderr, "Could not map image channel!\n");
            freeBmpImage(image);
            freeBmpImageChannel(imageChannel);
            goto error_exit;
        }
        freeBmpImageChannel(imageChannel);

        //Write the image back to disk
        if (saveBmpImage(image, output) != 0) {
            fprintf(stderr, "Could not save output to '%s'!\n", output);
            freeBmpImage(image);
            goto error_exit;
        };
    }

    // Shut down MPI
    MPI_Finalize();

    graceful_exit:
    ret = 0;
    error_exit:
    if (input)
        free(input);
    if (output)
        free(output);
    return ret;
};
