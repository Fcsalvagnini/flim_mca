#include "ift.h"
#include "include/ca.h"

#ifdef IFT_GPU
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <math_functions.h>
#endif

#define DEBUG 0
#define SAVE_FREQ 10

#ifdef IFT_GPU
// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, error, \
                    cudaGetErrorString(error), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device functions
__device__ float d_similarity_func(
    float *orig_vals, int *label, int p, int q, int label_evolving,
    int gray, int n_bands, int n_pixels
) {
    float g;
    if (!gray) {
        // Get pixel values for p and q across all bands
        float y_p[3], y_q[3];
        for (int b = 0; b < 3; b++) {
            y_p[b] = orig_vals[p + b * n_pixels];
            y_q[b] = orig_vals[q + b * n_pixels];
        }

        // Calculate LAB distance
        float lab_dist = sqrtf((y_p[0] - y_q[0]) * (y_p[0] - y_q[0]) +
                               (y_p[1] - y_q[1]) * (y_p[1] - y_q[1]) +
                               (y_p[2] - y_q[2]) * (y_p[2] - y_q[2]));

        const float THRESHOLD = 0.20f;
        const float BETA_STRONG = 0.6f;
        const float BETA_NORMAL = 1.0f;
        float beta;

        if ((lab_dist < THRESHOLD) && (label[q] == label_evolving) && (label_evolving == 1)) {
            beta = BETA_STRONG;
        } else {
            beta = BETA_NORMAL;
        }
        g = expf(-beta * lab_dist);
    } else {
        float y_p = orig_vals[p];
        float y_q = orig_vals[q];
        float beta;
        float dist = fabsf(y_p - y_q);

        if ((y_p > y_q) && (label[q] == label_evolving) && (label_evolving == 1)) {
            beta = 0.6f;
        } else {
            beta = 1.0f;
        }
        g = expf(-beta * dist);
    }

    return g;
}

// CUDA kernel for evolving state
__global__ void evolve_state_kernel(
    float *orig_vals, float *strength, float *new_strength,
    int *label, int *new_label, int *adj_dx, int *adj_dy, int *adj_dz,
    int n_adj, int label_evolving, int is_brain, int n_bands,
    int xsize, int ysize, int zsize, int n_pixels
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p >= n_pixels) return;

    float q_max = strength[p];
    int best_label = label[p];

    // Get voxel coordinates
    int z = p / (xsize * ysize);
    int y = (p % (xsize * ysize)) / xsize;
    int x = p % xsize;

    // Check all adjacent voxels
    for (int adj = 1; adj < n_adj; adj++) {
        int vx = x + adj_dx[adj];
        int vy = y + adj_dy[adj];
        int vz = z + adj_dz[adj];

        // Check if neighbor is valid
        if (vx >= 0 && vx < xsize && vy >= 0 && vy < ysize && vz >= 0 && vz < zsize) {
            int q = vz * xsize * ysize + vy * xsize + vx;

            float q_aux = d_similarity_func(orig_vals, label, p, q, label_evolving, 
                                           is_brain, n_bands, n_pixels) * strength[q];

            if (q_aux > q_max) {
                best_label = label[q];
                q_max = q_aux;
            }
        }
    }

    new_strength[p] = q_max;
    new_label[p] = best_label;
}

// CUDA kernel for computing difference
__global__ void compute_diff_kernel(
    float *new_strength, float *old_strength, float *diff, int n_pixels
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p >= n_pixels) return;

    diff[p] = sqrtf((new_strength[p] - old_strength[p]) *
                    (new_strength[p] - old_strength[p]));
}

// CUDA kernel for probability map
__global__ void compute_prob_map_kernel(
    float *fg_strength, float *bg_strength, float *prob_map, int n_pixels
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p >= n_pixels) return;

    const float EPS = 1e-9f;
    prob_map[p] = logf(bg_strength[p] + EPS) /
                  (logf(bg_strength[p] + EPS) + logf(fg_strength[p] + EPS));
}

// CUDA kernel for initialization
__global__ void init_strengths_kernel(
    float *orig_vals, int *saliency, float *fg_strength, float *bg_strength,
    int *label, int *brain_mask, int *aux, int is_brain, int i_max, int n_pixels
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p >= n_pixels) return;

    if (is_brain) {
        fg_strength[p] = (float)saliency[p] / 255.0f;
        bg_strength[p] = (orig_vals[p] == 0) ? 1.0f : 0.0f;
        label[p] = (saliency[p] != 0) ? 1 : 0;
        if (brain_mask) brain_mask[p] = (orig_vals[p] != 0) ? 1 : 0;
    } else {
        fg_strength[p] = (float)saliency[p] / (float)i_max;
        bg_strength[p] = (float)aux[p] / (float)i_max;
        label[p] = (saliency[p] != 0) ? 1 : 0;
    }
}

// GPU memory structure
typedef struct {
    float *d_orig_vals;
    float *d_fg_strength;
    float *d_bg_strength;
    float *d_new_fg_strength;
    float *d_new_bg_strength;
    int *d_label;
    int *d_new_label;
    int *d_saliency;
    int *d_brain_mask;
    int *d_adj_dx;
    int *d_adj_dy;
    int *d_adj_dz;
    float *d_diff;
    float *d_prob_map;
    int n_pixels;
    int n_bands;
} GPUMemory;

// Allocate GPU memory
GPUMemory* allocate_gpu_memory(iftCAModel *ca_model) {
    GPUMemory *gpu_mem = (GPUMemory*)calloc(1, sizeof(GPUMemory));

    gpu_mem->n_pixels = ca_model->orig->n;
    gpu_mem->n_bands = ca_model->orig->m;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_orig_vals,
                         gpu_mem->n_pixels * gpu_mem->n_bands * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_fg_strength, gpu_mem->n_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_bg_strength, gpu_mem->n_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_new_fg_strength, gpu_mem->n_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_new_bg_strength, gpu_mem->n_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_label, gpu_mem->n_pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_new_label, gpu_mem->n_pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_saliency, gpu_mem->n_pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_diff, gpu_mem->n_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_prob_map, gpu_mem->n_pixels * sizeof(float)));

    if (ca_model->brain_mask) {
        CUDA_CHECK(cudaMalloc(&gpu_mem->d_brain_mask, gpu_mem->n_pixels * sizeof(int)));
    }

    // Allocate adjacency relation
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_adj_dx, ca_model->neighborhood->n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_adj_dy, ca_model->neighborhood->n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_mem->d_adj_dz, ca_model->neighborhood->n * sizeof(int)));

    return gpu_mem;
}

// Free GPU memory
void free_gpu_memory(GPUMemory *gpu_mem) {
    if (!gpu_mem) return;

    cudaFree(gpu_mem->d_orig_vals);
    cudaFree(gpu_mem->d_fg_strength);
    cudaFree(gpu_mem->d_bg_strength);
    cudaFree(gpu_mem->d_new_fg_strength);
    cudaFree(gpu_mem->d_new_bg_strength);
    cudaFree(gpu_mem->d_label);
    cudaFree(gpu_mem->d_new_label);
    cudaFree(gpu_mem->d_saliency);
    cudaFree(gpu_mem->d_brain_mask);
    cudaFree(gpu_mem->d_adj_dx);
    cudaFree(gpu_mem->d_adj_dy);
    cudaFree(gpu_mem->d_adj_dz);
    cudaFree(gpu_mem->d_diff);
    cudaFree(gpu_mem->d_prob_map);

    free(gpu_mem);
}

// Transfer data to GPU
void transfer_to_gpu(iftCAModel *ca_model, GPUMemory *gpu_mem) {
    // Copy original image values
    float *orig_vals = (float*)malloc(gpu_mem->n_pixels * gpu_mem->n_bands * sizeof(float));
    for (int b = 0; b < gpu_mem->n_bands; b++) {
        for (int p = 0; p < gpu_mem->n_pixels; p++) {
            orig_vals[p + b * gpu_mem->n_pixels] = ca_model->orig->val[p][b];
        }
    }
    CUDA_CHECK(cudaMemcpy(gpu_mem->d_orig_vals, orig_vals,
                         gpu_mem->n_pixels * gpu_mem->n_bands * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(orig_vals);

    // Copy other data
    CUDA_CHECK(cudaMemcpy(gpu_mem->d_fg_strength, ca_model->fg_strength->val,
                         gpu_mem->n_pixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_mem->d_bg_strength, ca_model->bg_strength->val,
                         gpu_mem->n_pixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_mem->d_label, ca_model->label->val,
                         gpu_mem->n_pixels * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_mem->d_saliency, ca_model->saliency->val,
                         gpu_mem->n_pixels * sizeof(int), cudaMemcpyHostToDevice));

    if (ca_model->brain_mask) {
        CUDA_CHECK(cudaMemcpy(gpu_mem->d_brain_mask, ca_model->brain_mask->val,
                             gpu_mem->n_pixels * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Copy adjacency relation
    int *adj_dx = (int*)malloc(ca_model->neighborhood->n * sizeof(int));
    int *adj_dy = (int*)malloc(ca_model->neighborhood->n * sizeof(int));
    int *adj_dz = (int*)malloc(ca_model->neighborhood->n * sizeof(int));

    for (int i = 0; i < ca_model->neighborhood->n; i++) {
        adj_dx[i] = ca_model->neighborhood->dx[i];
        adj_dy[i] = ca_model->neighborhood->dy[i];
        adj_dz[i] = ca_model->neighborhood->dz[i];
    }

    CUDA_CHECK(cudaMemcpy(gpu_mem->d_adj_dx, adj_dx,
                         ca_model->neighborhood->n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_mem->d_adj_dy, adj_dy,
                         ca_model->neighborhood->n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_mem->d_adj_dz, adj_dz,
                         ca_model->neighborhood->n * sizeof(int), cudaMemcpyHostToDevice));

    free(adj_dx);
    free(adj_dy);
    free(adj_dz);
}

// GPU-accelerated evolution
double evolve_state_gpu(
    int *epoch, iftCAModel *ca_model, GPUMemory *gpu_mem,
    int label_evolving, int is_brain
) {
    int block_size = 256;
    int grid_size = (gpu_mem->n_pixels + block_size - 1) / block_size;

    // Launch evolution kernel
    evolve_state_kernel<<<grid_size, block_size>>>(
        gpu_mem->d_orig_vals,
        label_evolving ? gpu_mem->d_fg_strength : gpu_mem->d_bg_strength,
        label_evolving ? gpu_mem->d_new_fg_strength : gpu_mem->d_new_bg_strength,
        gpu_mem->d_label, gpu_mem->d_new_label,
        gpu_mem->d_adj_dx, gpu_mem->d_adj_dy, gpu_mem->d_adj_dz,
        ca_model->neighborhood->n, label_evolving, is_brain, gpu_mem->n_bands,
        ca_model->orig->xsize, ca_model->orig->ysize, ca_model->orig->zsize,
        gpu_mem->n_pixels
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute difference
    compute_diff_kernel<<<grid_size, block_size>>>(
        label_evolving ? gpu_mem->d_new_fg_strength : gpu_mem->d_new_bg_strength,
        label_evolving ? gpu_mem->d_fg_strength : gpu_mem->d_bg_strength,
        gpu_mem->d_diff, gpu_mem->n_pixels
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sum reduction for difference (simplified version)
    float *h_diff = (float*)malloc(gpu_mem->n_pixels * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_diff, gpu_mem->d_diff,
                         gpu_mem->n_pixels * sizeof(float), cudaMemcpyDeviceToHost));

    double dist = 0;
    for (int i = 0; i < gpu_mem->n_pixels; i++) {
        dist += h_diff[i];
    }
    dist = dist / gpu_mem->n_pixels;
    free(h_diff);

    // Swap pointers
    if (label_evolving) {
        float *temp = gpu_mem->d_fg_strength;
        gpu_mem->d_fg_strength = gpu_mem->d_new_fg_strength;
        gpu_mem->d_new_fg_strength = temp;
    } else {
        float *temp = gpu_mem->d_bg_strength;
        gpu_mem->d_bg_strength = gpu_mem->d_new_bg_strength;
        gpu_mem->d_new_bg_strength = temp;
    }

    int *temp_label = gpu_mem->d_label;
    gpu_mem->d_label = gpu_mem->d_new_label;
    gpu_mem->d_new_label = temp_label;

    (*epoch)++;

    return dist;
}

// GPU-accelerated CA evolution
void evolve_ca_model_gpu(iftCAModel *ca_model, GPUMemory *gpu_mem, double dist, int is_brain) {
    timer *tstart, *tend;
    int epoch = 0;

  if (DEBUG) {
    iftImage *l_debug = NULL;
    // SAVE FG
    iftImage *fg_debug = iftFImageToImage(ca_model->fg_strength, 255);
    iftWriteImageByExt(fg_debug, "debug/fg_debug.png");
    iftDestroyImage(&fg_debug);

    // Save BG
    iftImage *bg_debug = iftFImageToImage(ca_model->bg_strength, 255);
    iftWriteImageByExt(bg_debug, "debug/bg_debug.png");
    iftDestroyImage(&bg_debug);

    // Save Label
    l_debug = iftCreateImageFromImage(ca_model->label);
    for (size_t p = 0; p < ca_model->label->n; p++) {
      l_debug->val[p] = ca_model->label->val[p] * 255;
    }
    iftWriteImageByExt(l_debug, "debug/label_debug.png");
    iftDestroyImage(&l_debug);
  }

    tstart = iftTic();
    while (ca_model->fg_dist > dist) {
        ca_model->fg_dist = evolve_state_gpu(&epoch, ca_model, gpu_mem, 1, is_brain);
        printf("\r[INFO] Evolving fg strength epoch %d dist %lf", epoch, ca_model->fg_dist);
        fflush(stdout);
    }
    tend = iftToc();
    printf("\n[INFO] Evolution took %s\n", iftFormattedTime(iftCompTime(tstart, tend)));

    epoch = 0;
    tstart = iftTic();
    while (ca_model->bg_dist > dist) {
        ca_model->bg_dist = evolve_state_gpu(&epoch, ca_model, gpu_mem, 0, is_brain);
        printf("\r[INFO] Evolving bg strength epoch %d dist %lf", epoch, ca_model->bg_dist);
        fflush(stdout);
    }
    tend = iftToc();
    printf("\n[INFO] Evolution took %s\n\n", iftFormattedTime(iftCompTime(tstart, tend)));

    // Copy results back to CPU
    CUDA_CHECK(cudaMemcpy(ca_model->fg_strength->val, gpu_mem->d_fg_strength,
                         gpu_mem->n_pixels * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ca_model->bg_strength->val, gpu_mem->d_bg_strength,
                         gpu_mem->n_pixels * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ca_model->label->val, gpu_mem->d_label,
                         gpu_mem->n_pixels * sizeof(int), cudaMemcpyDeviceToHost));
}

// GPU-accelerated probability map
iftFImage *GetProbMapGPU(iftCAModel *ca_model, GPUMemory *gpu_mem) {
    int block_size = 256;
    int grid_size = (gpu_mem->n_pixels + block_size - 1) / block_size;

    compute_prob_map_kernel<<<grid_size, block_size>>>(
        gpu_mem->d_fg_strength, gpu_mem->d_bg_strength,
        gpu_mem->d_prob_map, gpu_mem->n_pixels
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    iftFImage *prob_map = iftCreateFImageFromFImage(ca_model->fg_strength);
    CUDA_CHECK(cudaMemcpy(prob_map->val, gpu_mem->d_prob_map,
                         gpu_mem->n_pixels * sizeof(float), cudaMemcpyDeviceToHost));

    return prob_map;
}
#endif // IFT_GPU

iftCAModel *InitializeCA(
    char *input_image_path, char *saliency_path, float neighborhood_size,
    int mode, int is_brain
) {
    iftAdjRel *A;
    iftCAModel *ca_model = (iftCAModel *)calloc(1, sizeof(iftCAModel));

    // Loads intensity image
    iftImage *orig = NULL;
    iftImage *saliency = NULL;
    // If orig image greater then saliency, and mode 0, subsample orig image
    if (mode == 0) {
        printf("[INFO] Subsampling Orig Image\n");
        iftImage *tmp = iftReadImageByExt(input_image_path);
        saliency = iftReadImageByExt(saliency_path);
        if (
            tmp->xsize == saliency->xsize && tmp->ysize == saliency->ysize
            && tmp->zsize == saliency->zsize
        ) {
            orig = tmp;
        } else {
            float sx = (float)saliency->xsize / (float)tmp->xsize;
            float sy = (float)saliency->ysize / (float)tmp->ysize;
            float sz = (float)saliency->zsize / (float)tmp->zsize;
            if (sz == 1) {
                orig = iftInterp2D(tmp, sx, sy);
            } else {
                orig = iftInterp(tmp, sx, sy, sz);
            }
            iftDestroyImage(&tmp);
        }
    }
    else if (mode == 1) {
        printf("[INFO] Oversampling Saliency Image\n");
        orig = iftReadImageByExt(input_image_path);
        iftImage *tmp = iftReadImageByExt(saliency_path);
        if (
            tmp->xsize == orig->xsize && tmp->ysize == orig->ysize
            && tmp->zsize == orig->zsize
        ) {
            saliency = tmp;
        } else {
            float sx = (float)orig->xsize / (float)tmp->xsize;
            float sy = (float)orig->ysize / (float)tmp->ysize;
            float sz = (float)orig->zsize / (float)tmp->zsize;
            if (sz == 1) {
                saliency = iftInterp2D(tmp, sx, sy);
            } else {
                saliency = iftInterp(tmp, sx, sy, sz);
            }
            iftDestroyImage(&tmp);
        }
    }

    // Reads saliency image, which will be used to initialize foreground seeds
    ca_model->saliency = saliency;
    ca_model->fg_strength = iftCreateFImage(orig->xsize, orig->ysize, orig->zsize);
    ca_model->bg_strength = iftCreateFImage(orig->xsize, orig->ysize, orig->zsize);
    ca_model->label = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);

    if(iftIs3DImage(orig) || is_brain) {
        ca_model->i_max = 65535;
        if (iftIs3DImage(orig)) {
            ca_model->neighborhood = iftSpheric(neighborhood_size);
        } else {
            ca_model->neighborhood = iftRectangular(neighborhood_size, neighborhood_size);
        }
        printf("N Neighbours is %d\n", ca_model->neighborhood->n);
    }
    else {
        ca_model->i_max = 255;
        A = iftCircular(10);
        ca_model->neighborhood = iftRectangular(neighborhood_size, neighborhood_size);
    }

    iftImage *aux = NULL;
    // Brain initialization
    if (is_brain) {
        ca_model->orig = iftImageToMImage(orig, GRAY_CSPACE);
        ca_model->brain_mask = iftCreateImageFromImage(orig);
        for (size_t p = 0; p < ca_model->orig->n; p++) {
            ca_model->orig->val[p][0] = ca_model->orig->val[p][0] / (float) ca_model->i_max;
            ca_model->fg_strength->val[p] = (float)ca_model->saliency->val[p] / 255;
            ca_model->brain_mask->val[p] = (orig->val[p] != 0);
            ca_model->bg_strength->val[p] = (orig->val[p] == 0);
            ca_model->label->val[p] = (ca_model->saliency->val[p] != 0);
        }
    }
    else {
        iftImage *dilated_s = iftDilate(ca_model->saliency, A, NULL);
        aux = iftComplement(dilated_s);
        iftDestroyImage(&dilated_s);
        ca_model->orig = iftImageToMImage(orig, LABNorm2_CSPACE);
        for (size_t p = 0; p < ca_model->orig->n; p++) {
            ca_model->fg_strength->val[p] = (float)ca_model->saliency->val[p] / ca_model->i_max;
            ca_model->bg_strength->val[p] = (float)aux->val[p]  / ca_model->i_max;
            ca_model->label->val[p] = (ca_model->saliency->val[p] != 0);
        }
        iftDestroyAdjRel(&A);
    }

    ca_model->fg_dist = IFT_INFINITY_DBL;
    ca_model->bg_dist = IFT_INFINITY_DBL;

    // Releases resources
    iftDestroyImage(&orig);
    iftDestroyImage(&aux);

    return ca_model;
}

void DestroyCAModel(iftCAModel **ca_model) {
    if ((*ca_model) != NULL) {
        iftDestroyMImage(&(*ca_model)->orig);
        iftDestroyImage(&(*ca_model)->saliency);
        iftDestroyImage(&(*ca_model)->label);
        iftDestroyFImage(&(*ca_model)->fg_strength);
        iftDestroyFImage(&(*ca_model)->bg_strength);
        iftDestroyAdjRel(&(*ca_model)->neighborhood);
        if (((*ca_model)->brain_mask) != NULL) {
            iftDestroyImage(&(*ca_model)->brain_mask);
        }
        iftFree(*ca_model);
    }
}

iftImage *ExtractParasites(iftImage *img) {
    iftImage *odomes     = iftOpenDomes(img,NULL,NULL);
    iftImage *resid      = iftSub(img,odomes);
    iftAdjRel *A         = iftCircular(1.0);
    iftImage *open       = iftOpen(resid,A,NULL);

    iftImage *parasite;
    int otsu_value = iftOtsu(open);
    if (otsu_value == 0) {
        parasite = iftCreateImageFromImage(img);
    } else {
        parasite = iftThreshold(open, otsu_value, IFT_INFINITY_INT, 255);
    }

    iftDestroyImage(&odomes);
    iftDestroyImage(&resid);
    iftDestroyImage(&open);
    iftDestroyAdjRel(&A);

    return parasite;
}

iftImage *ExtractTumor(iftImage *img, iftImage *brain_mask) {
  // Compute histogram of the input image within the brain mask
  int nbins = iftMaximumValue(img) + 1;
  int *hist = iftAllocIntArray(nbins);

  // Fill histogram for pixels within brain mask
  for (int p = 0; p < img->n; p++) {
    if (brain_mask->val[p] > 0) {
      hist[img->val[p]]++;
    }
  }

  // Find the peak of the highest intensities (ignore background)
  int peak_intensity = 0;
  int max_freq = 0;
  int start_search = (int)(nbins * 0.9); // Start searching from top 10% intensities

  for (int i = start_search; i < nbins; i++) {
    if (hist[i] > max_freq) {
      max_freq = hist[i];
      peak_intensity = i;
    }
  }

  // Calculate mean and standard deviation around the peak
  double sum = 0.0, sum_sq = 0.0;
  int count = 0;
  int window = (int)(nbins * 0.2); // 20% window around peak

  for (int i = peak_intensity - window; i <= peak_intensity + window; i++) {
    if (i >= 0 && i < nbins) {
      sum += i * hist[i];
      sum_sq += i * i * hist[i];
      count += hist[i];
    }
  }

  double mean = (count > 0) ? sum / count : 0;
  double variance = (count > 0) ? (sum_sq / count) - (mean * mean) : 0;
  double std_dev = sqrt(variance);

  // Threshold based on mean - k*std_dev (k is a parameter to adjust)
  double k = 2.5; // Adjust this parameter based on validation
  int threshold = (int)(mean - k * std_dev);

  // Ensure threshold is within valid range
  threshold = iftMax(threshold, 1);

  // Apply threshold
  iftImage *tumor = iftThreshold(img, threshold, IFT_INFINITY_INT, 255);
  iftFree(hist);

  return tumor;
}

// Global file pointer for CSV logging
FILE *timing_csv = NULL;

// Initialize CSV file and write header
int init_timing_csv(const char* filename) {
    timing_csv = fopen(filename, "w");
    if (timing_csv == NULL) {
        printf("Error: Could not create CSV file %s\n", filename);
        return -1;
    }

    // Write CSV header
    fprintf(timing_csv, "iteration,initialization,evolution,prob_map,extra_sal,total\n");
    fflush(timing_csv); // Ensure header is written immediately

    printf("CSV timing log initialized: %s\n", filename);
    return 0;
}

// Close CSV file
void close_timing_csv() {
    if (timing_csv != NULL) {
        fclose(timing_csv);
        timing_csv = NULL;
    }
}

int main(int argc, char **argv) {
    timer *tstart;
    int memory_start, memory_end;
    memory_start = iftMemoryUsed();
    tstart = iftTic();

    if (argc != 6) {
        iftError(
            "Usage: iftCAGpu <P1> <P2> <P3> <P4> <P5>\n"
            "P1: Saliency Folder.\n"
            "P2: Orig image folder.\n"
            "P3: Features Folder.\n"
            "P4: Output folder.\n"
            "P5: Parasites/Brain (0, 1)\n",
            "main"
        );
    }

#ifdef IFT_GPU
    // Initialize CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        iftError("No CUDA-capable devices found!", "main");
    }

    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));
    printf("Using GPU: %s\n", device_prop.name);
#else
    printf("WARNING: Compiled without GPU support. Use standard iftCA instead.\n");
    iftError("GPU support not enabled during compilation!", "main");
#endif

    iftFileSet *fs = iftLoadFileSetFromDirBySuffix(argv[1], ".png", 1);
    init_timing_csv("timing_results_gpu.csv");
    char orig_path[512];
    char output_path[512];
    char ext[8];
    int is_brain = atoi(argv[5]); // 1 for brain
    sprintf(ext, "%s", iftFileExt(fs->files[0]->path));

    struct timespec start, end;
    double initialization=0,evolution=0,get_prob_map=0,extract_sal=0;

    for (size_t i=0; i < fs->n; i++) {
        char *img_basename = iftFilename(fs->files[i]->path, ".png");
        printf("Processing Image: %s\n", img_basename);
        sprintf(orig_path, "%s/%s.png", argv[2], img_basename);

#ifdef IFT_GPU
        GPUMemory *gpu_mem = NULL;
#endif

        // 1. Initializes CA Data
        clock_gettime(CLOCK_MONOTONIC, &start);
        iftCAModel *ca_model = InitializeCA(
            orig_path, fs->files[i]->path, 3, 1, is_brain
        );

#ifdef IFT_GPU
        // Allocate GPU memory and transfer data
        gpu_mem = allocate_gpu_memory(ca_model);
        transfer_to_gpu(ca_model, gpu_mem);
#endif

        clock_gettime(CLOCK_MONOTONIC, &end);
        initialization = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

        iftImage *fg_init = iftFImageToImage(ca_model->fg_strength, 255);
        sprintf(output_path, "%s/%s_%s%s", argv[4], img_basename, "fg_init", ext);
        iftWriteImageByExt(fg_init, output_path);
        iftImage *bg_init = iftFImageToImage(ca_model->bg_strength, 255);
        sprintf(output_path, "%s/%s_%s%s", argv[4], img_basename, "bg_init", ext);
        iftWriteImageByExt(bg_init, output_path);
        iftDestroyImage(&fg_init);
        iftDestroyImage(&bg_init);

        // 2. Evolves CA Data using GPU
        clock_gettime(CLOCK_MONOTONIC, &start);
#ifdef IFT_GPU
        evolve_ca_model_gpu(ca_model, gpu_mem, 1e-12, is_brain);
#else
        // Fallback to CPU version - this should not happen
        iftError("GPU version called without GPU support!", "main");
#endif
        clock_gettime(CLOCK_MONOTONIC, &end);
        evolution = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

        // 3. Gets Probability Map from foreground and background strength
        iftFImage *prob_map = NULL;
        clock_gettime(CLOCK_MONOTONIC, &start);
#ifdef IFT_GPU
        prob_map = GetProbMapGPU(ca_model, gpu_mem);
#else
        iftError("GPU version called without GPU support!", "main");
#endif
        clock_gettime(CLOCK_MONOTONIC, &end);
        get_prob_map = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

        printf("Saliency size: %d %d \n", prob_map->xsize, prob_map->ysize);
        iftImage *output_prob_map = iftFImageToImage(prob_map, 255);
        iftImage *output_fg = iftFImageToImage(ca_model->fg_strength, ca_model->i_max);
        iftImage *output_bg = iftFImageToImage(ca_model->bg_strength, ca_model->i_max);

        // 4. Saves foreground, background, and prob_tumor.
        sprintf(output_path, "%s/%s_%s%s", argv[4], img_basename, "prob_map", ext);
        iftWriteImageByExt(output_prob_map, output_path);
        sprintf(output_path, "%s/%s_%s%s", argv[4], img_basename, "fg", ext);
        iftWriteImageByExt(output_fg, output_path);
        sprintf(output_path, "%s/%s_%s%s", argv[4], img_basename, "bg", ext);
        iftWriteImageByExt(output_bg, output_path);
        sprintf(output_path, "%s/%s_%s%s", argv[4], img_basename, "label", ext);
        for (size_t p=0; p < ca_model->label->n; p++) {
            ca_model->label->val[p] = ca_model->label->val[p] * 255;
        }
        iftWriteImageByExt(ca_model->label, output_path);

        // 5. Computes final saliency
        clock_gettime(CLOCK_MONOTONIC, &start);
        iftImage *output_saliency = NULL;
        if (!is_brain) {
            output_saliency = ExtractParasites(output_prob_map);
        } else {
            output_saliency = ExtractTumor(output_prob_map, ca_model->brain_mask);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        extract_sal = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        sprintf(output_path, "%s/%s_%s%s", argv[4], img_basename, "sal", ext);
        iftWriteImageByExt(output_saliency, output_path);

        // Write timing results
        if (timing_csv != NULL) {
            fprintf(
                timing_csv, "%zu,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                i+1, initialization, evolution, get_prob_map, extract_sal,
                initialization + evolution + get_prob_map + extract_sal
            );
            fflush(timing_csv);
        }

        // Releases allocated resource
#ifdef IFT_GPU
        free_gpu_memory(gpu_mem);
#endif
        iftDestroyFImage(&prob_map);
        iftDestroyImage(&output_saliency);
        iftDestroyImage(&output_prob_map);
        iftDestroyImage(&output_fg);
        iftDestroyImage(&output_bg);
        iftFree(img_basename);
        DestroyCAModel(&ca_model);

        if(i >= 1000) {
            break;
        }
    }

    // Releases allocated resource
    iftDestroyFileSet(&fs);
    close_timing_csv();

    puts("\nDone ...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
    memory_end = iftMemoryUsed();
    iftVerifyMemory(memory_start, memory_end);

    return 0;
}