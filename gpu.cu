#include "common.h"
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int binPerRow;
int binCount;
double binSize;
static texture<int2, 1, cudaReadModeElementType> old_pos_tex;
static texture<int2, 1, cudaReadModeElementType> old_vel_tex;
static texture<int2, 1, cudaReadModeElementType> old_acc_tex;
static texture<int,  1, cudaReadModeElementType> bin_index_tex;
static texture<int,  1, cudaReadModeElementType> particle_index_tex;
static texture<int,  1, cudaReadModeElementType> bin_start_tex;
static texture<int,  1, cudaReadModeElementType> bin_end_tex;
double *d_pos;
double *d_vel;
double *d_acc;
double *sorted_pos;
double *sorted_vel;
double *sorted_acc;
int *bin_index;
int *particle_index;
int *original_p_index;
int *bin_start;
int *bin_end;
int num_bins;

static __inline__ __device__ double fetch_double(texture<int2, 1> t, int i)
{
	int2 v = tex1Dfetch(t, i);
	return __hiloint2double(v.y, v.x);
}

__global__ void copyparts_2( particle_t* parts, int num_parts, double *pos, double *vel, double* acc)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_parts) return;

	particle_t* p = &parts[tid];

	pos[2*tid] = p -> x;
	pos[2*tid+1] = p -> y;
	vel[2*tid] = p -> vx;
	vel[2*tid+1] = p -> vy;
	acc[2*tid] = p -> ax;
	acc[2*tid+1] = p -> ay;
    
}
__global__ void copyparts_back( particle_t* parts, int num_parts, double *pos, double *vel, double* acc)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_parts) return;

	particle_t* p = &parts[tid];

	p -> x = pos[2*tid];
	p -> y = pos[2*tid+1];
	p -> vx = vel[2*tid];
	p -> vy = vel[2*tid+1];
	p -> ax = acc[2*tid];
	p -> ay = acc[2*tid+1];
}
void sort_particles(int *bin_index, int *particle_index, int num_parts)
{
	thrust::sort_by_key(thrust::device_ptr<int>(bin_index),
			thrust::device_ptr<int>(bin_index + num_parts),
			thrust::device_ptr<int>(particle_index));
}

static __inline__ __device__ int binNum(double &d_x, double &d_y, int binPerRow, double binSize) 
{
	int rowid = int(d_y / binSize);
	int colid = int(d_x / binSize);

	if (rowid >= binPerRow)
		rowid = binPerRow - 1;
	if (colid >= binPerRow)
		colid = binPerRow - 1;
	return rowid * binPerRow + colid;
}

__global__ void reorder_data_calc_bin(int *bin_start, int *bin_end, double *sorted_pos, 
	double *sorted_vel, double *sorted_acc, int *bin_index, int *particle_index, int *original_p_index, double *d_pos, 
	double *d_vel, double *d_acc, int num_parts, int num_bins)
{
	extern __shared__ int sharedHash[];    // blockSize + 1 elements
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int bi;
	if (index < num_parts) {
		bi = bin_index[index];
		sharedHash[threadIdx.x+1] = bi;
		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = bin_index[index-1];
		}
	}

	__syncthreads();

	if (index < num_parts) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || bi != sharedHash[threadIdx.x])
		{
			bin_start[bi] = index;
			if (index > 0)
				bin_end[sharedHash[threadIdx.x]] = index;
		}

		if (index == num_parts - 1)
		{
			bin_end[bi] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		int sortedIndex = particle_index[index];
		sorted_pos[2*index]   = d_pos[2*sortedIndex];
		sorted_pos[2*index+1] = d_pos[2*sortedIndex+1];
		sorted_vel[2*index]   = d_vel[2*sortedIndex];
		sorted_vel[2*index+1] = d_vel[2*sortedIndex+1];
		sorted_acc[2*index]   = d_acc[2*sortedIndex];
		sorted_acc[2*index+1] = d_acc[2*sortedIndex+1];
		original_p_index[sortedIndex] = index;

	}
}
__global__ void reorder_data_back(double *sorted_back_pos, double *sorted_back_vel, double *sorted_back_acc, 
	int *original_p_index, double *sorted_pos, double *sorted_vel, double *sorted_acc, int num_parts)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < num_parts) {
		// Now use the sorted index to reorder the pos and vel data
		int sortedIndex = original_p_index[index];
		sorted_back_pos[2*index]     = sorted_pos[2*sortedIndex];
		sorted_back_pos[2*index+1] = sorted_pos[2*sortedIndex+1];
		sorted_back_vel[2*index]       = sorted_vel[2*sortedIndex];
		sorted_back_vel[2*index+1]   = sorted_vel[2*sortedIndex+1];
		sorted_back_acc[2*index]   = sorted_acc[2*sortedIndex];
		sorted_back_acc[2*index+1] = sorted_acc[2*sortedIndex+1];
	}
}

__global__ void calculate_bin_index(int *bin_index, int *particle_index, double *d_pos, int num_parts, int binPerRow, double binSize)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index >= num_parts) return;
	double pos_x = fetch_double(old_pos_tex, 2*index);
	double pos_y = fetch_double(old_pos_tex, 2*index+1);
	int cbin = binNum( pos_x,pos_y,binPerRow, binSize);
	bin_index[index] = cbin;
	particle_index[index] = index;
}

__device__ void apply_force_gpu(double &particle_x, double &particle_y, double &particle_ax, double &particle_ay, double &neighbor_x, double &neighbor_y)
{
	double dx = neighbor_x - particle_x;
	double dy = neighbor_y - particle_y;
	double r2 = dx * dx + dy * dy;
	if( r2 > cutoff * cutoff )
		return;

	r2 = (r2 > min_r * min_r) ? r2 : min_r*min_r;
	double r = sqrt( r2 );

	//
	//  very simple short-range repulsive force
	//
	double coef = ( 1 - cutoff / r ) / r2 / mass;
	particle_ax += coef * dx;
	particle_ay += coef * dy;
}

__global__ void compute_forces_gpu(double *pos, double *acc, int num_parts, int binPerRow, int *bin_start, int *bin_end, double binSize)
{
	// Get thread (particle) ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_parts) 
		return;

	double pos_1x = fetch_double(old_pos_tex, 2*tid);
	double pos_1y = fetch_double(old_pos_tex, 2*tid+1);

	// find current particle's in, handle boundaries
	int cbin = binNum( pos_1x, pos_1y, binPerRow, binSize);
	// printf("Thread %d, cbin = %d, binPerRow = %d\n", tid, cbin, binPerRow);

	int lowi = -1, highi = 1, lowj = -1, highj = 1;
	if (cbin < binPerRow) // in the first row
		lowj = 0;
	if (cbin % binPerRow == 0) // in the first column
		lowi = 0;
	if (cbin % binPerRow == (binPerRow-1))
		highi = 0;
	if (cbin >= binPerRow*(binPerRow-1))
		highj = 0;

	double acc_x;
	double acc_y;
	acc_x = acc_y = 0;
	for (int i = lowi; i <= highi; i++)
		for (int j = lowj; j <= highj; j++)
		{
			int nbin = cbin + i + binPerRow*j;
			int bin_st = tex1Dfetch(bin_start_tex, nbin);
			if (bin_st != 0xffffffff) {
				int bin_et = tex1Dfetch(bin_end_tex, nbin);
				for (int k = bin_st; k < bin_et; k++ ) {
					double pos_2x = fetch_double(old_pos_tex, 2*k);
					double pos_2y = fetch_double(old_pos_tex, 2*k+1);
					// printf("Thread %d, neighbor_bin = %d\n", tid, k);

					apply_force_gpu( pos_1x, pos_1y, acc_x, acc_y, pos_2x, pos_2y );
				}
			}
		}
	acc[2*tid] = acc_x;
	acc[2*tid+1] = acc_y;
	// printf("Thread %d, acc_x = %f, acc_y = %f\n", tid, acc_x, acc_y);

}

__global__ void move_gpu (double *pos, double *vel, double *acc, int num_parts, double size)
{

	// Get thread (particle) ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_parts) 
		return;

	double acc_x = fetch_double(old_acc_tex, 2*tid);
	double acc_y = fetch_double(old_acc_tex, 2*tid+1);
	double vel_x = fetch_double(old_vel_tex, 2*tid);
	double vel_y = fetch_double(old_vel_tex, 2*tid+1);
	double pos_x = fetch_double(old_pos_tex, 2*tid);
	double pos_y = fetch_double(old_pos_tex, 2*tid+1);
	vel_x += acc_x * dt;
	vel_y += acc_y * dt;
	pos_x += vel_x * dt;
	pos_y += vel_y * dt;

	//
	//  bounce from walls
	//
	while( pos_x < 0 || pos_x > size )
	{
		pos_x = pos_x < 0 ? - (pos_x) : 2 * size - pos_x;
		vel_x = - (vel_x);
	}
	while( pos_y < 0 || pos_y > size )
	{
		pos_y = pos_y < 0 ? - (pos_y) : 2 * size - pos_y;
		vel_y = - (vel_y);
	}

	vel[2*tid] = vel_x;
	vel[2*tid+1] = vel_y;
	pos[2*tid] = pos_x;
    pos[2*tid+1] = pos_y;
	acc[2*tid] = 0;
	acc[2*tid+1] = 0;
}


void init_simulation(particle_t* parts, int num_parts, double size) 
{
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

	// GPU particle data structure
	cudaMalloc((void **) &d_pos, 2*num_parts * sizeof(double));
	cudaMalloc((void **) &d_vel, 2*num_parts * sizeof(double));
	cudaMalloc((void **) &d_acc, 2*num_parts * sizeof(double));

	cudaMalloc((void **) &sorted_pos, 2*num_parts * sizeof(double));
	cudaMalloc((void **) &sorted_vel, 2*num_parts * sizeof(double));
	cudaMalloc((void **) &sorted_acc, 2*num_parts * sizeof(double));
	
	cudaMalloc((void **) &bin_index, num_parts * sizeof(int));
	cudaMemset(bin_index, 0x0, num_parts * sizeof(int));
	cudaMalloc((void **) &particle_index, num_parts * sizeof(int));
	cudaMemset(particle_index, 0x0, num_parts * sizeof(int));

	cudaMalloc((void **) &original_p_index, num_parts * sizeof(int));
	cudaMemset(original_p_index, 0x0, num_parts * sizeof(int));

	binPerRow = int(size / (1.3 * cutoff));
	num_bins = binPerRow * binPerRow;

	cudaMalloc((void **) &bin_start, num_bins * sizeof(int));
	cudaMalloc((void **) &bin_end, num_bins * sizeof(int));
	cudaMemset(bin_start, 0x0, num_bins * sizeof(int));
	cudaMemset(bin_end, 0x0, num_bins * sizeof(int));
	cudaDeviceSynchronize();
	blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) 
{
	copyparts_2 <<< blks, NUM_THREADS >>> (parts, num_parts, d_pos, d_vel, d_acc);

    cudaBindTexture(0, old_pos_tex, d_pos, 2*num_parts * sizeof(int2));
    calculate_bin_index <<< blks, NUM_THREADS >>> (bin_index, particle_index, d_pos, num_parts, binPerRow, binSize);
    cudaUnbindTexture(old_pos_tex);

    cudaBindTexture(0, bin_index_tex, bin_index, num_parts * sizeof(int));
    cudaBindTexture(0, particle_index_tex, particle_index, num_parts * sizeof(int));
	sort_particles(bin_index, particle_index, num_parts);
    cudaUnbindTexture(bin_index_tex);
    cudaUnbindTexture(particle_index_tex);

    cudaMemset(bin_start, 0xffffffff, num_bins * sizeof(int));
    int smemSize = sizeof(int)*(NUM_THREADS+1);
    reorder_data_calc_bin <<< blks, NUM_THREADS, smemSize >>> (bin_start, bin_end, sorted_pos, sorted_vel, sorted_acc, bin_index, particle_index, original_p_index, d_pos, d_vel, d_acc, num_parts, num_bins);

    cudaBindTexture(0, old_pos_tex, sorted_pos, 2*num_parts * sizeof(int2));
    cudaBindTexture(0, bin_start_tex, bin_start, num_bins * sizeof(int));
    cudaBindTexture(0, bin_end_tex, bin_end, num_bins * sizeof(int));

    compute_forces_gpu <<< blks, NUM_THREADS >>> (sorted_pos, sorted_acc, num_parts, binPerRow, bin_start, bin_end, binSize);

    cudaUnbindTexture(old_pos_tex);
    cudaUnbindTexture(bin_start_tex);
    cudaUnbindTexture(bin_end_tex);


    cudaDeviceSynchronize(); //or add hasMoved to parts

    // Move particles
    cudaBindTexture(0, old_pos_tex, sorted_pos, 2*num_parts * sizeof(int2));
    cudaBindTexture(0, old_vel_tex, sorted_vel, 2*num_parts * sizeof(int2));
    cudaBindTexture(0, old_acc_tex, sorted_acc, 2*num_parts * sizeof(int2));
	move_gpu <<< blks, NUM_THREADS >>> (sorted_pos, sorted_vel, sorted_acc, num_parts, size);
    cudaUnbindTexture(old_pos_tex);
    cudaUnbindTexture(old_vel_tex);
    cudaUnbindTexture(old_acc_tex);

    // Swap particles between d_particles and sorted_particles
	reorder_data_back<<< blks, NUM_THREADS >>> (d_pos, d_vel, d_acc,  original_p_index, sorted_pos, sorted_vel, sorted_acc, num_parts);

	// cudaDeviceSynchronize();
	copyparts_back<<< blks, NUM_THREADS >>> (parts, num_parts, d_pos, d_vel, d_acc);

}
