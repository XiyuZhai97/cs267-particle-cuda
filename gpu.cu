#include "common.h"
#include <cuda.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

int binPerRow;
int binCount;
double binSize;

int* heads;
int* Llist;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}


__device__ int inline calcBin(particle_t& p, int binPerRow, double binSize) {
    int rowid = int(p.y / binSize);
    int colid = int(p.x / binSize);
    if (rowid == binPerRow) {
        rowid = binPerRow - 1;
    }
    if (colid == binPerRow) {
        colid = binPerRow - 1;
    }
    return rowid * binPerRow + colid;
}


__global__ void rebin(particle_t* particles, int num_parts, int binPerRow, int binCount, double binSize, int* heads, int* Llist) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < binCount) {
        heads[tid] = -1;
    }
    if (tid >= num_parts)
        return;
    int thisBin = calcBin(particles[tid], binPerRow, binSize);
    Llist[tid] = atomicExch(&heads[thisBin], tid);
}

__device__ void inline applyForceBetweenBlock(particle_t* parts, int pid, int neighborBin, int* heads, int* Llist) {
    int ptr = heads[neighborBin];
    for (; ptr != -1; ptr = Llist[ptr]) {
        apply_force_gpu(parts[pid], parts[ptr]);
    }
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int binPerRow, int binCount, double binSize, int* heads, int* Llist) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    int thisBin = calcBin(particles[tid], binPerRow, binSize);
    int lowi = -1, highi = 1, lowj = -1, highj = 1;
	if (thisBin < binPerRow) // in the first row
		lowj = 0;
	if (thisBin % binPerRow == 0) // in the first column
		lowi = 0;
	if (thisBin % binPerRow == (binPerRow-1))
		highi = 0;
	if (thisBin >= binPerRow*(binPerRow-1))
        highj = 0;
        
    for (int i = lowi; i <= highi; i++){
        for (int j = lowj; j <= highj; j++){
            int neighborBin = thisBin + i + binPerRow*j;
            applyForceBetweenBlock(particles, tid, neighborBin, heads, Llist);
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    binPerRow = size / cutoff;
    binCount = binPerRow * binPerRow;
    binSize = size / binPerRow;
    cudaMalloc((void**)&heads, binCount * sizeof(int));
    cudaMalloc((void**)&Llist, num_parts * sizeof(int));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function
    int tmp_blks = (binCount + NUM_THREADS - 1) / NUM_THREADS;
    rebin<<<tmp_blks, NUM_THREADS>>>(parts, num_parts, binPerRow, binCount, binSize, heads, Llist);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, binPerRow, binCount, binSize, heads, Llist);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);

}