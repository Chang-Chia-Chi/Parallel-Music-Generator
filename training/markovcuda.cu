#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <cuda.h>
#include "markov.h"

// global variable
cudaEvent_t start, stop;

// host matrices
int* major_high;
int* major_low;
int* minor_high;
int* minor_low;
int* major_chord;
int* minor_chord;

// host buffer
note_info* majorHighBuff;
note_info* majorLowBuff;
note_info* minorHighBuff;
note_info* minorLowBuff;

// memory transfer to device
note_info* device_majorHighBuff;
note_info* device_majorLowBuff;
note_info* device_minorHighBuff;
note_info* device_minorLowBuff;

int* device_majorHighNotes;
int* device_majorLowNotes;
int* device_minorHighNotes;
int* device_minorLowNotes;
int* device_majorChords;
int* device_minorChords;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

void matrix_alloc() {
    auto t_start = high_resolution_clock::now();

    major_high = (int*)malloc(sizeof(int) * (NUM_NOTE * NUM_NOTE));
    major_low = (int*)malloc(sizeof(int) * (NUM_NOTE * NUM_NOTE));
    minor_high = (int*)malloc(sizeof(int) * (NUM_NOTE * NUM_NOTE));
    minor_low = (int*)malloc(sizeof(int) * (NUM_NOTE * NUM_NOTE));

    major_chord = (int*)malloc(sizeof(int) * (NUM_CHORD * NUM_CHORD));
    minor_chord = (int*)malloc(sizeof(int) * (NUM_CHORD * NUM_CHORD));

    majorHighBuff = (note_info*)malloc(sizeof(note_info) * BUFFER_LEN);
    majorLowBuff = (note_info*)malloc(sizeof(note_info) * BUFFER_LEN);
    minorHighBuff = (note_info*)malloc(sizeof(note_info) * BUFFER_LEN);
    minorLowBuff = (note_info*)malloc(sizeof(note_info) * BUFFER_LEN);

    auto t_end = high_resolution_clock::now();
    auto t_spent = duration_cast<milliseconds>(t_end - t_start);
    std::cout << "Time spent for host memory allocation: " << t_spent.count() << " ms\n";
}

void free_matrix() {
    auto t_start = high_resolution_clock::now();

    free(major_high);
    free(major_low);
    free(minor_high);
    free(minor_low);
    free(major_chord);
    free(minor_chord);

    free(majorHighBuff);
    free(majorLowBuff);
    free(minorHighBuff);
    free(minorLowBuff);

    auto t_end = high_resolution_clock::now();
    auto t_spent = duration_cast<milliseconds>(t_end - t_start);
    std::cout << "Time spent for host memory free: " << t_spent.count() << " ms\n";
}


void cuda_malloc() {
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc(&device_majorHighBuff, sizeof(note_info) * BUFFER_LEN);
    cudaMalloc(&device_majorLowBuff, sizeof(note_info) * BUFFER_LEN);
    cudaMalloc(&device_minorHighBuff, sizeof(note_info) * BUFFER_LEN);
    cudaMalloc(&device_minorLowBuff, sizeof(note_info) * BUFFER_LEN);

    cudaMalloc(&device_majorHighNotes, sizeof(int) * NUM_NOTE * NUM_NOTE);
    cudaMemset(device_majorHighNotes, 0, sizeof(int) * NUM_NOTE * NUM_NOTE);
    cudaMalloc(&device_majorLowNotes, sizeof(int) * NUM_NOTE * NUM_NOTE);
    cudaMemset(device_majorLowNotes, 0, sizeof(int) * NUM_NOTE * NUM_NOTE);
    cudaMalloc(&device_minorHighNotes, sizeof(int) * NUM_NOTE * NUM_NOTE);
    cudaMemset(device_minorHighNotes, 0, sizeof(int) * NUM_NOTE * NUM_NOTE);
    cudaMalloc(&device_minorLowNotes, sizeof(int) * NUM_NOTE * NUM_NOTE);
    cudaMemset(device_minorLowNotes, 0, sizeof(int) * NUM_NOTE * NUM_NOTE);
    cudaMalloc(&device_majorChords, sizeof(int) * NUM_CHORD * NUM_CHORD);
    cudaMemset(device_majorChords, 0, sizeof(int) * NUM_CHORD * NUM_CHORD);
    cudaMalloc(&device_minorChords, sizeof(int) * NUM_CHORD * NUM_CHORD);
    cudaMemset(device_minorChords, 0, sizeof(int) * NUM_CHORD * NUM_CHORD);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time spent for device memory allocation: " << elapsedTime << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void cuda_free() {
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaFree(device_majorHighBuff);
    cudaFree(device_majorLowBuff);
    cudaFree(device_minorHighBuff);
    cudaFree(device_minorLowBuff);

    cudaFree(device_majorHighNotes);
    cudaFree(device_majorLowNotes);
    cudaFree(device_minorHighNotes);
    cudaFree(device_minorLowNotes);
    cudaFree(device_majorChords);
    cudaFree(device_minorChords);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time spent for device memory allocation: " << elapsedTime << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void cuda_to_host() {
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemcpy(major_high, device_majorHighNotes, sizeof(int) * NUM_NOTE * NUM_NOTE, cudaMemcpyDeviceToHost);
    cudaMemcpy(major_low, device_majorLowNotes, sizeof(int) * NUM_NOTE * NUM_NOTE, cudaMemcpyDeviceToHost);
    cudaMemcpy(minor_high, device_minorHighNotes, sizeof(int) * NUM_NOTE * NUM_NOTE, cudaMemcpyDeviceToHost);
    cudaMemcpy(minor_low, device_minorLowNotes, sizeof(int) * NUM_NOTE * NUM_NOTE, cudaMemcpyDeviceToHost);
    cudaMemcpy(major_chord, device_majorChords, sizeof(int) * NUM_CHORD * NUM_CHORD, cudaMemcpyDeviceToHost);
    cudaMemcpy(minor_chord, device_minorChords, sizeof(int) * NUM_CHORD * NUM_CHORD, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time spent for copy memory back: " << elapsedTime << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__device__ inline int cuda_getNoteIndex(int curr_tone, int curr_dur, int prev_tone_1, int prev_dur_1, int tune) {
    int col = curr_tone * NUM_DURATION + curr_dur ;

    // If previous tone is chord, get top note and find closest
    if (prev_tone_1 >= CHORD_BASE) {
        prev_tone_1 = (prev_tone_1 - CHORD_BASE) / 144; // Get top note
        if (curr_tone == NUM_TONE - 1) { // if curr_tone is Rest
            prev_tone_1 = prev_tone_1 + 12 * (2 * tune);
        } else {
            prev_tone_1 = curr_tone - (curr_tone % 12) + prev_tone_1;
        }
    }

    int row;
    row = prev_tone_1 * NUM_DURATION + prev_dur_1;

    return row * NUM_NOTE + col;
}

__device__ inline int cuda_getChordIndex(int curr_tone, int prev_tone) {
    if (prev_tone >= CHORD_BASE) {
        prev_tone = prev_tone - CHORD_BASE;
    } 
    else if (prev_tone == NUM_TONE - 1) {
        return -1;
    }
    else {
        prev_tone = (prev_tone % 12) + (prev_tone % 12) * 12 + (prev_tone % 12) * 144;
    }
    return prev_tone * NUM_CHORD + (curr_tone - CHORD_BASE);
}

__global__ void note_kernel(note_info* device_Buff, int* device_Mat, int use_len) {
    int start, end;
    int tx = threadIdx.x;
    start = tx * (use_len / NUM_THREADS) + 1;
    if (tx == NUM_THREADS - 1) {
        end = use_len;
    } else {
        end = (tx + 1) * (use_len / NUM_THREADS);
    }

    int index;
    int curr_tone, curr_dur, prev_tone, prev_dur, tune;
    for (int i = start; i < end; i++) {
        curr_tone = device_Buff[i].tone;
        curr_tone = device_Buff[i].dur;
        prev_tone = device_Buff[i - 1].tone;
        prev_dur = device_Buff[i - 1].dur;
        tune = device_Buff[i].tune;
        if (curr_tone < CHORD_BASE && prev_tone != -1) {
            index = cuda_getNoteIndex(curr_tone, curr_dur, prev_tone, prev_dur, tune);
            if (index != -1) {
                atomicAdd(&device_Mat[index], 1);
            }
        }
    }
}

__global__ void chord_kernel(note_info* device_Buff, int* device_Mat, int use_len) {
    int start, end;
    int tx = threadIdx.x;
    start = tx * (use_len / NUM_THREADS) + 1;
    if (tx == NUM_THREADS - 1) {
        end = use_len;
    } else {
        end = (tx + 1) * (use_len / NUM_THREADS);
    }

    int index;
    int curr_tone, prev_tone;
    for (int i = start; i < end; i++) {
        curr_tone = device_Buff[i].tone;
        prev_tone = device_Buff[i - 1].tone;
        if (curr_tone >= CHORD_BASE && prev_tone != -1) {
            index = cuda_getChordIndex(curr_tone, prev_tone);
            if (index != -1) {
                atomicAdd(&device_Mat[index], 1);
            }
        }
    }    
}

void buffer_copy(note_info* major_high_buff, note_info* major_low_buff, int major_high_len, int major_low_len,
                 note_info* minor_high_buff, note_info* minor_low_buff, int minor_high_len, int minor_low_len) {
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cudaMemcpy(device_majorHighBuff, major_high_buff, sizeof(note_info) * major_high_len, cudaMemcpyHostToDevice);
        cudaMemcpy(device_majorLowBuff, major_low_buff, sizeof(note_info) * major_low_len, cudaMemcpyHostToDevice);

        cudaMemcpy(device_minorHighBuff, minor_high_buff, sizeof(note_info) * minor_high_len, cudaMemcpyHostToDevice);
        cudaMemcpy(device_minorLowBuff, minor_low_buff, sizeof(note_info) * minor_low_len, cudaMemcpyHostToDevice);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Time spent for buffer copy: " << elapsedTime << " ms\n"; 
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
}

void cuda_note_count(int minor_high_len, int minor_low_len, int major_high_len, int major_low_len) {

    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    note_kernel<<<1, NUM_THREADS>>>(device_majorHighBuff, device_majorHighNotes, major_high_len);
    note_kernel<<<1, NUM_THREADS>>>(device_majorLowBuff, device_majorLowNotes, major_low_len);
    chord_kernel<<<1, NUM_THREADS>>>(device_majorHighBuff, device_majorChords, major_high_len);
    chord_kernel<<<1, NUM_THREADS>>>(device_majorLowBuff, device_majorChords, major_low_len);

    note_kernel<<<1, NUM_THREADS>>>(device_minorHighBuff,device_minorHighNotes, minor_high_len);
    note_kernel<<<1, NUM_THREADS>>>(device_minorLowBuff, device_minorLowNotes, minor_low_len);
    chord_kernel<<<1, NUM_THREADS>>>(device_minorHighBuff, device_minorChords, minor_high_len);
    chord_kernel<<<1, NUM_THREADS>>>(device_minorLowBuff, device_minorChords, minor_low_len);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time spent for matrix generation: " << elapsedTime << " ms\n";  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}