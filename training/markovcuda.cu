#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <cuda.h>
#include "markov.h"

cudaStream_t majorHighStream;
cudaStream_t majorLowStream;
cudaStream_t minorHighStream;
cudaStream_t minorLowStream;
cudaStream_t majorHighChordStream;
cudaStream_t majorLowChordStream;
cudaStream_t minorHighChordStream;
cudaStream_t minorLowChordStream;

// memory transfer to device
int* device_majorHighTone;
int* device_majorHighDur;
int* device_majorLowTone;
int* device_majorLowDur;
int* device_minorHighTone;
int* device_minorHighDur;
int* device_minorLowTone;
int* device_minorLowDur;

int* device_majorHighNotes;
int* device_majorLowNotes;
int* device_minorHighNotes;
int* device_minorLowNotes;
int* device_majorChords;
int* device_minorChords;

void cuda_stream_create() {
    cudaStreamCreate(&majorHighStream);
    cudaStreamCreate(&majorLowStream);
    cudaStreamCreate(&minorHighStream);
    cudaStreamCreate(&minorLowStream);
    cudaStreamCreate(&majorHighChordStream);
    cudaStreamCreate(&majorLowChordStream);
    cudaStreamCreate(&minorHighChordStream);
    cudaStreamCreate(&minorLowChordStream);
}

void cuda_stream_destroy() {
    cudaStreamDestroy(majorHighStream);
    cudaStreamDestroy(majorLowStream);
    cudaStreamDestroy(minorHighStream);
    cudaStreamDestroy(minorLowStream);
    cudaStreamDestroy(majorHighChordStream);
    cudaStreamDestroy(majorLowChordStream);
    cudaStreamDestroy(minorHighChordStream);
    cudaStreamDestroy(minorLowChordStream);
}

void cuda_malloc() {
    cudaMalloc(&device_majorHighTone, sizeof(int) * BUFFER_LEN);
    cudaMalloc(&device_majorHighDur, sizeof(int) * BUFFER_LEN);
    cudaMalloc(&device_majorLowTone, sizeof(int) * BUFFER_LEN);
    cudaMalloc(&device_majorLowDur, sizeof(int) * BUFFER_LEN);
    cudaMalloc(&device_minorHighTone, sizeof(int) * BUFFER_LEN);
    cudaMalloc(&device_minorHighDur, sizeof(int) * BUFFER_LEN);
    cudaMalloc(&device_minorLowTone, sizeof(int) * BUFFER_LEN);
    cudaMalloc(&device_minorLowDur, sizeof(int) * BUFFER_LEN);

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
}

void cuda_free() {
    cudaFree(device_majorHighTone);
    cudaFree(device_majorHighDur);
    cudaFree(device_majorLowTone);
    cudaFree(device_majorLowDur);
    cudaFree(device_minorHighTone);
    cudaFree(device_minorHighDur);
    cudaFree(device_minorLowTone);
    cudaFree(device_minorLowDur);

    cudaFree(device_majorHighNotes);
    cudaFree(device_majorLowNotes);
    cudaFree(device_minorHighNotes);
    cudaFree(device_minorLowNotes);
    cudaFree(device_majorChords);
    cudaFree(device_minorChords);
}

void cuda_stream_synch(int is_major) {
    if (is_major == 0)
    {
        cudaStreamSynchronize(minorHighStream);
        cudaStreamSynchronize(minorLowStream);
        cudaStreamSynchronize(minorHighChordStream);
        cudaStreamSynchronize(minorLowChordStream);
    }
    else if (is_major == 1)
    {
        cudaStreamSynchronize(majorHighStream);
        cudaStreamSynchronize(majorLowStream);
        cudaStreamSynchronize(majorHighChordStream);
        cudaStreamSynchronize(majorLowChordStream);
    }
}   

void cuda_to_host() {
    cudaMemcpy(major_high, device_majorHighNotes, sizeof(int) * NUM_NOTE * NUM_NOTE, cudaMemcpyDeviceToHost);
    cudaMemcpy(major_low, device_majorLowNotes, sizeof(int) * NUM_NOTE * NUM_NOTE, cudaMemcpyDeviceToHost);
    cudaMemcpy(minor_high, device_minorHighNotes, sizeof(int) * NUM_NOTE * NUM_NOTE, cudaMemcpyDeviceToHost);
    cudaMemcpy(minor_low, device_minorLowNotes, sizeof(int) * NUM_NOTE * NUM_NOTE, cudaMemcpyDeviceToHost);
    cudaMemcpy(major_chord, device_majorChords, sizeof(int) * NUM_CHORD * NUM_CHORD, cudaMemcpyDeviceToHost);
    cudaMemcpy(minor_chord, device_minorChords, sizeof(int) * NUM_CHORD * NUM_CHORD, cudaMemcpyDeviceToHost);
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

__global__ void note_kernel(int* device_Tone, int* device_Dur, int* device_mat, int use_len, int tune) {
    int start, end;
    int tx = threadIdx.x;
    start = tx * (use_len / NUM_THREADS) + 1;
    if (tx == NUM_THREADS - 1) {
        end = use_len;
    } else {
        end = (tx + 1) * (use_len / NUM_THREADS);
    }

    int index;
    int curr_Tone, curr_Dur, prev_Tone, prev_Dur;
    for (int i = start; i < end; i++) {
        curr_Tone = device_Tone[i];
        curr_Dur = device_Dur[i];
        prev_Tone = device_Tone[i - 1];
        prev_Dur = device_Dur[i - 1];
        if (curr_Tone < CHORD_BASE && prev_Tone != -1) {
            index = cuda_getNoteIndex(curr_Tone, curr_Dur, prev_Tone, prev_Dur, tune);
            if (index != -1) {
                atomicAdd(&device_mat[index], 1);
            }
        }
    }
}

__global__ void chord_kernel(int* device_Tone, int* device_mat, int use_len) {
    int start, end;
    int tx = threadIdx.x;
    start = tx * (use_len / NUM_THREADS) + 1;
    if (tx == NUM_THREADS - 1) {
        end = use_len;
    } else {
        end = (tx + 1) * (use_len / NUM_THREADS);
    }

    int index;
    int curr_Tone, prev_Tone;
    for (int i = start; i < end; i++) {
        curr_Tone = device_Tone[i];
        prev_Tone = device_Tone[i - 1];
        if (curr_Tone >= CHORD_BASE && prev_Tone != -1) {
            index = cuda_getChordIndex(curr_Tone, prev_Tone);
            if (index != -1) {
                atomicAdd(&device_mat[index], 1);
            }
        }
    }    
}

void cuda_note_count(int* high_tone, int* hign_dur, int* low_tone, int* low_dur, int high_len, int low_len, int is_major, int tune) {
    if (is_major == 0)
    {     
        cudaMemcpy(device_minorHighTone, high_tone, sizeof(int) * high_len, cudaMemcpyHostToDevice);
        cudaMemcpy(device_minorHighDur, hign_dur, sizeof(int) * high_len, cudaMemcpyHostToDevice);
        note_kernel<<<1, NUM_THREADS, 0>>>(device_minorHighTone, device_minorHighDur, device_minorHighNotes, high_len, tune);

        cudaMemcpy(device_minorLowTone, low_tone, sizeof(int) * low_len, cudaMemcpyHostToDevice);
        cudaMemcpy(device_minorLowDur, low_dur, sizeof(int) * low_len, cudaMemcpyHostToDevice);
        note_kernel<<<1, NUM_THREADS>>>(device_minorLowTone, device_minorLowDur, device_minorLowNotes, low_len, tune);

        cudaMemcpy(device_minorLowTone, low_tone, sizeof(int) * low_len, cudaMemcpyHostToDevice);
        cudaMemcpy(device_minorHighTone, high_tone, sizeof(int) * high_len, cudaMemcpyHostToDevice);
        chord_kernel<<<1, NUM_THREADS>>>(device_minorLowTone, device_minorChords, low_len);
        chord_kernel<<<1, NUM_THREADS>>>(device_minorHighTone, device_minorChords, high_len);
    }
    else if (is_major == 1)
    {
        cudaMemcpy(device_majorHighTone, high_tone, sizeof(int) * high_len, cudaMemcpyHostToDevice);
        cudaMemcpy(device_majorHighDur, hign_dur, sizeof(int) * high_len, cudaMemcpyHostToDevice);
        note_kernel<<<1, NUM_THREADS>>>(device_majorHighTone, device_majorHighDur, device_majorHighNotes, high_len, tune);

        cudaMemcpy(device_majorLowTone, low_tone, sizeof(int) * low_len, cudaMemcpyHostToDevice);
        cudaMemcpy(device_majorLowDur, low_dur, sizeof(int) * low_len, cudaMemcpyHostToDevice);
        note_kernel<<<1, NUM_THREADS>>>(device_minorLowTone, device_minorLowDur, device_majorLowNotes, low_len, tune);

        cudaMemcpy(device_majorLowTone, low_tone, sizeof(int) * low_len, cudaMemcpyHostToDevice);
        cudaMemcpy(device_minorHighTone, high_tone, sizeof(int) * high_len, cudaMemcpyHostToDevice);
        chord_kernel<<<1, NUM_THREADS>>>(device_majorLowTone, device_majorChords, low_len);
        chord_kernel<<<1, NUM_THREADS>>>(device_majorHighTone, device_majorChords, high_len);
    }
}