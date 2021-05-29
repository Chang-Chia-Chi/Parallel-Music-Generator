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
}

void cuda_free() {
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

void buffer_copy(note_info* high_buff, note_info* low_buff, int high_len, int low_len, int is_major) {
    if (is_major == 0) 
    {
        cudaMemcpy(device_minorHighBuff, high_buff, sizeof(note_info) * high_len, cudaMemcpyHostToDevice);
        cudaMemcpy(device_minorLowBuff, low_buff, sizeof(note_info) * low_len, cudaMemcpyHostToDevice);
    }
    else if (is_major == 1)
    {
        cudaMemcpy(device_majorHighBuff, high_buff, sizeof(note_info) * high_len, cudaMemcpyHostToDevice);
        cudaMemcpy(device_majorLowBuff, low_buff, sizeof(note_info) * low_len, cudaMemcpyHostToDevice);
    }
}

void cuda_note_count(int high_len, int low_len, int is_major) {
    if (is_major == 0)
    {     
        note_kernel<<<1, NUM_THREADS>>>(device_minorHighBuff,device_minorHighNotes, high_len);
        note_kernel<<<1, NUM_THREADS>>>(device_minorLowBuff, device_minorLowNotes, low_len);
        chord_kernel<<<1, NUM_THREADS>>>(device_minorLowBuff, device_minorChords, low_len);
        chord_kernel<<<1, NUM_THREADS>>>(device_minorHighBuff, device_minorChords, high_len);
    }
    else if (is_major == 1)
    {
        note_kernel<<<1, NUM_THREADS>>>(device_majorHighBuff, device_majorHighNotes, high_len);
        note_kernel<<<1, NUM_THREADS>>>(device_minorLowBuff, device_majorLowNotes, low_len);
        chord_kernel<<<1, NUM_THREADS>>>(device_majorLowBuff, device_majorChords, low_len);
        chord_kernel<<<1, NUM_THREADS>>>(device_majorHighBuff, device_majorChords, high_len);
    }
}