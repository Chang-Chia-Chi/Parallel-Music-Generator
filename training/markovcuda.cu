#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <cuda.h>
#include "markov.h"

int buffer_size = sizeof(int) * BUFFER_LEN;
cudaStream_t majorHighStream;
cudaStream_t majorLowStream;
cudaStream_t minorHighStream;
cudaStream_t minorLowStream;
cudaStream_t majorChordStream;
cudaStream_t minorChordStream;

// host pinned-memory
int* majorHighTone;
int* majorHighDur;
int* majorLowTone;
int* majorLowDur;
int* minorHighTone;
int* minorHighDur;
int* minorLowTone;
int* minorLowDur;

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

void cuda_pinned_alloc() {
    cudaHostAlloc(&majorHighTone, buffer_size, cudaHostAllocMapped);
    cudaHostAlloc(&majorHighDur, buffer_size, cudaHostAllocMapped);
    cudaHostAlloc(&majorLowTone, buffer_size, cudaHostAllocMapped);
    cudaHostAlloc(&majorLowDur, buffer_size, cudaHostAllocMapped);
    cudaHostAlloc(&minorHighTone, buffer_size, cudaHostAllocMapped);
    cudaHostAlloc(&minorHighDur, buffer_size, cudaHostAllocMapped);
    cudaHostAlloc(&minorLowTone, buffer_size, cudaHostAllocMapped);
    cudaHostAlloc(&minorLowDur, buffer_size, cudaHostAllocMapped);
}

void cuda_stream_create() {
    cudaStreamCreate(&majorHighStream);
    cudaStreamCreate(&majorLowStream);
    cudaStreamCreate(&minorHighStream);
    cudaStreamCreate(&minorLowStream);
    cudaStreamCreate(&majorChordStream);
    cudaStreamCreate(&minorChordStream);
}

void cuda_stream_destroy() {
    cudaStreamDestroy(majorHighStream);
    cudaStreamDestroy(majorLowStream);
    cudaStreamDestroy(minorHighStream);
    cudaStreamDestroy(minorLowStream);
    cudaStreamDestroy(majorChordStream);
    cudaStreamDestroy(minorChordStream);
}

void cuda_malloc() {
    cudaMalloc(&device_majorHighTone, buffer_size);
    cudaMalloc(&device_majorHighDur, buffer_size);
    cudaMalloc(&device_majorLowTone, buffer_size);
    cudaMalloc(&device_majorLowDur, buffer_size);
    cudaMalloc(&device_minorHighTone, buffer_size);
    cudaMalloc(&device_minorHighDur, buffer_size);
    cudaMalloc(&device_minorLowTone, buffer_size);
    cudaMalloc(&device_minorLowDur, buffer_size);

    cudaMalloc(&device_majorHighNotes, sizeof(int) * NUM_TONE * NUM_TONE);
    cudaMemsetAsync(device_majorHighNotes, 0, sizeof(int) * NUM_TONE * NUM_TONE, majorHighStream);
    cudaMalloc(&device_majorLowNotes, sizeof(int) * NUM_TONE * NUM_TONE);
    cudaMemsetAsync(device_majorLowNotes, 0, sizeof(int) * NUM_TONE * NUM_TONE, majorLowStream);
    cudaMalloc(&device_minorHighNotes, sizeof(int) * NUM_TONE * NUM_TONE);
    cudaMemsetAsync(device_minorHighNotes, 0, sizeof(int) * NUM_TONE * NUM_TONE, minorHighStream);
    cudaMalloc(&device_minorLowNotes, sizeof(int) * NUM_TONE * NUM_TONE);
    cudaMemsetAsync(device_minorLowNotes, 0, sizeof(int) * NUM_TONE * NUM_TONE, minorLowStream);
    cudaMalloc(&device_majorChords, sizeof(int) * NUM_CHORD * NUM_CHORD);
    cudaMemsetAsync(device_majorChords, 0, sizeof(int) * NUM_TONE * NUM_TONE, majorChordStream);
    cudaMalloc(&device_minorChords, sizeof(int) * NUM_CHORD * NUM_CHORD);
    cudaMemsetAsync(device_minorChords, 0, sizeof(int) * NUM_TONE * NUM_TONE, minorChordStream);
}

void cuda_host_free() {
    cudaFreeHost(majorHighTone);
    cudaFreeHost(majorHighDur);
    cudaFreeHost(majorLowTone);
    cudaFreeHost(majorLowDur);
    cudaFreeHost(minorHighTone);
    cudaFreeHost(minorHighDur);
    cudaFreeHost(minorLowTone);
    cudaFreeHost(minorLowDur);
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

__device__
inline int cuda_getChordIndex(int curr_tone, int curr_dur, int prev_tone_1, int prev_dur_1, int tune) {
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

__device__
inline int cuda_getChordIndex(int curr_tone, int prev_tone) {
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

__global__ void note_kernel(int* device_Tone, int* device_Dur, int* device_mat, int use_len) {
    
}

__global__ void chord_kernel(int* device_tone, int* device_mat, int use_len) {
    
}

void cuda_note_count(int low_len, int high_len, int is_major) {
    if (is_major == 0)
    {     
        cudaMemcpyAsync(device_minorHighTone, minorHighTone, buffer_size, cudaMemcpyHostToDevice, minorHighStream);
        cudaMemcpyAsync(device_minorHighDur, minorHighDur, buffer_size, cudaMemcpyHostToDevice, minorHighStream);
        note_kernel<<1, NUM_THREADS, 0, minorHighStream>>(device_minorHighTone, device_minorHighDur, device_minorHigh, high_len);

        cudaMemcpyAsync(device_minorLowTone, minorLowTone, buffer_size, cudaMemcpyHostToDevice, minorLowStream);
        cudaMemcpyAsync(device_minorLowDur, minorLowDur, buffer_size, cudaMemcpyHostToDevice, minorLowStream);
        note_kernel<<1, NUM_THREADS, 0, minorHLowStream>>(device_minorLowTone, device_minorLowDur, device_minorLow, low_len);

        cudaMemcpyAsync(device_minorLowTone, minorLowTone, buffer_size, cudaMemcpyHostToDevice, minorChordStream);
        cudaMemcpyAsync(device_minorHighTone, minorHighTone, buffer_size, cudaMemcpyHostToDevice, minorChordStream);
        chord_kernel<<1, NUM_THREADS, 0, minorChordStream>>(device_minorLowTone, device_minorChords, low_len);
        chord_kernel<<1, NUM_THREADS, 0, minorChordStream>>(device_minorHighTone, device_minorChords, high_len);
    }
    else if (is_major == 1)
    {
        cudaMemcpyAsync(device_majorHighTone, majorHighTone, buffer_size, cudaMemcpyHostToDevice, majorHighStream);
        cudaMemcpyAsync(device_majorHighDur, majorHighDur, buffer_size, cudaMemcpyHostToDevice, majorHighStream);
        note_kernel<<1, NUM_THREADS, 0, majorHighStream>>(device_majorHighTone, device_majorHighDur, device_majorHigh, high_len);

        cudaMemcpyAsync(device_majorLowTone, majorLowTone, buffer_size, cudaMemcpyHostToDevice, majorLowStream);
        cudaMemcpyAsync(device_majorLowDur, majorLowDur, buffer_size, cudaMemcpyHostToDevice, majorLowStream);
        note_kernel<<1, NUM_THREADS, 0, majorHLowStream>>(device_minorLowTone, device_minorLowDur, device_majorLow, low_len);

        cudaMemcpyAsync(device_majorLowTone, majorLowTone, buffer_size, cudaMemcpyHostToDevice, majorChordStream);
        cudaMemcpyAsync(device_minorHighTone, minorHighTone, buffer_size, cudaMemcpyHostToDevice, majorChordStream);
        chord_kernel<<1, NUM_THREADS, 0, majorChordStream>>(device_majorLowTone, device_majorChords, low_len);
        chord_kernel<<1, NUM_THREADS, 0, majorChordStream>>(device_majorHighTone, device_majorChords, high_len);
    }
}

void cuda_stream_synch(int is_major) {
    if (is_major == 0)
    {
        cudaStreamSynchronize(minorHighStream);
        cudaStreamSynchronize(minorLowStream);
        cudaStreamSynchronize(minorChordStream);
    }
    else if (is_major == 1)
    {
        cudaStreamSynchronize(majorHighStream);
        cudaStreamSynchronize(majorLowStream);
        cudaStreamSynchronize(majorChordStream);
    }
}   

void cuda_to_host() {
    cudaMemcpyAsync(major_high, device_majorHighNotes, sizeof(int) * NUM_TONE * NUM_TONE, cudaMemcpyDeviceToHost, majorHighStream);
    cudaMemcpyAsync(major_low, device_majorLowNotes, sizeof(int) * NUM_TONE * NUM_TONE, cudaMemcpyDeviceToHost, majorLowStream);
    cudaMemcpyAsync(minor_high, device_minorHighNotes, sizeof(int) * NUM_TONE * NUM_TONE, cudaMemcpyDeviceToHost, minorHighStream);
    cudaMemcpyAsync(minor_low, device_minorLowNotes, sizeof(int) * NUM_TONE * NUM_TONE, cudaMemcpyDeviceToHost, minorLowStream);
    cudaMemcpyAsync(major_chord, device_majorChords, sizeof(int) * NUM_CHORD * NUM_CHORD, cudaMemcpyDeviceToHost, majorChordStream);
    cudaMemcpyAsync(minor_chord, device_minorChords, sizeof(int) * NUM_CHORD * NUM_CHORD, cudaMemcpyDeviceToHost, minorChordStream);
}

bool cuda_matrix_generation(char* major_path, char* minor_path) {
    std::cout << "Start parsing major & minor txt files" << std::endl;
    std::ifstream major_file(major_path);
    if (!major_file) {
        std::cerr << "Cannot open " << major_path << " !" <<std::endl;
        return false;
    }
    std::ifstream minor_file(minor_path);
    if (!minor_file) {
        std::cerr << "Cannot open " << minor_path << " !" <<std::endl;
        return false;        
    }

    cuda_stream_create();
    cuda_pinned_alloc();
    cuda_malloc();

    int tune = 1;
    int curr_tone = -1;
    int curr_dur = -1;
    int prev_tone_1 = -1;
    int prev_dur_1 = -1;
    int is_major = 1;
    int newMidi_flag = 0;

    int high_len = 0;
    int low_len = 0;
    int num_finished = 0;
    int is_major = 1;

    size_t split_idx;
    int cell_idx;
    std::string line;

    while (num_finished != 2) {
        if (is_major == 1) // major file
        { 
            if (!std::getline(major_file, line)) {
                cuda_note_count(low_len, high_len, is_major);
                cuda_stream_synch(is_major);
                high_len = 0;
                low_len = 0;
                is_major = 0;
                num_finished++;
                continue;
            }
            if (line.find('S') != std::string::npos && newMidi_flag == 0) { // start of a midi file
                curr_tone = -1;
                curr_dur = -1; 
                prev_tone_1 = -1; 
                prev_dur_1 = -1; 
                newMidi_flag = 1;
                continue;
            }
            if (line.find('L') != std::string::npos) { // low melody
                tune = 1;
                continue;
            }
            else if (line.find('H') != std::string::npos) { // high melody
                tune = 2;
                continue;
            }
            else if (line.find('X') != std::string::npos) { // end of a midi file
                newMidi_flag = 0;
                continue;
            }
            else if ((split_idx = line.find(' ')) != std::string::npos) {
                curr_tone = std::stoi(line.substr(0, split_idx));
                curr_dur = std::stoi(line.substr(split_idx));                
                if (tune == 1) {
                    majorLowTone[low_len] = curr_tone;
                    majorLowDur[low_len] = curr_dur;
                    low_len++;
                } 
                else if (tune == 2) {
                    majorHighTone[high_len] = curr_tone;
                    majorHighDur[high_len] = curr_dur;
                    high_len++;
                }

                prev_tone_1 = curr_tone;
                prev_dur_1 = curr_dur;

                if (high_len > BUFFER_LEN || low_len > BUFFER_LEN) {
                    cuda_notec_ount(low_len, high_len, is_major);
                    cuda_stream_synch(is_major);
                    high_len = 0;
                    low_len = 0;
                    is_major = 0;
                    continue;
                }
            }
        }
        else // minor file
        {
            if (!std::getline(minor_file, line)) {
                cuda_note_count(low_len, high_len, is_major);
                cuda_stream_synch(is_major);
                high_len = 0;
                low_len = 0;
                is_major = 1;
                num_finished++;
                continue;
            }
            if (line.find('S') != std::string::npos && newMidi_flag == 0) { // start of a midi file
                curr_tone = -1;
                curr_dur = -1; 
                prev_tone_1 = -1; 
                prev_dur_1 = -1; 
                newMidi_flag = 1;
                continue;
            }
            if (line.find('L') != std::string::npos) { // low melody
                tune = 1;
                continue;
            }
            else if (line.find('H') != std::string::npos) { // high melody
                tune = 2;
                continue;
            }
            else if (line.find('X') != std::string::npos) { // end of a midi file
                newMidi_flag = 0;
                continue;
            }
            else if ((split_idx = line.find(' ')) != std::string::npos) {
                curr_tone = std::stoi(line.substr(0, split_idx));
                curr_dur = std::stoi(line.substr(split_idx));                
                if (tune == 1) {
                    minorLowTone[low_len] = curr_tone;
                    minorLowDur[low_len] = curr_dur;
                    low_len++;
                } 
                else if (tune == 2) {
                    minorHighTone[high_len] = curr_tone;
                    minorHighDur[high_len] = curr_dur;
                    high_len++;
                }

                prev_tone_1 = curr_tone;
                prev_dur_1 = curr_dur;

                if (high_len > BUFFER_LEN || low_len > BUFFER_LEN) {
                    cuda_note_count(low_len, high_len, is_major);
                    cuda_stream_synch(is_major);
                    high_len = 0;
                    low_len = 0;
                    is_major = 1;
                    continue;
                }
            }
        }
    }
    
    // copy memory back to host
    cuda_to_host();
    cuda_stream_synch(0);
    cuda_stream_synch(1);

    // free pinned_memory
    cuda_host_free();
    cuda_free();
    cuda_stream_destroy();
}