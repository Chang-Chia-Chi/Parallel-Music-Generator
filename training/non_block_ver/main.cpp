#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include "markov.h"

// global variable
int* major_high;
int* major_low;
int* minor_high;
int* minor_low;
int* major_chord;
int* minor_chord;

// host pinned-memory
int* majorHighTone;
int* majorHighDur;
int* majorLowTone;
int* majorLowDur;
int* minorHighTone;
int* minorHighDur;
int* minorLowTone;
int* minorLowDur;

void matrix_alloc() {
    // Allocation of major & minor notes transfer matrices //
    major_high = (int*)malloc(sizeof(int) * (NUM_NOTE * NUM_NOTE));
    major_low = (int*)malloc(sizeof(int) * (NUM_NOTE * NUM_NOTE));
    minor_high = (int*)malloc(sizeof(int) * (NUM_NOTE * NUM_NOTE));
    minor_low = (int*)malloc(sizeof(int) * (NUM_NOTE * NUM_NOTE));

    // Allocation of major & minor chords transfer matrices //
    major_chord = (int*)malloc(sizeof(int) * (NUM_CHORD * NUM_CHORD));
    minor_chord = (int*)malloc(sizeof(int) * (NUM_CHORD * NUM_CHORD));
}

void free_matrix() {
    free(major_high);
    free(major_low);
    free(minor_high);
    free(minor_low);
    free(major_chord);
    free(minor_chord);
}

void cuda_pinned_alloc() {
    cudaHostAlloc(&majorHighTone, sizeof(int) * BUFFER_LEN, cudaHostAllocMapped);
    cudaHostAlloc(&majorHighDur, sizeof(int) * BUFFER_LEN, cudaHostAllocMapped);
    cudaHostAlloc(&majorLowTone, sizeof(int) * BUFFER_LEN, cudaHostAllocMapped);
    cudaHostAlloc(&majorLowDur, sizeof(int) * BUFFER_LEN, cudaHostAllocMapped);
    cudaHostAlloc(&minorHighTone, sizeof(int) * BUFFER_LEN, cudaHostAllocMapped);
    cudaHostAlloc(&minorHighDur, sizeof(int) * BUFFER_LEN, cudaHostAllocMapped);
    cudaHostAlloc(&minorLowTone, sizeof(int) * BUFFER_LEN, cudaHostAllocMapped);
    cudaHostAlloc(&minorLowDur, sizeof(int) * BUFFER_LEN, cudaHostAllocMapped);
}

void cuda_pinned_free() {
    cudaFreeHost(majorHighTone);
    cudaFreeHost(majorHighDur);
    cudaFreeHost(majorLowTone);
    cudaFreeHost(majorLowDur);
    cudaFreeHost(minorHighTone);
    cudaFreeHost(minorHighDur);
    cudaFreeHost(minorLowTone);
    cudaFreeHost(minorLowDur);
}

void remove_old() {
    remove("MajorHighMatrix.txt");
    remove("MajorLowMatrix.txt");
    remove("MinorHighMatrix.txt");
    remove("MinorLowMatrix.txt");
    remove("ChordHighMatrix.txt");
    remove("ChordLowMatrix.txt");
}

void rename_new() {
    std::rename("MajorHighMatrixTemp.txt", "MajorHighMatrix.txt");
    std::rename("MajorLowMatrixTemp.txt", "MajorLowMatrix.txt");
    std::rename("MinorHighMatrixTemp.txt", "MinorHighMatrix.txt");
    std::rename("MinorLowMatrixTemp.txt", "MinorLowMatrix.txt");
    std::rename("ChordHighMatrixTemp.txt", "ChordHighMatrix.txt");
    std::rename("ChordLowMatrixTemp.txt", "ChordLowMatrix.txt");
}

/**
 * @brief Output generated matrices to txt files
 */
bool matrix_output() {
    std::cout << "Start output matrices trained" << std::endl;
    std::ofstream output_file;
    std::cout << "Start output major high note -- (1/6)" << std::endl;
    output_file.open("MajorHighMatrixTemp.txt");
    if (!output_file) {
        std::cerr << "Cannot open MajorHighMatrixTemp.txt !" <<std::endl;
        return false;
    }
    for (int i = 0; i < NUM_NOTE; i++) 
    {
        for (int j = 0; j < NUM_NOTE; j++)
        {
            // one line per prev_1 to curr
            // j = coordinate of curr note
            // i = coordinate of prev_1 note
            output_file << major_high[i * NUM_NOTE + j] << " ";
        }
        output_file << "\n";
    }
    output_file.close();
    std::cout << "Complete output major high note -- (1/6)" << std::endl;

    std::cout << "Start output major low note -- (2/6)" << std::endl;
    output_file.open("MajorLowMatrixTemp.txt");
    if (!output_file) {
        std::cerr << "Cannot open MajorLowMatrixTemp.txt !" <<std::endl;
        return false;
    }
    for (int i = 0; i < NUM_NOTE; i++) 
    {
        for (int j = 0; j < NUM_NOTE; j++)
        {
            // one line per prev_1 to curr
            // j = coordinate of curr note
            // i = NUM_NOTE = coordinate of prev_1 note
            output_file << major_low[i * NUM_NOTE + j] << " ";
        }
        output_file << "\n";
    }
    output_file.close();
    std::cout << "Complete output major low note -- (2/6)" << std::endl;

    std::cout << "Start output minor high note -- (3/6)" << std::endl;
    output_file.open("MinorHighMatrixTemp.txt");
    if (!output_file) {
        std::cerr << "Cannot open MinorHighMatrixTemp.txt !" <<std::endl;
        return false;
    }
    for (int i = 0; i < NUM_NOTE; i++) 
    {
        for (int j = 0; j < NUM_NOTE; j++)
        {
            // one line per prev_1 to curr
            // j = coordinate of curr note
            // i = NUM_NOTE = coordinate of prev_1 note
            output_file << minor_high[i * NUM_NOTE + j] << " ";
        }
        output_file << "\n";
    }
    output_file.close();
    std::cout << "Complete output minor high note -- (3/6)" << std::endl;

    std::cout << "Start output minor low note -- (4/6)" << std::endl;
    output_file.open("MinorLowMatrixTemp.txt");
    if (!output_file) {
        std::cerr << "Cannot open MinorLowMatrixTemp.txt !" <<std::endl;
        return false;
    }
    for (int i = 0; i < NUM_NOTE; i++) 
    {
        for (int j = 0; j < NUM_NOTE; j++)
        {
            // one line per prev_1 to curr
            // j = coordinate of curr note
            // i = NUM_NOTE = coordinate of prev_1 note
            output_file << minor_low[i * NUM_NOTE + j] << " ";
        }
        output_file << "\n";
    }
    output_file.close();
    std::cout << "Complete output minor low note -- (4/6)" << std::endl;

    std::cout << "Start output major chord -- (5/6)" << std::endl;
    output_file.open("MajorChordMatrixTemp.txt");
    if (!output_file) {
        std::cerr << "Cannot open MajorChordMatrixTemp.txt !" <<std::endl;
        return false;
    }
    for (int i = 0; i < NUM_CHORD ; i++) 
    {
        for (int j = 0; j < NUM_CHORD; j++)
        {
            // one line per prev_chord line
            // j = coordinate of curr chord
            // i = coordinate of prev chord
            output_file << major_chord[i * NUM_CHORD + j] << " ";
        }
        output_file << "\n";
    }
    output_file.close();
    std::cout << "Complete output major chord note -- (5/6)" << std::endl;    

    std::cout << "Start output minor chord -- (6/6)" << std::endl;
    output_file.open("MinorChordMatrixTemp.txt");
    if (!output_file) {
        std::cerr << "Cannot open MinorChordMatrixTemp.txt !" <<std::endl;
        return false;
    }
    for (int i = 0; i < NUM_CHORD ; i++) 
    {
        for (int j = 0; j < NUM_CHORD; j++)
        {
            // one line per prev_chord line
            // j = coordinate of curr chord
            // i = coordinate of prev chord
            output_file << minor_chord[i * NUM_CHORD + j] << " ";
        }
        output_file << "\n";
    }
    output_file.close();
    std::cout << "Complete output minor chord note -- (6/6)" << std::endl;

    remove_old();
    rename_new();
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage ./markovSeq.out <major note path> <minor note path>" << std::endl;
        exit(0);
    }
    std::cout << "Start matrix generation" << std::endl;
    bool success;
    char* major_path = argv[1];
    char* minor_path = argv[2];

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // allocate matrices
    matrix_alloc();

    auto t_start = high_resolution_clock::now();
    cuda_pinned_alloc();
    auto t_end = high_resolution_clock::now();
    auto t_spent = duration_cast<milliseconds>(t_end - t_start);
    std::cout << "Time spent for host pinned memory allocation: " << t_spent.count() << "ms\n";
    
    t_start = high_resolution_clock::now();
    cuda_stream_create();
    cuda_malloc();
    t_end = high_resolution_clock::now();
    t_spent = duration_cast<milliseconds>(t_end - t_start);
    std::cout << "Time spent for device memory allocation: " << t_spent.count() << "ms\n";

    int major_tune = 1;
    int major_curr_tone = -1;
    int major_curr_dur = -1;
    int major_prev_tone_1 = -1;
    int major_prev_dur_1 = -1;
    int major_new_Midi = 0;

    int minor_tune = 1;
    int minor_curr_tone = -1;
    int minor_curr_dur = -1;
    int minor_prev_tone_1 = -1;
    int minor_prev_dur_1 = -1;
    int minor_new_Midi = 0;

    int is_major = 1;
    int high_len = 0;
    int low_len = 0;
    int num_finished = 0;

    size_t split_idx;
    std::string line;

    // markov training through GPU
    std::cout << "Start parsing major & minor txt files" << std::endl;
    t_start = high_resolution_clock::now();

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
    while (num_finished != 2) {
        if (is_major == 1) // major file
        { 
            if (!std::getline(major_file, line)) {
                cuda_note_count(majorHighTone, majorHighDur, majorLowTone, majorLowDur, high_len, low_len, is_major, major_tune);
                cuda_stream_synch(is_major);
                high_len = 0;
                low_len = 0;
                is_major = 0;
                num_finished++;
                continue;
            }
            if (line.find('S') != std::string::npos && major_new_Midi == 0) { // start of a midi file
                major_curr_tone = -1;
                major_curr_dur = -1; 
                major_prev_tone_1 = -1; 
                major_prev_dur_1 = -1; 
                major_new_Midi = 1;
                continue;
            }
            if (line.find('L') != std::string::npos) { // low melody
                major_tune = 1;
                continue;
            }
            else if (line.find('H') != std::string::npos) { // high melody
                major_tune = 2;
                continue;
            }
            else if (line.find('X') != std::string::npos) { // end of a midi file
                major_new_Midi = 0;
                continue;
            }
            else if ((split_idx = line.find(' ')) != std::string::npos) {
                major_curr_tone = std::stoi(line.substr(0, split_idx));
                major_curr_dur = std::stoi(line.substr(split_idx));                
                if (major_tune == 1) {
                    majorLowTone[low_len] = major_curr_tone;
                    majorLowDur[low_len] = major_curr_dur;
                    low_len++;
                }
                else if (major_tune == 2) {
                    majorHighTone[high_len] = major_curr_tone;
                    majorHighDur[high_len] = major_curr_dur;
                    high_len++;
                }

                major_prev_tone_1 = major_curr_tone;
                major_prev_dur_1 = major_curr_dur;

                if (high_len > BUFFER_LEN || low_len > BUFFER_LEN) {
                    cuda_note_count(majorHighTone, majorHighDur, majorLowTone, majorLowDur, high_len, low_len, is_major, major_tune);
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
                cuda_note_count(minorHighTone, minorHighDur, minorLowTone, minorLowDur, high_len, low_len, is_major, minor_tune);
                cuda_stream_synch(is_major);
                high_len = 0;
                low_len = 0;
                is_major = 1;
                num_finished++;
                continue;
            }
            if (line.find('S') != std::string::npos && minor_new_Midi == 0) { // start of a midi file
                minor_curr_tone = -1;
                minor_curr_dur = -1; 
                minor_prev_tone_1 = -1; 
                minor_prev_dur_1 = -1; 
                minor_new_Midi = 1;
                continue;
            }
            if (line.find('L') != std::string::npos) { // low melody
                minor_tune = 1;
                continue;
            }
            else if (line.find('H') != std::string::npos) { // high melody
                minor_tune = 2;
                continue;
            }
            else if (line.find('X') != std::string::npos) { // end of a midi file
                minor_new_Midi = 0;
                continue;
            }
            else if ((split_idx = line.find(' ')) != std::string::npos) {
                minor_curr_tone = std::stoi(line.substr(0, split_idx));
                minor_curr_dur = std::stoi(line.substr(split_idx));                
                if (minor_tune == 1) {
                    minorLowTone[low_len] = minor_curr_tone;
                    minorLowDur[low_len] = minor_curr_dur;
                    low_len++;
                } 
                else if (minor_tune == 2) {
                    minorHighTone[high_len] = minor_curr_tone;
                    minorHighDur[high_len] = minor_curr_dur;
                    high_len++;
                }

                minor_prev_tone_1 = minor_curr_tone;
                minor_prev_dur_1 = minor_curr_dur;

                if (high_len > BUFFER_LEN || low_len > BUFFER_LEN) {
                    cuda_note_count(minorHighTone, minorHighDur, minorLowTone, minorLowDur, high_len, low_len, is_major, minor_tune);
                    cuda_stream_synch(is_major);
                    high_len = 0;
                    low_len = 0;
                    is_major = 1;
                    continue;
                }
            }
        }
    }

    t_end = high_resolution_clock::now();
    t_spent = duration_cast<milliseconds>(t_end - t_start);
    std::cout << "File parsing completed" << std::endl;
    std::cout << "Time spent for file parsing: " << t_spent.count() << "ms\n";

    // copy memory back to host
    t_start = high_resolution_clock::now();
    cuda_to_host();
    cuda_stream_synch(0);
    cuda_stream_synch(1);
    t_end = high_resolution_clock::now();
    t_spent = duration_cast<milliseconds>(t_end - t_start);
    std::cout << "Time spent for copy back to host: " << t_spent.count() << "ms\n";   
    std::cout << "Matrix generation successed" << std::endl;
    
    // output matrices to txt files
    success = matrix_output();
    if (success) {
        std::cout << "Matrix output successed" << std::endl;
    } else {
        std::cout << "Matrix output failed" << std::endl;
    }

    // free memory allocated
    t_start = high_resolution_clock::now();
    cuda_pinned_free();
    cuda_free();
    cuda_stream_destroy();
    t_end = high_resolution_clock::now();
    t_spent = duration_cast<milliseconds>(t_end - t_start);
    std::cout << "Time spent for free host and device memory: " << t_spent.count() << "ms\n";
    free_matrix();
    return 0;
}