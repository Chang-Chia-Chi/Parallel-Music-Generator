#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include "markov.h"

int major_high_len;
int major_low_len;
int minor_high_len;
int minor_low_len;

void remove_old() {
    remove("MajorHighMatrix.txt");
    remove("MajorLowMatrix.txt");
    remove("MinorHighMatrix.txt");
    remove("MinorLowMatrix.txt");
    remove("MajorChordMatrix.txt");
    remove("MinorChordMatrix.txt");
}

void rename_new() {
    std::rename("MajorHighMatrixTemp.txt", "MajorHighMatrix.txt");
    std::rename("MajorLowMatrixTemp.txt", "MajorLowMatrix.txt");
    std::rename("MinorHighMatrixTemp.txt", "MinorHighMatrix.txt");
    std::rename("MinorLowMatrixTemp.txt", "MinorLowMatrix.txt");
    std::rename("MajorChordMatrixTemp.txt", "MajorChordMatrix.txt");
    std::rename("MinorChordMatrixTemp.txt", "MinorChordMatrix.txt");
}

bool file_parsing(char* major_path, char* minor_path) {
    int curr_tone = 0;
    int curr_dur = 0;
    int tune = 1;
    int newMidi_flag = 0;
    size_t split_idx;

    // Major Notes
    major_low_len = 0;
    major_high_len = 0;
    std::cout << "Start major notes parsing" << std::endl;
    std::ifstream major_file(major_path);
    if (!major_file) {
        std::cerr << "Cannot open " << major_path << " !" <<std::endl;
        return false;
    }
    std::string line;
    while (std::getline(major_file, line) && major_high_len < BUFFER_LEN && major_low_len < BUFFER_LEN) {
        if (line.find('S') != std::string::npos && newMidi_flag == 0) { // start of a midi file
            newMidi_flag = 1;
        }
        if (line.find('L') != std::string::npos) { // low melody
            tune = 1;
        }
        else if (line.find('H') != std::string::npos) { // high melody
            tune = 2;
        }
        else if (line.find('X') != std::string::npos) { // end of a midi file
            newMidi_flag = 0;
        }
        else if ((split_idx = line.find(' ')) != std::string::npos) {
            curr_tone = std::stoi(line.substr(0, split_idx));
            curr_dur = std::stoi(line.substr(split_idx));
            // first note do nothing
            if (tune == 1) {
                majorLowBuff[major_low_len].tone = curr_tone;
                majorLowBuff[major_low_len].dur = curr_dur;
                majorLowBuff[major_low_len].tune = tune;
                major_low_len++;
            } 
            else if (tune == 2) {
                majorHighBuff[major_high_len].tone = curr_tone;
                majorHighBuff[major_high_len].dur = curr_dur;
                majorHighBuff[major_high_len].tune = tune;
                major_high_len++;
            }
        }
    }
    major_file.close();

    // Minor Notes
    minor_low_len = 0;
    minor_high_len = 0;
    std::cout << "Start minor notes parsing" << std::endl;
    std::ifstream minor_file(minor_path);
    if (!minor_file) {
        std::cerr << "Cannot open " << minor_path << " !" <<std::endl;
        return false;        
    }
    while (std::getline(minor_file, line) && minor_high_len < BUFFER_LEN && minor_low_len < BUFFER_LEN) {
        if (line.find('S') != std::string::npos && newMidi_flag == 0) { // start of a midi file
            newMidi_flag = 1;
        }
        if (line.find('L') != std::string::npos) { // low melody
            tune = 1;
        }
        else if (line.find('H') != std::string::npos) { // high melody
            tune = 2;
        }
        else if (line.find('X') != std::string::npos) { // end of a midi file
            newMidi_flag = 0;
        }
        else if ((split_idx = line.find(' ')) != std::string::npos) {
            curr_tone = std::stoi(line.substr(0, split_idx));
            curr_dur = std::stoi(line.substr(split_idx));
            // first note do nothing
            if (tune == 1) {
                minorLowBuff[minor_low_len].tone = curr_tone;
                minorLowBuff[minor_low_len].dur = curr_dur;
                minorLowBuff[minor_low_len].tune = tune;
                minor_low_len++;
            } 
            else if (tune == 2) {
                minorHighBuff[minor_high_len].tone = curr_tone;
                minorHighBuff[minor_high_len].dur = curr_dur;
                minorHighBuff[minor_high_len].tune = tune;
                minor_high_len++;
            }
        }
    }
    minor_file.close();
    return true;
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
        std::cerr << "Usage ./main <major note path> <minor note path>" << std::endl;
        exit(0);
    }
    std::cout << "Start matrix generation" << std::endl;
    bool success;
    char* major_path = argv[1];
    char* minor_path = argv[2];

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::microseconds;

    // allocate host memory
    matrix_alloc();

    // allocate device memory
    cuda_malloc();
    
    std::cout << "Start parsing major & minor txt files" << std::endl;
    success = file_parsing(major_path, minor_path);
    if (success) {
        std::cout << "File parsing successed" << std::endl;
    } else {
        std::cout << "File parsing failed" << std::endl;
    }

    // markov training through GPU
    buffer_copy(majorHighBuff, majorLowBuff, major_high_len, major_low_len,
                minorHighBuff, minorLowBuff, minor_high_len, minor_low_len);

    // markov training through GPU
    cuda_note_count(major_high_len, major_low_len, minor_high_len, minor_low_len);

    // copy memory back to host
    cuda_to_host();
    
    //  output matrices to txt files
    success = matrix_output();
    if (success) {
        std::cout << "Matrix output successed" << std::endl;
    } else {
        std::cout << "Matrix output failed" << std::endl;
    }

    // free memory allocated
    cuda_free();
    free_matrix();
    return 0;
}