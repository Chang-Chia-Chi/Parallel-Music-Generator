#include <iostream>
#include <string>
#include <fstream>      
#include <chrono>
#include "markov.h"

// global variable
int* major_high;
int* major_low;
int* minor_high;
int* minor_low;
int* major_chord;
int* minor_chord;

// buffer
note_info* majorHighBuff;
note_info* majorLowBuff;
note_info* minorHighBuff;
note_info* minorLowBuff;

int major_high_len;
int major_low_len;
int minor_high_len;
int minor_low_len;

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

void buffer_alloc() {
    // Allocation of major & minor notes transfer matrices //
    majorHighBuff = (note_info*)malloc(sizeof(note_info) * BUFFER_LEN);
    majorLowBuff = (note_info*)malloc(sizeof(note_info) * BUFFER_LEN);
    minorHighBuff = (note_info*)malloc(sizeof(note_info) * BUFFER_LEN);
    minorLowBuff = (note_info*)malloc(sizeof(note_info) * BUFFER_LEN);
}

void free_buffer() {
    free(majorHighBuff);
    free(majorLowBuff);
    free(minorHighBuff);
    free(minorLowBuff);
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
 * @brief Compute index of note matrix according to current and previous tone & duration
 * 
 * @param curr_tone     Current tone
 * @param curr_dur      Duration of current tone
 * @param prev_tone_1   Previous tone
 * @param prev_dur_1    Duration of previous tone
 */
inline int get_note_index(int curr_tone, int curr_dur, int prev_tone_1, int prev_dur_1, int tune) {
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

/** 
 * @brief Compute index of chord matrix according to current tone and previous tone
 * 
 * @param curr_tone      Current  tone
 * @param prev_tone      previous tone
 */
inline int get_chord_index(int curr_tone, int prev_tone) {
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
    while (std::getline(major_file, line)) {
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
    while (std::getline(minor_file, line)) {
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

void matrix_generation() {
    int i;
    int index, curr_tone, curr_dur, prev_tone, prev_dur, tune;
    for (i = 1; i < major_high_len; i++) {
        curr_tone = majorHighBuff[i].tone;
        curr_dur = majorHighBuff[i].dur;
        tune = majorHighBuff[i].tune;
        prev_tone = majorHighBuff[i - 1].tone;
        prev_dur = majorHighBuff[i - 1].dur;
        if (curr_tone < CHORD_BASE && prev_tone != -1) {
            index = get_note_index(curr_tone, curr_dur, prev_tone, prev_dur, tune);
            if (index != -1) {
                major_high[index]++;
            }
        }
        else if (curr_tone >= CHORD_BASE && prev_tone != -1) {
            index = get_chord_index(curr_tone, prev_tone);
            if (index != -1) {
                major_chord[index]++;
            }
        }
    }

    for (i = 1; i < major_low_len; i++) {
        curr_tone = majorLowBuff[i].tone;
        curr_dur = majorLowBuff[i].dur;
        tune = majorLowBuff[i].tune;
        prev_tone = majorLowBuff[i - 1].tone;
        prev_dur = majorLowBuff[i - 1].dur;
        if (curr_tone < CHORD_BASE && prev_tone != -1) {
            index = get_note_index(curr_tone, curr_dur, prev_tone, prev_dur, tune);
            if (index != -1) {
                major_low[index]++;
            }
        }
        else if (curr_tone >= CHORD_BASE && prev_tone != -1) {
            index = get_chord_index(curr_tone, prev_tone);
            if (index != -1) {
                major_chord[index]++;
            }
        }
    }

    for (i = 1; i < minor_high_len; i++) {
        curr_tone = minorHighBuff[i].tone;
        curr_dur = minorHighBuff[i].dur;
        tune = minorHighBuff[i].tune;
        prev_tone = minorHighBuff[i - 1].tone;
        prev_dur = minorHighBuff[i - 1].dur;
        if (curr_tone < CHORD_BASE && prev_tone != -1) {
            index = get_note_index(curr_tone, curr_dur, prev_tone, prev_dur, tune);
            if (index != -1) {
                minor_high[index]++;
            }
        }
        else if (curr_tone >= CHORD_BASE && prev_tone != -1) {
            index = get_chord_index(curr_tone, prev_tone);
            if (index != -1) {
                minor_chord[index]++;
            }
        }
    }

    for (i = 1; i < minor_low_len; i++) {
        curr_tone = minorLowBuff[i].tone;
        curr_dur = minorLowBuff[i].dur;
        tune = minorLowBuff[i].tune;
        prev_tone = minorLowBuff[i - 1].tone;
        prev_dur = minorLowBuff[i - 1].dur;
        if (curr_tone < CHORD_BASE && prev_tone != -1) {
            index = get_note_index(curr_tone, curr_dur, prev_tone, prev_dur, tune);
            if (index != -1) {
                minor_low[index]++;
            }
        }
        else if (curr_tone >= CHORD_BASE && prev_tone != -1) {
            index = get_chord_index(curr_tone, prev_tone);
            if (index != -1) {
                minor_chord[index]++;
            }
        }
    }
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

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    bool success;
    char* major_path = argv[1];
    char* minor_path = argv[2];

    // allocate matrices
    auto t_start = high_resolution_clock::now();
    matrix_alloc();
    buffer_alloc();
    auto t_end = high_resolution_clock::now();
    auto t_spent = duration_cast<milliseconds>(t_end - t_start);
    std::cout << "Time spent for host memory allocation: " << t_spent.count() << "ms\n";

    std::cout << "Start parsing major & minor txt files" << std::endl;
    success = file_parsing(major_path, minor_path);
    if (success) {
        std::cout << "File parsing successed" << std::endl;
    } else {
        std::cout << "File parsing failed" << std::endl;
    }

    t_start = high_resolution_clock::now();
    matrix_generation();
    t_end = high_resolution_clock::now();
    t_spent = duration_cast<milliseconds>(t_end - t_start);
    std::cout << "Time spent for matrix generation: " << t_spent.count() << "ms\n";

    // output matrices to txt files 
    success = matrix_output();
    if (success) {
        std::cout << "Matrix output successed" << std::endl;
    } else {
        std::cout << "Matrix output failed" << std::endl;
    }

    t_start = high_resolution_clock::now();
    free_buffer();
    free_matrix();
    t_end = high_resolution_clock::now();
    t_spent = duration_cast<milliseconds>(t_end - t_start);
    std::cout << "Time spent for free host and device memory: " << t_spent.count() << "ms\n";
    return 0;
}