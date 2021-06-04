#ifndef MARKOV_H
#define MARKOV_H

#define NUM_TONE 73
#define NUM_DURATION 15
#define NUM_CHORD 1728
#define NUM_NOTE NUM_TONE * NUM_DURATION
#define CHORD_BASE 101
#define BUFFER_LEN 7500000
#define NUM_THREADS 1024

struct note_info {
    int tone;
    int dur;
    int tune;
};

extern int* major_high;
extern int* major_low;
extern int* minor_high;
extern int* minor_low;
extern int* major_chord;
extern int* minor_chord;

extern note_info* majorHighBuff;
extern note_info* majorLowBuff;
extern note_info* minorHighBuff;
extern note_info* minorLowBuff;

extern note_info* device_majorHighBuff;
extern note_info* device_majorHighBuff;
extern note_info* device_majorLowBuff;
extern note_info* device_majorLowBuff;

extern int* device_majorHighNotes;
extern int* device_majorLowNotes;
extern int* device_minorHighNotes;
extern int* device_minorLowNotes;
extern int* device_majorChords;
extern int* device_minorChords;

// cuda functions called by main
void cuda_note_count(int major_high_len, int major_low_len, int minor_high_len, int minor_low_len);
void buffer_copy(note_info* major_high_buff, note_info* majorlow_buff, int major_high_len, int major_low_len,
                 note_info* minor_high_buff, note_info* minorlow_buff, int minor_high_len, int minor_low_len);
void matrix_alloc();
void free_matrix();
void cuda_malloc();
void cuda_to_host();
void cuda_free();
void cuda_note_count(int high_len, int low_len, int is_major);

#endif 