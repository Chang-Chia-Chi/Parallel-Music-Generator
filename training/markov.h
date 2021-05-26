#ifndef MARKOV_H
#define MARKOV_H

#define NUM_TONE 73
#define NUM_DURATION 15
#define NUM_CHORD 1728
#define NUM_NOTE NUM_TONE * NUM_DURATION
#define CHORD_BASE 99
#define BUFFER_LEN 200000
#define NUM_THREADS 1024

extern int* major_high;
extern int* major_low;
extern int* minor_high;
extern int* minor_low;
extern int* major_chord;
extern int* minor_chord;

extern int* cuda_majorHighTone;
extern int* cuda_majorHighDur;
extern int* cuda_majorLowTone;
extern int* cuda_majorLowDur;
extern int* cuda_minorHighTone;
extern int* cuda_minorHighDur;
extern int* cuda_minorLowTone;
extern int* cuda_minorLowDur;

extern int* device_majorHighTone;
extern int* device_majorHighDur;
extern int* device_majorLowTone;
extern int* device_majorLowDur;
extern int* device_minorHighTone;
extern int* device_minorHighDur;
extern int* device_minorLowTone;
extern int* device_minorLowDur;

extern int* device_majorHighNotes;
extern int* device_majorLowNotes;
extern int* device_minorHighNotes;
extern int* device_minorLowNotes;
extern int* device_majorChords;
extern int* device_minorChords;

// cuda functions called by main
void cuda_note_count(int* high_tone, int* hign_dur, int* low_tone, int* low_dur, int high_len, int low_len, int is_major, int tune);
void cuda_malloc();
void cuda_stream_create();
void cuda_to_host();
void cuda_free();
void cuda_stream_destroy();
void cuda_stream_synch(int is_major);

#endif 