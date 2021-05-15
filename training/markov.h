#ifndef MARKOV_H
#define MARKOV_H

#define NUM_TONE 73
#define NUM_DURATION 15
#define NUM_CHORD 1728
#define NUM_NOTE NUM_TONE * NUM_DURATION
#define CHORD_BASE 99

extern int* major_high;
extern int* major_low;
extern int* minor_high;
extern int* minor_low;
extern int* major_chord;
extern int* minor_chord;

#endif 