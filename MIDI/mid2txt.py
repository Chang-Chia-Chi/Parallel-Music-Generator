import os
import time
import glob
import music21
import asyncio
import aiofiles
from constant import *
import multiprocessing as mp

def parse_files(files:list, num_files:int, num_p:int, pid:int, current_folder:str, output_folder:str):
    interval = num_files // num_p
    start = pid * interval
    end = (pid+1) * interval
    if pid == num_p - 1:
        end = num_files
    tasks = []
    for i in range(start, end):
        file_name = files[i]
        print("Start parsing File " + file_name)
        if not file_name.endswith(".mid"):
            print("midiname should be a midi format file!")
            return ""

        midi_path = os.sep.join([current_folder, file_name])
        txt_str = mid2txt(midi_path)
        if txt_str != "major\nX\n" and txt_str != "minor\nX\n":
            output_name = file_name[:-4] + ".txt"
            output_path = os.sep.join([output_folder, output_name])
            tasks.append(asyncio.ensure_future(write_info(txt_str, output_path)))
        print("Complete parsing File " + file_name)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        print("create new folder" + output_folder)
    
    print("Process {} start writing txt file".format(pid))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))
    print("Process {} complete writing txt file".format(pid))

def mid2txt(midi_path:str):
    if not os.path.isfile(midi_path):
        print("midi file not exists")
        return ""

    score = music21.converter.parse(midi_path)
    key = score.analyze("key")
    mode = key.mode
    txt_str = mode+"\n"
    for part in score.parts:
        notes_list = [n.pitch for n in part.notes if n.isNote]
        chords_list = [c.pitches[0] for c in part.notes if c.isChord]
        notes_list += chords_list
        if not notes_list:
            print("No element in this part")
            continue
        
        txt_str += "S\n"
        notes_list = sorted(notes_list)
        num_notes = len(notes_list)
        mid_note = notes_list[num_notes//2]
        if mid_note < music21.pitch.Pitch("C4"):
            txt_str += "HIGH\n"
        else:
            txt_str += "LOW\n"
        
        for note in part.notesAndRests:
            if note.isNote:
                pitch = int(note.pitch.ps) - LOWEST_NOTE
                if pitch < 0:
                    pitch = pitch % OCTAVE_NUM
                elif pitch >= NOTE_RANGE:
                    pitch = pitch % OCTAVE_NUM + HIGHEST_OFFSET
            elif note.isChord:
                bit = 0
                pitch = 0
                for p in sorted(note.pitches):
                    pitch += (int(p.ps) % 12)*pow(OCTAVE_NUM, bit)
                    bit += 1
                pitch += CHORD_BASE
            else:
                pitch = REST

            duration = note.duration.quarterLength
            duration_diff = [abs(duration-num) for num in NOTE_DURATIONS]
            duration_index = duration_diff.index(min(duration_diff))
            txt_str += "{:d} {:.3f}\n".format(pitch, duration_index)

        txt_str += "E\n"
    txt_str += "X\n"

    return txt_str

async def write_info(txt_str:str, output_path:str):
    async with aiofiles.open(output_path, 'w') as f:
        await f.write(txt_str)

def get_MIDIpaths(current_folder:str):
    os.chdir(current_folder)
    midi_files = glob.glob("**/*.mid", recursive=True)
    return midi_files

def main():
    start_t = time.time()
    print("Start Conversion !")
    current_folder = os.getcwd()
    files = get_MIDIpaths(current_folder)
    num_files = len(files)
    output_folder = TXT_FOLDER
    num_p = PROCESS_NUMBER

    processes = [mp.Process(target=parse_files, args=(files, num_files, num_p, i, current_folder, output_folder)) for i in range(num_p)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("End Conversion in: {:.2f} sec!".format(time.time()-start_t))

if __name__ == "__main__":
    main()
    