import os
import time
import glob
import asyncio
import aiofiles
import multiprocessing as mp

async def write_major(major_queue:mp.Queue):
    async with aiofiles.open("Major_Notes.txt", "w") as txt_write:
        while True:
            # queue.get() is already block
            txt_str = major_queue.get()
            if txt_str == "END":
                break
            await txt_write.write(txt_str)
                
    print("Complete Major Notes Processing")

async def write_minor(minor_queue:mp.Queue):
    async with aiofiles.open("Minor_Notes.txt", "w") as txt_write:
        while True:
            # queue.get() is already block
            txt_str = minor_queue.get()
            if txt_str == "END":
                break
            await txt_write.write(txt_str)

    print("Complete Minor Notes Processing")
    
async def read_txt(pid:int, txt_files:list, major_queue:mp.Queue, minor_queue:mp.Queue):
    num_files = len(txt_files)
    if pid == 0:
        process_files = txt_files[:num_files//2]
    elif pid == 1:
        process_files = txt_files[num_files//2:]
    else:
        print("Code designed only for 2 processes to read txt files")
        exit(0)
        
    for txt_file in process_files:
        txt_str = ""
        try:
            async with aiofiles.open(txt_file, "r") as txt_read:
                mood = await txt_read.readline()
                txt_str = await txt_read.read()
            if mood == "major\n":
                major_queue.put(txt_str)
            elif mood == "minor\n":
                minor_queue.put(txt_str)
            else:
                print("Read Skipped ! " + txt_file)
                continue
        except:
            print("Read Failed ! " + txt_file)
            continue

    major_queue.put("END")
    minor_queue.put("END")

def get_TXTpaths(current_folder:str):
    os.chdir(current_folder)
    txt_files = glob.glob("**/*.txt", recursive=True)
    return txt_files

def process(func, *args):
    asyncio.run(func(*args))

def main():
    start_t = time.time()
    print("Start Combination !")
    current_folder = os.getcwd()
    txt_files = get_TXTpaths(current_folder)

    manager = mp.Manager()
    major_queue = manager.Queue()
    minor_queue = manager.Queue()

    p1_read_txt = mp.Process(target=process, args=(read_txt, 0, txt_files, major_queue, minor_queue))
    p2_read_txt = mp.Process(target=process, args=(read_txt, 1, txt_files, major_queue, minor_queue))
    p3_write_major = mp.Process(target=process, args=(write_major, major_queue))
    p4_write_minor = mp.Process(target=process, args=(write_minor, minor_queue))
    processes = [p1_read_txt, p2_read_txt, p3_write_major, p4_write_minor]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("End Combination in: {:.2f} sec!".format(time.time()-start_t))

if __name__ == "__main__":
    main()