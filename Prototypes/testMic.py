import pyaudio
import wave
import sys
import time
import msvcrt
import argparse


parser = argparse.ArgumentParser(description='Duration and filename data.')
parser.add_argument('--filename', help='output filename', default='output.wav', dest='filename')
parser.add_argument('--duration', help='duration of recording', default=10, dest='duration')

args = parser.parse_args()
WAVE_OUTPUT_FILENAME = args.filename
RECORD_SECONDS = int(args.duration)
def keyPressDetection():
    if msvcrt.kbhit():
        return ord(msvcrt.getch())
    return 0

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


print("* recording")
chordChangeTimes = []
frames = []
startTime= time.time()
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
    if keyPressDetection() == 32:
        chordChangeTimes.append(time.time() - startTime)

print("* done recording")
print("chord change times will be at:{}\n".format(chordChangeTimes))
stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()