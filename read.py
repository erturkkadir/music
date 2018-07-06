from __future__ import unicode_literals
import youtube_dl
import librosa


# ydl_opts = {
#    'format': 'bestaudio/best',
#    'postprocessors': [{
#        'key': 'FFmpegExtractAudio',
#        'preferredcodec': 'mp3',
#        'preferredquality': '192',
#    }],
# }

# with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#    ydl.download(['https://www.youtube.com/watch?v=1miwaIZwJbk'])
#    ydl.download(['https://www.youtube.com/watch?v=fI2UWHnjs10'])
#    ydl.download(['https://www.youtube.com/watch?v=PVuOTWmrx_w'])
#    ydl.download(['https://www.youtube.com/watch?v=VEKnYLHhYVg'])

# waveform and sampling rate
wave, sr = librosa.load('p1.mp3', mono=True)
wave = wave[::3]
print("Shape wave :", wave.shape)
mfcc = librosa.feature.mfcc(wave, sr=sr)
print("y", wave)
print("sr", sr)
print("mfcc : ", mfcc.shape)
