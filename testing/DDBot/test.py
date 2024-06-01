import torch
from TTS.api import TTS
import re
import num2words
from pydub import AudioSegment
from pydub.playback import  play

def toInd(text: str):
    tmptext = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), text)
    tmptext = tmptext.replace("ten", "sepuluh")
    tmptext = tmptext.replace("eleven", "sebelas")
    tmptext = tmptext.replace("twelve", "dua belas")
    tmptext = tmptext.replace("thirteen", "tiga belas")
    tmptext = tmptext.replace("fourteen", "empat belas")
    tmptext = tmptext.replace("fifteen", "lima belas")
    tmptext = tmptext.replace("sixteen", "enam belas")
    tmptext = tmptext.replace("seventeen", "tujuh belas")
    tmptext = tmptext.replace("eighteen", "delapan belas")
    tmptext = tmptext.replace("nineteen", "sembilan belas")
    tmptext = tmptext.replace("twenty", "dua puluh")
    tmptext = tmptext.replace("thirty", "tiga puluh")
    tmptext = tmptext.replace("forty", "empat puluh")
    tmptext = tmptext.replace("fifty", "lima puluh")
    tmptext = tmptext.replace("sixty", "enam puluh")
    tmptext = tmptext.replace("seventy", "tujuh puluh")
    tmptext = tmptext.replace("eighty", "delapan puluh")
    tmptext = tmptext.replace("ninety", "sembilan puluh")
    tmptext = tmptext.replace("one hundred", "seratus")
    tmptext = tmptext.replace("one thousand", "seribu")

    tmptext = tmptext.replace("one", "satu")
    tmptext = tmptext.replace("two", "dua")
    tmptext = tmptext.replace("three", "tiga")
    tmptext = tmptext.replace("four", "empat")
    tmptext = tmptext.replace("five", "lima")
    tmptext = tmptext.replace("six", "enam")
    tmptext = tmptext.replace("seven", "tujuh")
    tmptext = tmptext.replace("eight", "delapan")
    tmptext = tmptext.replace("nine", "sembilan")
    tmptext = tmptext.replace("zero", "nol")

    tmptext = tmptext.replace("hundred", "ratus")
    tmptext = tmptext.replace("thousand", "ribu")
    tmptext = tmptext.replace("million", "juta")
    tmptext = tmptext.replace("and", "dan")
    return tmptext

text = "Python adalah bahasa pemrograman tujuan umum yang ditafsirkan, tingkat tinggi. Dibuat oleh Guido van Rossum dan pertama kali dirilis pada tahun 1991"

text = toInd(text)

tts = TTS(model_path='datasets/DDBot/indonesia/model.pth', config_path='datasets/DDBot/indonesia/config.json', gpu=True)
print(tts.is_multi_lingual)
tts.tts_to_file(text, file_path="output.wav", speaker='wibowo', language='indonesian')
play(AudioSegment.from_wav('output.wav'))
