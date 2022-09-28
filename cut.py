from pydub import AudioSegment

filePath = 'C:/Users/Administrator/Desktop/111/0/'


# 操作函数
def get_wav_make(dataDir):
    sound = AudioSegment.from_wav(dataDir)
    duration = sound.duration_seconds * 1000  # 音频时长（ms）

    begin = 0
    end = int(duration)
    sound = sound[begin:end]
    aa = []
    end = int(sound.duration_seconds * 1000)

    cc = 5

    for i in range(end):
        if i % cc == 0:
            aa.append(i)
    dd = int(3000 / cc)
    bb = aa[:-dd]



    for j in range(len(bb)):
        cut_wav = sound[aa[j]:aa[j+dd]]  # 以毫秒为单位截取[begin, end]区间的音频

        cut_wav.export(filePath + str(j) +'test.wav', format='wav')




if __name__ == '__main__':
    get_wav_make('C:/Users/Administrator/Desktop/111/a/video.wav')
