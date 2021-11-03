import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from os import listdir
from os.path import isdir, isfile, join
from kspon_jamo import text_to_tokens, tokens_to_text, n_symbols, normalize_ksponspeech, SOS, EOS

class KSponSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_files = [{'wav': join(data_dir, file),\
                            'txt': join(data_dir, file[:-3] + 'txt')} for file in listdir(data_dir) if '.wav' in file]

    def _get_audio(self, file):
        # (time,)
        wav, _ = librosa.core.load(file, sr=16000, mono=True)
        # (512, time)
        S = librosa.feature.melspectrogram(wav, sr=16000, n_fft=1024, n_mels=80, hop_length=256, power=1.0)
        S = np.log10(S + 1e-5) 
        # (time, 512)
        return S.T
            
    def _get_text(self, file):
        with open(file, 'r', encoding='cp949') as f:
            l = f.read()
            l = normalize_ksponspeech(l)
            array = text_to_tokens(l)
        # Insert SOS and EOS
        array = np.concatenate([[SOS], array, [EOS]])
        return array
        
    def __getitem__(self, index):
        while True:
            text = self._get_text(self.data_files[index]['txt'])
            if len(text) > 180:
                index = (index + 1) % self.__len__()
                continue

            audio = self._get_audio(self.data_files[index]['wav'])    
            if len(audio) > 450:
                index = (index + 1) % self.__len__()
                continue
                
            break
        
        return torch.FloatTensor(audio), torch.LongTensor(text)
        
    def __len__(self):
        return len(self.data_files)
    
class KSponSpeechDataCollate():
    def __call__(self, batch):
        audio_lengths = []
        text_lengths = []
        for audio, text in batch:
            audio_lengths.append(len(audio))
            text_lengths.append(len(text))
            
        audio_max_length = max(audio_lengths)
        text_max_length = max(text_lengths)
        
        audio_padded = torch.FloatTensor(len(batch), audio_max_length, 80)
        audio_padded.fill_(-5)
        audio_lengths = torch.from_numpy(np.array(audio_lengths)).long()
        
        text_padded = torch.LongTensor(len(batch), text_max_length)
        text_padded.zero_()
        text_lengths = torch.from_numpy(np.array(text_lengths)).long()
        
        for i, (audio, text) in enumerate(batch):
            audio_padded[i, :len(audio)] = audio
            text_padded[i, :len(text)] = text
            
        outputs = {'audio': audio_padded,
                   'audio_lengths': audio_lengths,
                   'text': text_padded,
                   'text_lengths': text_lengths
                  }
        
        return outputs