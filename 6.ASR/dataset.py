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
        S = (np.log10(S + 1e-5) + 5) / 5
        # (time, 512)
        return wav, S.T
            
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

            wav, mel = self._get_audio(self.data_files[index]['wav'])    
            if len(mel) > 450:
                index = (index + 1) % self.__len__()
                continue
                
            break
        
        return torch.FloatTensor(wav), torch.FloatTensor(mel), torch.LongTensor(text)
        
    def __len__(self):
        return len(self.data_files)
    
class KSponSpeechDataCollate():
    def __call__(self, batch):
        wav_lengths = []
        mel_lengths = []
        text_lengths = []
        for wav, mel, text in batch:
            wav_lengths.append(len(wav))
            mel_lengths.append(len(mel))
            text_lengths.append(len(text))
            
        wav_max_length = max(wav_lengths)
        mel_max_length = max(mel_lengths)
        text_max_length = max(text_lengths)
        
        wav_padded = torch.FloatTensor(len(batch), wav_max_length)
        wav_padded.zero_()
        mel_padded = torch.FloatTensor(len(batch), mel_max_length, 80)
        mel_padded.fill_(-5)
        mel_lengths = torch.from_numpy(np.array(mel_lengths)).long()
        
        text_padded = torch.LongTensor(len(batch), text_max_length)
        text_padded.zero_()
        text_lengths = torch.from_numpy(np.array(text_lengths)).long()
        
        for i, (wav, mel, text) in enumerate(batch):
            wav_padded[i, :len(wav)] = wav
            mel_padded[i, :len(mel)] = mel
            text_padded[i, :len(text)] = text
            
        outputs = {'wav': wav_padded,
                   'mel': mel_padded,
                   'mel_lengths': mel_lengths,
                   'text': text_padded,
                   'text_lengths': text_lengths
                  }
        
        return outputs
    
def to_cuda(batch):
    batch['mel'] = batch['mel'].cuda()
    batch['mel_lengths'] = batch['mel_lengths'].cuda()
    batch['text'] = batch['text'].cuda()
    batch['text_lengths'] = batch['text_lengths'].cuda()
    
    return batch