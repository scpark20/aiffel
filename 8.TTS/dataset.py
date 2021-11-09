import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import librosa

def get_utf8_values(text):
    text_utf = text.encode()
    ts = [0]
    for t in text_utf:
        ts.append(t)
    utf8_values = np.array(ts)

    return utf8_values

class KSSDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, root_dir, meta_file, max_length):
        self.hparams = hparams
        self.data_files = self._get_data_files(root_dir, meta_file)
        self.max_length = max_length
        self.mel_matrix = librosa.filters.mel(sr=hparams.sampling_rate, 
                                              n_fft=hparams.win_length, 
                                              n_mels=hparams.n_mel_channels, 
                                              fmin=hparams.mel_fmin, 
                                              fmax=hparams.mel_fmax)
        
    def _get_data_files(self, root_dir, meta_file):
        meta_path = root_dir + meta_file
        data_files = []
        with open(meta_path, 'r') as f:
            l = f.readline().strip()
            while l:
                l = l.split('|')
                wav_file = l[0]
                wav_path = root_dir + 'kss/' + wav_file
                text = l[2]
                data_files.append((wav_path, text, wav_file))
                l = f.readline().strip()
                
        return data_files
    
    def _get_mel(self, wav_file):
        wav, _ = librosa.core.load(wav_file, sr=22050)
        wav, _ = librosa.effects.trim(wav, top_db=40)
        S = librosa.stft(wav, n_fft=self.hparams.win_length, hop_length=self.hparams.hop_length, 
                         win_length=self.hparams.win_length, window="hann", pad_mode="reflect")
        S = np.abs(S).T
        mel = S @ self.mel_matrix.T
        mel = np.log10(np.maximum(1e-5, mel))
        
        return mel.T
    
    def __getitem__(self, index):
        while True:
            audio = self._get_mel(self.data_files[index][0])
            if len(audio) > self.max_length:
                index = (index + 1) % self.__len__()
                continue
            text = get_utf8_values(self.data_files[index][1])
            break
        
        return torch.LongTensor(text), torch.FloatTensor(audio), self.data_files[index][2]
    
    def __len__(self):
        return len(self.data_files)
    
class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        wav_files = []
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            wav_files.append(batch[ids_sorted_decreasing[i]][2])

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, wav_files
    
