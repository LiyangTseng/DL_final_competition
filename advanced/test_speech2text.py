import warnings
warnings.filterwarnings('ignore')
import csv
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from speech2text_model import SpeechRecognitionModel
from dataset import TextTransform

use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

def GreedyDecoder(output, blank_label=28, collapse_repeated=True):
	arg_maxes = torch.argmax(output, dim=2)
	decodes = []
	for i, args in enumerate(arg_maxes):
		decode = []
		for j, index in enumerate(args):
			if index != blank_label:
				if collapse_repeated and j != 0 and index == args[j -1]:
					continue
				decode.append(index.item())
		decodes.append(text_transform.int_to_text(decode))
	return decodes


text_transform = TextTransform()

hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": 5e-4,
        "batch_size": 10,
        "epochs": 10
    }

model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)
ckpt_path = 'speech2text.pt'
model.load_state_dict(torch.load(ckpt_path))
model.eval()

test_data_num = 4721

with torch.no_grad():
    with open('speech2text_prediction.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for index in tqdm(range(test_data_num)):
            melspec = torch.load('Features/test/{:06}.pt'.format(index))
            spec = melspec.squeeze(0).transpose(0, 1)
            spec = nn.utils.rnn.pad_sequence([spec], batch_first=True).unsqueeze(1).transpose(2, 3).to(device)
            output = model(spec)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            # output = output.transpose(0, 1) # (time, batch, n_class)
            decoded_preds = GreedyDecoder(output)
                    
            writer.writerow([index, decoded_preds])
