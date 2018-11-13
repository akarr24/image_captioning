import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size) 
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout = 0.2, batch_first = True ) 
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size))
    
    def forward(self, features, captions):
        features = features.unsqueeze(1)
        cap_embedding = self.embed(captions[:,:-1])
        embedding_vector = torch.cat((features, cap_embedding), 1)
        lstm_out, self.hidden = self.lstm(embedding_vector)
        outputs = self.linear(lstm_out)
        return outputs
        
        

    def sample(self, features, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        for i in range(max_len):

            if states is None:

                inputs = features        
            else:     
                embeddings = self.embed(states)

                inputs = torch.cat((features, embeddings), 1)

                 
            lstm_out, hidden = self.lstm(inputs)

            out = self.linear(lstm_out)

            val, states = out.max(2)

                 
        output = states.tolist()[0]

        return output