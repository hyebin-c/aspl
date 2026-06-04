import torch
import torch.nn as nn

from palm.encoders import AudioEncoder
from palm.encoders import TextEncoder


class PromptLearner(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        classnames = args.classnames
        n_cls = len(classnames)

        ctx_dim = args.ctx_dim # 1024

        print("Initializing a generic context")
        ctx = torch.empty(n_cls, ctx_dim) # [n_cls, ctx_dim] 
        torch.nn.init.normal_(ctx, std=0.02)
        self.ctx = torch.nn.Parameter(ctx) # 클래스마다 하나씩 lambda 값을 가짐. 
        self.n_cls = n_cls
        self.lambdas = nn.Parameter(torch.rand(n_cls))
        
    def forward(self, audio_features, text_features):
        
        # audio_features: [16, 1024], 16 = Number of audio input
        # text_features: [4, 1024]
        lambdas = torch.sigmoid(self.lambdas).reshape(-1,1)  # [n_cls, 1]
        updated_text_features = (1-lambdas)*text_features + (lambdas*self.ctx)     # [n_text_prompts, 1024]
        updated_text_features = updated_text_features / updated_text_features.norm(dim=-1, keepdim=True)

        return audio_features, updated_text_features

class CustomPENGI(nn.Module):
    def __init__(self,args,pengi):
        super().__init__()

        self.args = args
        pengi_args  = pengi.args
        self.pengi_args = pengi_args

        if args.aspl == 1: #aspl
            self.audio_encoder = AudioEncoder(
                    pengi_args.audioenc_name, pengi_args.out_emb, pengi_args.d_proj,
                    pengi_args.sampling_rate, pengi_args.window_size, pengi_args.hop_size, pengi_args.mel_bins, pengi_args.fmin, pengi_args.fmax, pengi_args.classes_num, 
                    pengi_args.specaug, pengi_args.mixup, pengi_args.use_pretrained_audioencoder, pengi_args.freeze_audio_encoder_weights,
                    pengi_args.use_precomputed_melspec, pengi_args.pretrained_audioencoder_path, spectral=True, prompt_audio_dim=1)
        
        elif args.aspl == 2: # aspl+
            self.audio_encoder = AudioEncoder(
                    pengi_args.audioenc_name, pengi_args.out_emb, pengi_args.d_proj,
                    pengi_args.sampling_rate, pengi_args.window_size, pengi_args.hop_size, pengi_args.mel_bins, pengi_args.fmin, pengi_args.fmax, pengi_args.classes_num, 
                    pengi_args.specaug, pengi_args.mixup, pengi_args.use_pretrained_audioencoder, pengi_args.freeze_audio_encoder_weights,
                    pengi_args.use_precomputed_melspec, pengi_args.pretrained_audioencoder_path, spectral=True, prompt_audio_dim=1,
                    prompt_audio_blocks=[1])
        else:
            self.audio_encoder = AudioEncoder(
                    pengi_args.audioenc_name, pengi_args.out_emb, pengi_args.d_proj,
                    pengi_args.sampling_rate, pengi_args.window_size, pengi_args.hop_size, pengi_args.mel_bins, pengi_args.fmin, pengi_args.fmax, pengi_args.classes_num, 
                    pengi_args.specaug, pengi_args.mixup, pengi_args.use_pretrained_audioencoder, pengi_args.freeze_audio_encoder_weights,
                    pengi_args.use_precomputed_melspec, pengi_args.pretrained_audioencoder_path)
        


        self.text_encoder = TextEncoder(
                    pengi_args.d_proj, 
                    pengi_args.text_model, pengi_args.transformer_embed_dim,
                    pengi_args.freeze_text_encoder_weights)


        # load the weights of the pengi pre-trained audio and text encoders
        print("\n\nPALM: loading the weights of the pengi pre-trained audio and text encoders ...\n\n")
        self.audio_encoder.load_state_dict(pengi.model.audio_encoder.state_dict(), strict=False) # because of soft prompt
        self.text_encoder.load_state_dict(pengi.model.caption_encoder.state_dict())

        self.audio_encoder.eval()
        self.text_encoder.eval()

        self.prompt_learner = PromptLearner(args)

        self.process_text = pengi.preprocess_text
        self.device = args.device


    def forward(self, audio):

        audio_features = self.audio_encoder(audio)[0] # audio_features shape [n_audio_files, 1024]
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
 

        prompts = [f"{class_name}" for class_name in self.args.classnames]
        tokenized_prompts = self.process_text(prompts, enc_tok=True, add_text=False)
        prompts_tokens = tokenized_prompts['input_ids'].to(self.device) 
        prompts_attention_mask = tokenized_prompts['attention_mask'].to(self.device)

        with torch.no_grad():
            prompts_token_embeddings = self.text_encoder.base.embeddings.token_embedding(prompts_tokens)   # [batch_size, seq_length, embed_dim]
        
        text = {"input_ids": prompts_tokens, "inputs_embeds": prompts_token_embeddings, "attention_mask": prompts_attention_mask}
        text_features = self.text_encoder(text) # text_features shape [n_text_prompts, 1024]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        audio_features, text_features = self.prompt_learner(audio_features, text_features) # text_features shape [n_text_prompts, 1024]
        
        logit_scale = 100.0
        logits = logit_scale * audio_features @ text_features.t()  # logits shape [n_audio_files, n_text_prompts]
        # breakpoint()

        return logits


