from dataclasses import asdict, dataclass

from transformers import T5Config, T5ForConditionalGeneration

from .vocab import DECODER_START_ID, EOS_ID, PAD_ID, TigerVocab


@dataclass
class ModelConfig:
    d_model: int = 128
    d_kv: int = 64
    d_ff: int = 1024
    num_heads: int = 6
    num_layers: int = 4  # encoder layers
    num_decoder_layers: int = 4
    dropout: float = 0.1
    activation: str = "relu"
    tie_embeddings: bool = False  # T5 v1.1 / T5X default: separate output projection

    def to_dict(self):
        return asdict(self)


def build_model(vocab: TigerVocab, config: ModelConfig = None):
    config = config or ModelConfig()
    t5 = T5Config(
        vocab_size=vocab.vocab_size,
        d_model=config.d_model,
        d_kv=config.d_kv,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        dropout_rate=config.dropout,
        feed_forward_proj=config.activation,
        pad_token_id=PAD_ID,
        eos_token_id=EOS_ID,
        decoder_start_token_id=DECODER_START_ID,
        tie_word_embeddings=config.tie_embeddings,
    )
    return T5ForConditionalGeneration(t5)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
