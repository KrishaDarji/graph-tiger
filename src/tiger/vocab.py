from dataclasses import dataclass
import zlib

PAD_ID = 0
EOS_ID = 1
DECODER_START_ID = PAD_ID


@dataclass(frozen=True)
class TigerVocab:
    level_sizes: tuple = (256, 256, 256, 256)
    user_buckets: int = 2000

    @property
    def num_levels(self):
        return len(self.level_sizes)

    @property
    def codeword_offset(self):
        return 2

    def level_offset(self, level):
        return self.codeword_offset + sum(self.level_sizes[:level])

    def level_range(self, level):
        lo = self.level_offset(level)
        return lo, lo + self.level_sizes[level]

    @property
    def user_offset(self):
        return self.codeword_offset + sum(self.level_sizes)

    @property
    def vocab_size(self):
        return self.user_offset + self.user_buckets

    def user_token(self, user_id):
        bucket = zlib.crc32(str(user_id).encode("utf-8")) % self.user_buckets
        return self.user_offset + bucket

    def codes_to_tokens(self, codes):
        if len(codes) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} codes, got {len(codes)}")
        tokens = []
        for level, code in enumerate(codes):
            if not 0 <= code < self.level_sizes[level]:
                raise ValueError(f"code {code} out of range for level {level}")
            tokens.append(self.level_offset(level) + int(code))
        return tokens

    def tokens_to_codes(self, tokens):
        return tuple(int(t) - self.level_offset(level) for level, t in enumerate(tokens))
