class BaseUnitizer(object):
    def unitize_lines(self, tessfile, tokenizer):
        raise NotImplementedError

    def unitize_phrases(self, tessfile, tokenizer):
        raise NotImplementedError
