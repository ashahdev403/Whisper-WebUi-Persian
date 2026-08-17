# -*- coding: utf-8 -*-
import sys
import unittest

import torch

sys.path.append('../whisper-webui-persian')

from src.whisper.transformersWhisperContainer import _degeneracy_reason, _truncate_prompt_ids

# Verbatim from a real Persian transcription run. The first two are healthy and must survive; the
# last two are the two ways a decoder loop actually collapsed in that same run.
HEALTHY = (
    "خب، تو مرحله اول، ببخشید، تو مرحله اول، بانک مرکزی یعنی بانکهای مرکزی، بزرگترین بازیگران این "
    "بازارن، چرا چون ارزهای کشورهای مختلف داره توی این بازار خرید و فروش میشه"
)
HEALTHY_WITH_REPEATED_WORDS = "نه نه نه، منظورم این نبود، ببخشید ببخشید"
COLLAPSED_CHARACTER = "بانکهای مرکزی میان با اعمال نرخهای بهره، و" + "ن" * 400
COLLAPSED_PHRASE = "اوپک در چند که " + "سپس روزپر پروژکت " * 30


class TestDegeneracyDetection(unittest.TestCase):
    def test_keeps_healthy_persian(self):
        self.assertIsNone(_degeneracy_reason(HEALTHY))

    def test_keeps_natural_word_repetition(self):
        # Speech genuinely repeats words; that is not a decoder loop
        self.assertIsNone(_degeneracy_reason(HEALTHY_WITH_REPEATED_WORDS))

    def test_keeps_english(self):
        self.assertIsNone(_degeneracy_reason("The quick brown fox jumps over the lazy dog."))

    def test_keeps_empty(self):
        self.assertIsNone(_degeneracy_reason("   "))

    def test_detects_repeated_character(self):
        self.assertIsNotNone(_degeneracy_reason(COLLAPSED_CHARACTER))

    def test_detects_repeated_phrase(self):
        # A three word phrase looping leaves each single word at only a third of the text, so this
        # is only caught by looking at consecutive repetition rather than word frequency
        self.assertIsNotNone(_degeneracy_reason(COLLAPSED_PHRASE))


class TestPromptTruncation(unittest.TestCase):
    def test_long_prompt_is_cut_to_half_the_context(self):
        prompt_ids = torch.arange(600)
        truncated = _truncate_prompt_ids(prompt_ids, 448)

        # OpenAI keeps n_ctx // 2 - 1 tokens
        self.assertEqual(truncated.shape[-1], 223)
        # The <|startofprev|> marker has to stay in front
        self.assertEqual(truncated[0].item(), 0)
        # ...and what is kept is the most recent context, not the oldest
        self.assertEqual(truncated[-1].item(), 599)

    def test_short_prompt_is_untouched(self):
        prompt_ids = torch.arange(50)
        self.assertTrue(torch.equal(_truncate_prompt_ids(prompt_ids, 448), prompt_ids))

    def test_prompt_plus_output_fits_the_decoder(self):
        # The crash was decoder_input_ids of 458 against max_target_positions of 448
        truncated = _truncate_prompt_ids(torch.arange(1000), 448)
        self.assertLess(truncated.shape[-1], 448)


if __name__ == '__main__':
    unittest.main()
