# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import random
import re
import unicodedata
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

from nemo.collections.common.tokenizers.text_to_speech.tokenizer_utils import normalize_unicode_text
from nemo.collections.tts.g2p.models.base import BaseG2p
from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p
from nemo.collections.tts.g2p.utils import GRAPHEME_CASE_UPPER, set_grapheme_case
from nemo.utils import logging


_ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN_WORD_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[\-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*$")

_SUN_LETTERS = {
    "ت",
    "ث",
    "د",
    "ذ",
    "ر",
    "ز",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ل",
    "ن",
}

_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")


def _has_arabic_diacritics(text: str) -> bool:
    return re.search(r"[\u064B-\u0652\u0670]", text) is not None


def _normalize_punctuation(text: str) -> str:
    """Normalize special Unicode punctuation to ASCII equivalents.

    This prevents unknown-character warnings for common typographic variants
    that may appear in real-world text (copy-paste from word processors, etc.).
    """
    # Dashes / hyphens
    text = text.replace("\u2014", "-")  # em dash —
    text = text.replace("\u2013", "-")  # en dash –
    text = text.replace("\u2011", "-")  # non-breaking hyphen ‑
    text = text.replace("\u2010", "-")  # hyphen ‐
    text = text.replace("\u2212", "-")  # minus sign −
    text = text.replace("\u2015", "-")  # horizontal bar ―
    # Spaces
    text = text.replace("\u00A0", " ")  # NBSP
    text = text.replace("\u202F", " ")  # narrow NBSP
    text = text.replace("\u2009", " ")  # thin space
    # Quotes (normalize to ASCII)
    text = text.replace("\u2018", "'")  # left single quote '
    text = text.replace("\u2019", "'")  # right single quote '
    text = text.replace("\u201C", '"')  # left double quote "
    text = text.replace("\u201D", '"')  # right double quote "
    # Ellipsis
    text = text.replace("\u2026", "...")  # …
    # Turkish/special Latin letters -> IPA-friendly equivalents
    text = text.replace("Ş", "ʃ")  # Turkish Ş -> IPA sh
    text = text.replace("ş", "ʃ")  # Turkish ş -> IPA sh
    text = text.replace("İ", "I")  # Turkish İ -> I
    text = text.replace("ı", "i")  # Turkish ı -> i
    text = text.replace("Ğ", "")   # Turkish Ğ -> silent/drop
    text = text.replace("ğ", "")   # Turkish ğ -> silent/drop
    # Remove stray Cyrillic letters
    text = re.sub(r"[\u0400-\u04FF]", "", text)
    return text


def _mixed_arabic_word_tokenize(text: str):
    """Tokenize mixed Arabic + Latin text.

    Returns: List[Tuple[List[str], bool]] compatible with BaseG2p.

    Groups:
      1) Latin words (incl. accents)
      2) Arabic word chunks (Arabic letters + combining marks)
      3) |unchanged sequences|
      4) punctuation/whitespace/other
    """
    # Normalize special Unicode punctuation before tokenization
    text = _normalize_punctuation(text)

    latin_chars = r"A-Za-zÀ-ÖØ-öø-ÿ"
    arabic_letters = r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
    arabic_marks = r"\u064B-\u0652\u0670"  # harakat + superscript alif

    words_re = re.compile(
        fr"([{latin_chars}]+(?:[{latin_chars}\-']*[{latin_chars}]+)*)"
        fr"|([{arabic_letters}]+(?:[{arabic_letters}{arabic_marks}]*[{arabic_letters}]+)*)"
        fr"|(\|[^|]*\|)"
        fr"|(\s+)"
        fr"|([^{latin_chars}{arabic_letters}|\s]+)"
    )

    result = []
    for match in words_re.findall(text):
        latin_word, arabic_word, maybe_without_changes, whitespace, other = match

        without_changes = False
        if latin_word:
            token = [latin_word]
        elif arabic_word:
            token = [arabic_word]
        elif maybe_without_changes:
            without_changes = True
            token = maybe_without_changes[1:-1].split(" ")
        elif whitespace:
            token = [whitespace]
        elif other:
            token = [other]
        else:
            raise ValueError(f"Unexpected empty regex match: {match}")

        result.append((token, without_changes))

    return result


class EmiratiG2P(IpaG2p):
    """Emirati Arabic (ar-AE) G2P that outputs IPA-like tokens.

    This G2P is designed for Arabic script input and mixed Arabic/English text.

    Emirati-specific rules implemented:
      - ق -> ɡ (default)
      - Diphthongs aw/ay (and Arabic-script او/اي) -> oː/eː (default)
      - Optional ك -> tʃ under conservative triggers (default off)
      - Optional ج -> j (default off; dialect dependent)

    Custom pronunciation dictionary:
      - Provide `custom_dictionary` (alias) or `phoneme_dict`.
      - File format: `WORD␠␠PRONUNCIATION` (two spaces recommended; any whitespace works).
      - Example: `قهوة  gahwa` (will be tokenized as ["ɡ", "a", "h", "w", "a"]).
        If you want explicit IPA symbols, you can write `قهوة  ɡahwa`.

    Notes:
      - NeMo's generic IPA locale validation doesn't include ar-AE; this class uses `locale=None`.
      - Dictionary entries override rule-based output.
    """

    # Use true IPA U+0261 for /g/.
    IPA_G = "ɡ"

    _RULE_SYMBOLS = {
        # Core Emirati deltas
        IPA_G,
        "o",
        "e",
        "ː",
        # Affricates (char-level)
        "t",
        "ʃ",
        "d",
        "ʒ",
        # Common Arabic consonants/diacritics used in the built-in map
        "ʔ",
        "θ",
        "ð",
        "ˤ",
        "ħ",
        "x",
        "ʕ",
        "ɣ",
        "b",
        "f",
        "h",
        "j",
        "k",
        "l",
        "m",
        "n",
        "p",
        "r",
        "s",
        "t",
        "v",
        "w",
        "z",
        "a",
        "i",
        "u",
        "ɪ",
    }

    def __init__(
        self,
        phoneme_dict: Optional[Union[str, pathlib.Path, Dict[str, List[List[str]]]]] = None,
        *,
        custom_dictionary: Optional[Union[str, pathlib.Path]] = None,
        enable_sun_letter_assimilation: bool = False,
        enable_vowel_insertion: bool = False,
        vowel_insertion_vowel: str = "ɪ",
        unknown_arabic_to_glottal_stop: bool = True,
        enable_en_g2p: bool = False,
        en_locale: str = "en-US",
        en_phoneme_dict: Optional[Union[str, pathlib.Path, Dict[str, List[List[str]]]]] = None,
        en_use_stresses: Optional[bool] = None,
        en_oov_fallback: str = "chars",
        latin_oov_as_chars: bool = False,
        latin_fallback_to_emirati: bool = False,
        arabic_diacritizer: Optional[Callable[[str], str]] = None,
        enable_arabic_diacritizer: bool = False,
        apply_to_oov_word: Optional[Callable[[str], List[str]]] = None,
        ignore_ambiguous_words: bool = True,
        use_chars: bool = False,
        phoneme_probability: Optional[float] = None,
        use_stresses: Optional[bool] = True,
        grapheme_case: Optional[str] = GRAPHEME_CASE_UPPER,
        grapheme_prefix: Optional[str] = "",
        enable_k_to_tsh: bool = False,
        enable_jeem_to_y: bool = False,
        enable_jeem_to_zh: bool = False,
        mapping_file: Optional[str] = None,
    ) -> None:
        self.enable_k_to_tsh = enable_k_to_tsh
        self.enable_jeem_to_y = enable_jeem_to_y
        self.enable_jeem_to_zh = enable_jeem_to_zh

        self.enable_sun_letter_assimilation = enable_sun_letter_assimilation
        self.enable_vowel_insertion = enable_vowel_insertion
        self.vowel_insertion_vowel = normalize_unicode_text(vowel_insertion_vowel)
        self.unknown_arabic_to_glottal_stop = unknown_arabic_to_glottal_stop

        self.enable_en_g2p = enable_en_g2p
        self.en_locale = en_locale
        self.en_use_stresses = en_use_stresses
        self.en_oov_fallback = en_oov_fallback
        self.latin_oov_as_chars = latin_oov_as_chars
        self.latin_fallback_to_emirati = latin_fallback_to_emirati
        self.en_g2p: Optional[IpaG2p] = None

        self.arabic_diacritizer = arabic_diacritizer
        self.enable_arabic_diacritizer = enable_arabic_diacritizer

        # Mirror IpaG2p init settings needed by its helpers.
        self.use_stresses = use_stresses
        self.grapheme_case = grapheme_case
        self.grapheme_prefix = grapheme_prefix
        self.phoneme_probability = phoneme_probability
        self.locale = None
        self._rng = random.Random()
        self.heteronyms = None

        if not use_chars and self.phoneme_probability is not None:
            self.use_chars = True
            logging.warning(
                "phoneme_probability was not None, characters will be enabled even though use_chars was set to False."
            )
        else:
            self.use_chars = use_chars

        if phoneme_dict is None and custom_dictionary is None:
            raise ValueError("Provide `phoneme_dict` or `custom_dictionary` for EmiratiG2P.")

        if phoneme_dict is None and custom_dictionary is not None:
            phoneme_dict = custom_dictionary

        phoneme_dict_obj = self._parse_emirati_phoneme_dict(phoneme_dict)

        if not phoneme_dict_obj:
            raise ValueError(f"{phoneme_dict} contains no entries!")

        _phoneme_dict, self.symbols = self._normalize_dict(phoneme_dict_obj)

        # Ensure tokens emitted by Emirati rules are included in the vocab.
        self.symbols.update(self._RULE_SYMBOLS)

        # Ensure configured vowel-insertion token exists in the vocab.
        if self.enable_vowel_insertion:
            self.symbols.add(self.vowel_insertion_vowel)

        # Optional: bilingual code-switching for Latin words via a dedicated English IPA G2P.
        if self.enable_en_g2p:
            # Prefer a real English IPA lexicon when available; otherwise fall back to whatever the caller provided.
            en_source = en_phoneme_dict
            if en_source is None:
                try:
                    nemo_root = pathlib.Path(__file__).resolve().parents[5]
                    default_en = nemo_root / "scripts" / "tts_dataset_files" / "ipa_cmudict-0.7b_nv23.01.txt"
                    if default_en.exists():
                        en_source = default_en
                except Exception:
                    en_source = None

            if en_source is None:
                en_source = phoneme_dict
            try:
                # If the Emirati lexicon was provided as multiple sources (base + overrides),
                # build a merged dict-object for the English IpaG2p.
                if isinstance(en_source, (list, tuple)):
                    en_source = EmiratiG2P._parse_emirati_phoneme_dict(en_source)

                self.en_g2p = IpaG2p(
                    phoneme_dict=en_source,
                    locale=self.en_locale,
                    apply_to_oov_word=None,
                    # For code-switching, it's better to pick a deterministic pronunciation than to fall back to
                    # grapheme tokens for common (often ambiguous) function words like "the"/"and".
                    ignore_ambiguous_words=False,
                    heteronyms=None,
                    use_chars=False,
                    phoneme_probability=None,
                    use_stresses=(use_stresses if self.en_use_stresses is None else self.en_use_stresses),
                    grapheme_case=grapheme_case,
                    grapheme_prefix=grapheme_prefix,
                )
                self.symbols.update(self.en_g2p.symbols)
            except Exception as e:
                logging.warning(
                    f"enable_en_g2p=True but failed to initialize English IpaG2p (locale={self.en_locale}). "
                    f"Falling back to EmiratiG2P latin handling. Error: {e}"
                )
                self.en_g2p = None

        # Optional: keep Latin OOV words as characters (useful for rare names).
        # If enabled, we emit grapheme tokens prefixed by `grapheme_prefix` (recommended) to avoid collisions
        # with IPA phonemes like "a".
        if self.latin_oov_as_chars or (self.enable_en_g2p and self.en_oov_fallback == "chars"):
            prefix = self.grapheme_prefix or ""

            base_chars = set("abcdefghijklmnopqrstuvwxyz")
            base_chars.update(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            base_chars.update({"-", "'"})

            # Include Latin-1 accented letters to reduce OOV drops for common names.
            # (This matches the tokenizer regex allowance used in _LATIN_WORD_RE.)
            for cp in range(0x00C0, 0x0100):
                ch = chr(cp)
                if ch.isalpha():
                    base_chars.add(ch)

            if prefix:
                self.symbols.update({f"{prefix}{c}" for c in base_chars})
            else:
                self.symbols.update(base_chars)

        BaseG2p.__init__(
            self,
            phoneme_dict=_phoneme_dict,
            word_tokenize_func=_mixed_arabic_word_tokenize,
            apply_to_oov_word=apply_to_oov_word,
            mapping_file=mapping_file,
        )

        self.ignore_ambiguous_words = ignore_ambiguous_words

    def _en_parse_one_word(self, word: str) -> Optional[List[str]]:
        if self.en_g2p is None:
            return None
        pron, handled = self.en_g2p.parse_one_word(word)
        if handled and pron:
            return pron
        return None

    def _try_parse_en_variants(self, word: str) -> Optional[List[str]]:
        # 1) Direct lookup.
        pron = self._en_parse_one_word(word)
        if pron is not None:
            return pron

        # 2) Hyphenated compounds: try split into parts.
        if "-" in word:
            parts = [p for p in word.split("-") if p]
            if parts:
                all_prons: List[str] = []
                for p in parts:
                    p_pron = self._en_parse_one_word(p)
                    if p_pron is None:
                        all_prons = []
                        break
                    all_prons.extend(p_pron)
                if all_prons:
                    return all_prons

        # 3) Common contractions: expand a small safe subset.
        if "'" in word:
            contractions = {
                "'RE": "ARE",
                "'M": "AM",
                "'LL": "WILL",
                "'VE": "HAVE",
            }
            for suffix, expansion in contractions.items():
                if word.endswith(suffix) and len(word) > len(suffix):
                    base = word[: -len(suffix)]
                    base_pron = self._en_parse_one_word(base)
                    exp_pron = self._en_parse_one_word(expansion)
                    if base_pron is not None and exp_pron is not None:
                        return base_pron + exp_pron

            # 4) Fallback: try removing apostrophes (handles some tokenization quirks).
            no_apos = word.replace("'", "")
            if no_apos and no_apos != word:
                pron = self._en_parse_one_word(no_apos)
                if pron is not None:
                    return pron

        return None

    @staticmethod
    def _parse_emirati_phoneme_dict(
        phoneme_dict: Union[
            str,
            pathlib.Path,
            Dict[str, List[List[str]]],
            List[Union[str, pathlib.Path, Dict[str, List[List[str]]]]],
            Tuple[Union[str, pathlib.Path, Dict[str, List[List[str]]]], ...],
        ]
    ) -> Dict[str, List[List[str]]]:
        """Parse a custom dict file that can start with Arabic script.

        Supports:
          - Path-like: lines split by first whitespace into word + pron.
          - Dict object: `{"word": [["ɡ", "a"], ...]}`
        """

        # Merge mode: accept multiple lexicon sources (base + overrides).
        # Later sources override earlier ones for the same word.
        if isinstance(phoneme_dict, (list, tuple)):
            merged: Dict[str, List[List[str]]] = {}
            for source in phoneme_dict:
                if source is None:
                    continue
                parsed = EmiratiG2P._parse_emirati_phoneme_dict(source)
                for word, prons in parsed.items():
                    merged[word] = prons
            return merged

        if isinstance(phoneme_dict, str) or isinstance(phoneme_dict, pathlib.Path):
            phoneme_dict_obj: Dict[str, List[List[str]]] = defaultdict(list)
            alt_re = re.compile(r"\([0-9]+\)$")

            with open(phoneme_dict, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = normalize_unicode_text(raw_line).strip("\n")
                    if not line.strip():
                        continue
                    if line.lstrip().startswith("#"):
                        continue

                    parts = line.strip().split(maxsplit=1)
                    if len(parts) != 2:
                        continue

                    word_raw, pron_raw = parts
                    word = re.sub(alt_re, "", word_raw)

                    tokens = EmiratiG2P._tokenize_dictionary_pron(pron_raw)
                    phoneme_dict_obj[word].append(tokens)

            return dict(phoneme_dict_obj)

        # Dict-object mode (recommended when you want multi-char tokens like "tʃ" as one token).
        phoneme_dict_obj: Dict[str, List[List[str]]] = {}
        for word, prons in phoneme_dict.items():
            assert isinstance(prons, list), f"Pronunciation type <{type(prons)}> is not supported. Please convert to <list>."
            word = normalize_unicode_text(word)
            prons = [[normalize_unicode_text(p) for p in pron] for pron in prons]
            phoneme_dict_obj[word] = prons

        return phoneme_dict_obj

    @staticmethod
    def _tokenize_dictionary_pron(pron: str) -> List[str]:
        pron = normalize_unicode_text(pron).strip()

        # If the user provided whitespace-delimited tokens, honor them.
        if re.search(r"\s", pron):
            tokens = []
            for t in pron.split():
                tokens.extend(EmiratiG2P._expand_compact_pron_token(t))
            return tokens

        # Otherwise, treat it as a compact string and do a tiny amount of greedy tokenization.
        return EmiratiG2P._expand_compact_pron_token(pron)

    @staticmethod
    def _expand_compact_pron_token(token: str) -> List[str]:
        """Expand compact tokens into NeMo-style IPA token sequences.

        Examples:
          - "gahwa" -> ["ɡ", "a", "h", "w", "a"]
          - "oː" -> ["o", "ː"]
          - "tʃ" -> ["t", "ʃ"]
        """

        t = normalize_unicode_text(token)

        out: List[str] = []
        i = 0
        while i < len(t):
            if t.startswith("tʃ", i):
                out.extend(["t", "ʃ"])
                i += 2
                continue
            if t.startswith("dʒ", i):
                out.extend(["d", "ʒ"])
                i += 2
                continue
            if t.startswith("oː", i):
                out.extend(["o", "ː"])
                i += 2
                continue
            if t.startswith("eː", i):
                out.extend(["e", "ː"])
                i += 2
                continue
            if t.startswith("aː", i):
                out.extend(["a", "ː"])
                i += 2
                continue
            if t.startswith("iː", i):
                out.extend(["i", "ː"])
                i += 2
                continue
            if t.startswith("uː", i):
                out.extend(["u", "ː"])
                i += 2
                continue

            c = t[i]
            if c == "g":
                out.append(EmiratiG2P.IPA_G)
            else:
                out.append(c)
            i += 1

        return out

    def parse_one_word(self, word: str) -> Tuple[List[str], bool]:
        word = set_grapheme_case(word, case=self.grapheme_case)

        # Handle Arabic-script word chunks (or mixed chunks containing Arabic letters)
        if _ARABIC_LETTER_RE.search(word) is not None:
            # Dict overrides
            if word in self.phoneme_dict and (
                not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word)
            ):
                return self.phoneme_dict[word][0], True

            tokens = self._arabic_word_to_ipa_tokens(word)
            return tokens, True

        # Latin-only tokens: support simple Emirati-friendly transliteration behavior.
        # Dictionary still takes precedence via the superclass.
        if _LATIN_WORD_RE.match(word):
            if word in self.phoneme_dict and (
                not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word)
            ):
                return self.phoneme_dict[word][0], True

            if self.enable_en_g2p:
                pron = self._try_parse_en_variants(word)
                if pron is not None:
                    return pron, True

                # PRIORITY FALLBACK ORDER (DO NOT emit grapheme tokens):
                # 1. Try Emirati transliteration for better phonemic output
                # 2. Only drop if explicitly requested, never fall back to raw graphemes
                if self.latin_fallback_to_emirati:
                    return self._latin_word_to_emirati_tokens(word), True

                if self.en_oov_fallback == "emirati":
                    return self._latin_word_to_emirati_tokens(word), True
                if self.en_oov_fallback == "drop":
                    return [], False

                # Default fallback: use Emirati transliteration instead of grapheme tokens
                # This prevents G_R, G_E, etc. from appearing in phoneme sequences
                return self._latin_word_to_emirati_tokens(word), True

            # English routing disabled: keep the historical Emirati-biased latin handling.
            return self._latin_word_to_emirati_tokens(word), True

        # Other tokens (punctuation, digits, etc): use the generic IPA logic.
        return super().parse_one_word(word)

    def _latin_word_to_emirati_tokens(self, word: str) -> List[str]:
        """Minimal Latin transliteration helper.

        Implements only the requested Emirati-specific vowel shift:
          - aw -> oː
          - ay -> eː
        Also maps 'g' -> IPA ɡ.

        Output tokens follow NeMo's IPA token conventions (single chars + combining marks),
        so oː is emitted as ["o", "ː"].
        """

        w = normalize_unicode_text(word).lower()
        out: List[str] = []
        i = 0
        while i < len(w):
            if w.startswith("aw", i):
                out.extend(["o", "ː"])
                i += 2
                continue
            if w.startswith("ay", i):
                out.extend(["e", "ː"])
                i += 2
                continue

            c = w[i]
            if c == "g":
                out.append(self.IPA_G)
            else:
                out.append(c)
            i += 1

        return out

    def _arabic_word_to_ipa_tokens(self, word: str) -> List[str]:
        """Very lightweight Arabic-script -> IPA-ish tokenizer.

        This is intentionally conservative: it implements Emirati-specific deltas and a basic consonant map.
        For high accuracy, rely on the custom dictionary for irregular/OOV words.
        """

        original_word = normalize_unicode_text(word)

        if (
            self.enable_arabic_diacritizer
            and self.arabic_diacritizer is not None
            and _ARABIC_LETTER_RE.search(original_word) is not None
            and not _has_arabic_diacritics(original_word)
        ):
            try:
                original_word = normalize_unicode_text(self.arabic_diacritizer(original_word))
            except Exception as e:
                logging.warning(f"Arabic diacritizer failed on '{word}'. Falling back to heuristic Arabic G2P. Error: {e}")

        # Emirati suffix-only kaf palatalization trigger.
        # Only apply /tʃ/ when the 2nd-person feminine suffix is explicitly marked as كِ.
        # This avoids overgeneration on undiacritized text.
        has_feminine_suffix_kasra = original_word.endswith("ك\u0650")

        # Keep diacritics if present so we can use them (esp. shadda + short vowels).
        word_with_marks = "".join(
            ch
            for ch in original_word
            if unicodedata.category(ch).startswith(("L", "M")) and ch != "\u0640"  # tatweel
        )
        if not word_with_marks:
            return []

        # Stripped version for fallback heuristics / digraph checks.
        word = re.sub(_ARABIC_DIACRITICS_RE, "", word_with_marks)

        tokens: List[str] = []
        i = 0

        # Basic consonant map (subset + common Arabic IPA)
        consonant_map: Dict[str, List[str]] = {
            "ء": ["ʔ"],
            "أ": ["ʔ"],
            "إ": ["ʔ"],
            "ؤ": ["ʔ"],
            "ئ": ["ʔ"],
            "ا": ["a", "ː"],
            "ٱ": ["a"],
            "آ": ["ʔ", "a", "ː"],
            "ب": ["b"],
            "ت": ["t"],
            "ث": ["θ"],
            "ج": (["j"] if self.enable_jeem_to_y else (["ʒ"] if self.enable_jeem_to_zh else ["d", "ʒ"])),
            "ح": ["ħ"],
            "خ": ["x"],
            "د": ["d"],
            "ذ": ["ð"],
            "ر": ["r"],
            "ز": ["z"],
            "س": ["s"],
            "ش": ["ʃ"],
            "ص": ["s", "ˤ"],
            "ض": ["d", "ˤ"],
            "ط": ["t", "ˤ"],
            "ظ": ["ð", "ˤ"],
            "ع": ["ʕ"],
            "غ": ["ɣ"],
            "ف": ["f"],
            "ق": [self.IPA_G],
            "ك": ["k"],  # may rewrite to tʃ below
            "ل": ["l"],
            "م": ["m"],
            "ن": ["n"],
            "ه": ["h"],
            "و": ["w"],
            "ي": ["j"],
            # ة handled contextually below
            "ى": ["a", "ː"],
            # Loan letters
            "پ": ["p"],
            "ڤ": ["v"],
            "چ": ["t", "ʃ"],
            # Arabic-adjacent variants of /g/
            "گ": [self.IPA_G],
            "ڬ": [self.IPA_G],
            "ݣ": [self.IPA_G],
        }

        # Optional: sun-letter assimilation for the definite article (الـ).
        # Very conservative: only applies at the beginning of a word.
        # Example: الشمس -> a ʃ ʃ m s (instead of a l ʃ m s)
        if self.enable_sun_letter_assimilation and word.startswith("ال") and len(word) >= 3:
            sun = word[2]
            if sun in _SUN_LETTERS:
                tokens.append("a")
                sun_mapped = consonant_map.get(sun)
                if sun_mapped is None:
                    # If we cannot map the sun letter, fall back to a plain "al" prefix.
                    tokens.extend(["l"])
                else:
                    # Geminate the sun letter by duplicating its token sequence.
                    tokens.extend(sun_mapped)
                    tokens.extend(sun_mapped)
                    # Consume "ال" and the sun letter.
                    i = 3

        def _apply_unknown_char_policy(ch: str) -> List[str]:
            if self.unknown_arabic_to_glottal_stop:
                return ["ʔ"]
            logging.warning(f"Unknown Arabic letter '{ch}' (U+{ord(ch):04X}) in word '{original_word}'. Dropping.")
            return []

        def _apply_default_vowel_insertion(seq: List[str]) -> List[str]:
            # Insert a default short vowel conservatively to reduce illegal consonant pileups
            # on unvocalized Arabic.
            #
            # Policy (minimal but higher quality):
            #   - Repair runs of 2+ consonant *units* by inserting a short vowel.
            #   - Avoid inserting inside gemination (identical consecutive consonant units).
            #   - Limit insertions to keep the output predictable.
            vowel_tokens = {"a", "i", "u", "o", "e", "ə", "ɪ", "ʊ", "ɑ", "ɛ", "ɔ"}
            vowel_tokens.add(self.vowel_insertion_vowel)

            consonant_tokens = {
                "ʔ",
                "b",
                "t",
                "θ",
                "d",
                "ʒ",
                "j",
                "ħ",
                "x",
                "ð",
                "r",
                "z",
                "s",
                "ʃ",
                "ˤ",  # modifier (handled separately)
                "ʕ",
                "ɣ",
                "f",
                self.IPA_G,
                "k",
                "l",
                "m",
                "n",
                "h",
                "w",
                "p",
                "v",
            }
            consonant_modifiers = {"ˤ"}
            length_mark = "ː"

            def split_units(tokens_in: List[str]):
                units = []
                idx = 0
                while idx < len(tokens_in):
                    tok = tokens_in[idx]
                    # Treat affricates as single consonant units so we never insert between their parts.
                    if idx + 1 < len(tokens_in) and tok == "d" and tokens_in[idx + 1] == "ʒ":
                        units.append(("cons", ["d", "ʒ"]))
                        idx += 2
                        continue
                    if idx + 1 < len(tokens_in) and tok == "t" and tokens_in[idx + 1] == "ʃ":
                        units.append(("cons", ["t", "ʃ"]))
                        idx += 2
                        continue
                    if tok in vowel_tokens:
                        unit = [tok]
                        if idx + 1 < len(tokens_in) and tokens_in[idx + 1] == length_mark:
                            unit.append(length_mark)
                            idx += 1
                        units.append(("vowel", unit))
                    elif tok in consonant_tokens and tok not in consonant_modifiers:
                        unit = [tok]
                        j = idx + 1
                        while j < len(tokens_in) and tokens_in[j] in consonant_modifiers:
                            unit.append(tokens_in[j])
                            j += 1
                        units.append(("cons", unit))
                        idx = j - 1
                    else:
                        units.append(("other", [tok]))
                    idx += 1
                return units

            units = split_units(seq)

            def is_cons(i: int) -> bool:
                return 0 <= i < len(units) and units[i][0] == "cons"

            def is_geminate_pair(i: int) -> bool:
                # True when units[i] and units[i+1] are identical consonant units.
                if not (is_cons(i) and is_cons(i + 1)):
                    return False
                return units[i][1] == units[i + 1][1]

            def insert_after_unit(units_in: List[Tuple[str, List[str]]], idx: int) -> List[Tuple[str, List[str]]]:
                return units_in[: idx + 1] + [("vowel", [self.vowel_insertion_vowel])] + units_in[idx + 1 :]

            # Fast path: no consonant pileups.
            cons_count = sum(1 for t, _ in units if t == "cons")
            if cons_count < 2:
                return seq

            max_insertions = 2
            insertions = 0

            # 1) Two-letter Arabic words often need a vowel (e.g., "كم" -> k a m).
            if cons_count == 2 and not any(t == "vowel" for t, _ in units):
                for i_u in range(len(units) - 1):
                    if units[i_u][0] == "cons" and units[i_u + 1][0] == "cons" and not is_geminate_pair(i_u):
                        units = insert_after_unit(units, i_u)
                        insertions += 1
                        break

            # 2) If we have 4+ consonants, break a truly word-final CC cluster when present
            # (common in forms like "عندك" or "قمر").
            cons_count = sum(1 for t, _ in units if t == "cons")
            if insertions < max_insertions and cons_count >= 4:
                non_other_idx = [i for i, (t, _) in enumerate(units) if t != "other"]
                if len(non_other_idx) >= 2:
                    i_prev = non_other_idx[-2]
                    i_last = non_other_idx[-1]
                    if (
                        units[i_prev][0] == "cons"
                        and units[i_last][0] == "cons"
                        and i_last == i_prev + 1
                        and not is_geminate_pair(i_prev)
                    ):
                        units = insert_after_unit(units, i_prev)
                        insertions += 1

            # 3) General repair: ensure there are no 3 consecutive consonant units.
            # Insert position depends on whether the run spans the entire word.
            changed = True
            while changed and insertions < max_insertions:
                changed = False
                for i_u in range(len(units) - 2):
                    if units[i_u][0] != "cons" or units[i_u + 1][0] != "cons" or units[i_u + 2][0] != "cons":
                        continue
                    # If any adjacent pair is gemination, don't insert.
                    if is_geminate_pair(i_u) or is_geminate_pair(i_u + 1):
                        continue

                    # Determine if the triple is effectively the whole word (ignoring "other").
                    non_other = [(t, u) for (t, u) in units if t != "other"]
                    all_cons = non_other and all(t == "cons" for t, _ in non_other)
                    total_cons = sum(1 for t, _ in non_other if t == "cons")

                    # If it's a short all-consonant word (e.g., "شفت"), prefer CVC(C): insert after 1st.
                    # Otherwise, prefer keeping up to a 2-consonant onset: insert between 2nd and 3rd.
                    if all_cons and total_cons == 3:
                        insert_at = i_u
                    elif i_u == 0 and units[0][1] in (["ʔ"], ["ʕ"]):
                        # For common Arabic forms starting with ʔ/ʕ, a CV onset sounds more natural.
                        insert_at = i_u
                    else:
                        insert_at = i_u + 1

                    units = insert_after_unit(units, insert_at)
                    insertions += 1
                    changed = True
                    break

            # Flatten back.
            out: List[str] = []
            for _, u_tokens in units:
                out.extend(u_tokens)
            return out

        def _tokenize_with_diacritics(text_with_marks: str) -> List[str]:
            """Diacritic-aware tokenization.

            Uses shadda (0651) for gemination and harakat for short vowels when present.
            Also applies lengthening for classic long vowels:
              - َ + ا -> aː
              - ُ + و -> uː
              - ِ + ي -> iː
            """

            SHADDA = "\u0651"
            SUKUN = "\u0652"
            FATHA = "\u064E"
            DAMMA = "\u064F"
            KASRA = "\u0650"
            FATHATAN = "\u064B"
            DAMMATAN = "\u064C"
            KASRATAN = "\u064D"

            diacritic_to_vowel = {
                FATHA: "a",
                DAMMA: "u",
                KASRA: "i",
            }
            tanwin_to_vowel = {
                FATHATAN: "a",
                DAMMATAN: "u",
                KASRATAN: "i",
            }

            # Group letters with their combining marks.
            units: List[Tuple[str, List[str]]] = []
            current_letter: Optional[str] = None
            current_marks: List[str] = []

            for ch in text_with_marks:
                cat = unicodedata.category(ch)
                if cat.startswith("L"):
                    if current_letter is not None:
                        units.append((current_letter, current_marks))
                    current_letter = ch
                    current_marks = []
                elif cat.startswith("M"):
                    if current_letter is None:
                        continue
                    current_marks.append(ch)
                else:
                    continue
            if current_letter is not None:
                units.append((current_letter, current_marks))

            out: List[str] = []
            last_short_vowel_pos: Optional[int] = None
            last_short_vowel: Optional[str] = None

            def append_short_vowel(v: str):
                nonlocal last_short_vowel_pos, last_short_vowel
                out.append(v)
                last_short_vowel_pos = len(out) - 1
                last_short_vowel = v

            def try_lengthen_with(letter: str) -> bool:
                nonlocal last_short_vowel_pos, last_short_vowel
                if last_short_vowel_pos is None or last_short_vowel is None:
                    return False
                if letter == "ا" and last_short_vowel == "a":
                    out.insert(last_short_vowel_pos + 1, "ː")
                    last_short_vowel_pos = None
                    last_short_vowel = None
                    return True
                if letter == "و" and last_short_vowel == "u":
                    out.insert(last_short_vowel_pos + 1, "ː")
                    last_short_vowel_pos = None
                    last_short_vowel = None
                    return True
                if letter == "ي" and last_short_vowel == "i":
                    out.insert(last_short_vowel_pos + 1, "ː")
                    last_short_vowel_pos = None
                    last_short_vowel = None
                    return True
                return False

            for idx_u, (letter, marks) in enumerate(units):
                # Lengtheners: if previous letter had a short vowel, convert to long vowel and skip mapping.
                if letter in {"ا", "و", "ي"} and try_lengthen_with(letter):
                    continue

                # Contextual ta marbuta.
                if letter == "ة":
                    is_final = idx_u == len(units) - 1
                    base_tokens = ["a"] if is_final else ["t"]
                else:
                    base_tokens = consonant_map.get(letter)
                    if base_tokens is None:
                        base_tokens = _apply_unknown_char_policy(letter)

                # Emirati kaf to /tʃ/ (suffix-only, high precision)
                if (
                    letter == "ك"
                    and self.enable_k_to_tsh
                    and has_feminine_suffix_kasra
                    and idx_u == len(units) - 1
                ):
                    base_tokens = ["t", "ʃ"]

                # Apply shadda gemination by duplicating the consonant unit.
                if SHADDA in marks and base_tokens:
                    out.extend(base_tokens)
                    out.extend(base_tokens)
                else:
                    out.extend(base_tokens)

                # Apply short vowels / tanwin if present.
                if any(m in diacritic_to_vowel for m in marks):
                    # Prefer the last vowel mark if multiple are present.
                    vowel_mark = None
                    for m in marks:
                        if m in diacritic_to_vowel:
                            vowel_mark = m
                    if vowel_mark is not None:
                        append_short_vowel(diacritic_to_vowel[vowel_mark])

                if any(m in tanwin_to_vowel for m in marks):
                    # Tanwin implies final -n.
                    tanwin_mark = None
                    for m in marks:
                        if m in tanwin_to_vowel:
                            tanwin_mark = m
                    if tanwin_mark is not None:
                        append_short_vowel(tanwin_to_vowel[tanwin_mark])
                        out.append("n")

                # Sukun explicitly means no vowel; nothing to do.
                if SUKUN in marks:
                    last_short_vowel_pos = None
                    last_short_vowel = None

            return out

        # Heuristic: treat a token as "meaningfully vocalized" only if it has shadda (gemination cue)
        # or at least two vowel marks (harakat/tanwin). This avoids turning single-purpose marks
        # (like the kasra in كِ used only as a palatalization trigger) into spurious vowels.
        vowel_marks = {"\u064E", "\u064F", "\u0650", "\u064B", "\u064C", "\u064D"}
        has_shadda = "\u0651" in word_with_marks
        vowel_mark_count = sum(1 for ch in word_with_marks if ch in vowel_marks)

        if has_shadda or vowel_mark_count >= 2:
            tokens = _tokenize_with_diacritics(word_with_marks)

            if self.enable_vowel_insertion:
                tokens = _apply_default_vowel_insertion(tokens)

            return tokens if tokens else ["ʔ"]

        while i < len(word):
            # Emirati monophthongization: او -> oː, اي -> eː
            digraph = word[i : i + 2]
            if digraph == "او":
                tokens.extend(["o", "ː"])
                i += 2
                continue
            if digraph == "اي":
                tokens.extend(["e", "ː"])
                i += 2
                continue

            ch = word[i]

            # SAFER: Only treat word-final ي/و as long vowels, otherwise always emit j/w.
            # This avoids systematic errors in undiacritized text (see docstring).
            if ch == "ي":
                if i == 0:
                    tokens.extend(["j"])
                elif i == len(word) - 1:
                    tokens.extend(["i", "ː"])
                else:
                    tokens.extend(["j"])
                i += 1
                continue

            if ch == "و":
                if i == 0:
                    tokens.extend(["w"])
                elif i == len(word) - 1:
                    tokens.extend(["u", "ː"])
                else:
                    tokens.extend(["w"])
                i += 1
                continue

            # Contextual ta marbuta: final ة -> /a/ (pause); before suffixes -> /t/.
            if ch == "ة":
                if i == len(word) - 1:
                    tokens.extend(["a"])
                else:
                    tokens.extend(["t"])
                i += 1
                continue

            # Emirati kaf to /tʃ/ (suffix-only, high precision)
            if ch == "ك" and self.enable_k_to_tsh and has_feminine_suffix_kasra and i == len(word) - 1:
                tokens.extend(["t", "ʃ"])
                i += 1
                continue

            mapped = consonant_map.get(ch)
            if mapped is None:
                tokens.extend(_apply_unknown_char_policy(ch))
            else:
                tokens.extend(mapped)

            i += 1

        if self.enable_vowel_insertion:
            tokens = _apply_default_vowel_insertion(tokens)

        # Never return an empty token sequence for an Arabic word.
        return tokens if tokens else ["ʔ"]
