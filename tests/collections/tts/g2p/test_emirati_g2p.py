import pytest

from nemo.collections.tts.g2p.models.ar_ae_ipa import EmiratiG2P


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_qaf_to_g_rule():
    g2p = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]})
    assert g2p("ق") == ["ɡ"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_diphthong_monophthongization_arabic_script():
    g2p = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]})
    assert g2p("او اي") == ["o", "ː", " ", "e", "ː"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_diphthong_monophthongization_latin():
    g2p = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]})
    assert g2p("aw ay") == ["o", "ː", " ", "e", "ː"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_kaf_to_tsh_optional():
    g2p_off = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]}, enable_k_to_tsh=False)
    g2p_on = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]}, enable_k_to_tsh=True)

    # Suffix-only: final ك is palatalized only when kasra is explicitly present (كِ).
    assert g2p_off("كتابكِ") == ["k", "t", "a", "ː", "b", "k"]
    assert g2p_on("كتابكِ") == ["k", "t", "a", "ː", "b", "t", "ʃ"]
    # No diacritic => no palatalization.
    assert g2p_on("كتابك") == ["k", "t", "a", "ː", "b", "k"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_custom_dictionary_overrides_rules(tmp_path):
    d = tmp_path / "emirati_dict.txt"
    d.write_text("قهوة  gahwa\n", encoding="utf-8")

    g2p = EmiratiG2P(custom_dictionary=str(d))

    # Dictionary wins (note extra 'a' after the initial /g/)
    assert g2p("قهوة") == ["ɡ", "a", "h", "w", "a"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_jeem_toggle():
    g2p_default = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]}, enable_jeem_to_y=False)
    g2p_colloq = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]}, enable_jeem_to_y=True)

    assert g2p_default("ج") == ["d", "ʒ"]
    assert g2p_colloq("ج") == ["j"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_dictionary_tokenization_space_delimited_tokens(tmp_path):
    d = tmp_path / "emirati_dict_space_tokens.txt"
    d.write_text("قهوة  g a h w a\n", encoding="utf-8")
    g2p = EmiratiG2P(custom_dictionary=str(d))

    # Space-delimited tokens are honored; latin 'g' is normalized to IPA 'ɡ'.
    assert g2p("قهوة") == ["ɡ", "a", "h", "w", "a"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_explicit_chee_maps_to_tsh_even_when_kaf_rule_disabled():
    g2p = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]}, enable_k_to_tsh=False)
    assert g2p("چ") == ["t", "ʃ"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_mixed_arabic_latin_punctuation_roundtrip():
    g2p = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]})
    assert g2p("ق!aw?") == ["ɡ", "!", "o", "ː", "?"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_sun_letter_assimilation_optional():
    g2p_off = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]}, enable_sun_letter_assimilation=False)
    g2p_on = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]}, enable_sun_letter_assimilation=True)

    assert g2p_off("الشمس") == ["a", "ː", "l", "ʃ", "m", "s"]
    assert g2p_on("الشمس") == ["a", "ʃ", "ʃ", "m", "s"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_vowel_insertion_heuristic_optional():
    g2p_off = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]}, enable_vowel_insertion=False)
    g2p_on = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]}, enable_vowel_insertion=True, vowel_insertion_vowel="a")

    assert g2p_off("شفت") == ["ʃ", "f", "t"]
    # Cluster repair: insert to break consonant pileups.
    assert g2p_on("شفت") == ["ʃ", "a", "f", "t"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_vowel_insertion_repairs_four_consonant_run_like_indak():
    g2p = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]}, enable_vowel_insertion=True, vowel_insertion_vowel="a")
    # "عندك" is 4 consonants in our naive map; we should break clusters.
    assert g2p("عندك") == ["ʕ", "a", "n", "d", "a", "k"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_jeem_to_zh_optional():
    g2p_default = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]})
    g2p_zh = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]}, enable_jeem_to_zh=True)

    assert g2p_default("ج") == ["d", "ʒ"]
    assert g2p_zh("ج") == ["ʒ"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_true_code_switching_with_en_g2p_dict_object_mode():
    # Emirati dict minimal; English dict contains the Latin word.
    en_dict = {"TODAY": [["ˈ", "t", "u", "d", "e", "ɪ"]]}
    g2p = EmiratiG2P(
        phoneme_dict={"DUMMY": [["a"]]},
        enable_en_g2p=True,
        en_locale="en-US",
        en_phoneme_dict=en_dict,
    )

    assert g2p("today") == ["ˈ", "t", "u", "d", "e", "ɪ"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_latin_oov_as_prefixed_graphemes_no_emirati_bias():
    g2p = EmiratiG2P(
        phoneme_dict={"DUMMY": [["a"]]},
        enable_en_g2p=True,
        en_phoneme_dict={},
        latin_oov_as_chars=True,
        latin_fallback_to_emirati=False,
        grapheme_prefix="G_",
    )

    assert g2p("abc") == ["G_A", "G_B", "G_C"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_arabic_punctuation_is_stripped_from_arabic_chunks():
    g2p = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]})
    # Arabic comma U+060C should not appear in the phoneme stream.
    assert g2p("وقت،") == ["w", "ɡ", "t"]
    # Arabic question mark U+061F should not appear in the phoneme stream.
    assert g2p("وقت؟") == ["w", "ɡ", "t"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_en_g2p_handles_ambiguous_words_and_basic_contractions():
    # AND has multiple pronunciations; with English routing enabled, we should still choose deterministically.
    en_dict = {
        "AND": [["ə", "n", "d"], ["ˈ", "æ", "n", "d"]],
        "YOU": [["j", "u"]],
        "ARE": [["ɑ", "ɹ"]],
        "I": [["ˈ", "a", "ɪ"]],
        "AM": [["æ", "m"]],
    }
    g2p = EmiratiG2P(
        phoneme_dict={"DUMMY": [["a"]]},
        enable_en_g2p=True,
        en_locale="en-US",
        en_phoneme_dict=en_dict,
        latin_oov_as_chars=True,
        grapheme_prefix="G_",
    )

    assert g2p("and") == ["ə", "n", "d"]
    assert g2p("you're") == ["j", "u", "ɑ", "ɹ"]
    assert g2p("I'm") == ["ˈ", "a", "ɪ", "æ", "m"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_yaa_maps_to_j():
    g2p = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]})
    assert g2p("ي") == ["j"]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_emirati_unknown_arabic_letter_defaults_to_glottal_stop():
    g2p = EmiratiG2P(phoneme_dict={"DUMMY": [["a"]]})
    # ژ (U+0698) is in Arabic block but not mapped; should not leak as a token.
    assert g2p("ژ") == ["ʔ"]
