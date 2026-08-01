import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from androidllm.tokenizer import ByteLevelBPE, render_template, convert_hf_tokenizer

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_tokenizer")


def make_small_model_dir():
    os.makedirs(TMP, exist_ok=True)
    # tiny byte-level vocab: byte symbols + merge results + specials
    vocab = ["", "", "", "", "hello", "Ġworld", "Ġ", "world", "h", "e", "l",
             "o", "w", "r", "d", "Ġw", "Ġwo", "Ġwor", "Ġworl", "<|im_start|>",
             "<|im_end|>"]
    with open(os.path.join(TMP, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")
    merges = ["h e", "he l", "hel l", "hell o", "Ġ w", "Ġw o", "Ġwo r",
              "Ġwor l", "Ġworl d"]
    with open(os.path.join(TMP, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(merges) + "\n")
    with open(os.path.join(TMP, "special_tokens.json"), "w", encoding="utf-8") as f:
        f.write('{"<|im_start|>": 19, "<|im_end|>": 20}')
    with open(os.path.join(TMP, "template.txt"), "w", encoding="utf-8") as f:
        f.write("{% for message in messages %}{% if message['role'] == 'user' %}"
                "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def test_bpe_roundtrip():
    make_small_model_dir()
    tok = ByteLevelBPE(TMP)
    ids = tok.encode("hello world")
    assert 4 in ids  # im_start? no - "hello" is id 4
    text = tok.decode(ids)
    assert text == "hello world", text
    print("bpe roundtrip OK:", ids, "->", repr(text))


def test_specials():
    tok = ByteLevelBPE(TMP)
    ids = tok.encode("<|im_start|>hello<|im_end|>")
    assert ids[0] == 19 and ids[-1] == 20, ids
    print("specials OK:", ids)


def test_template():
    tok = ByteLevelBPE(TMP)
    out = tok.apply_template([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert out == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", repr(out)
    print("template OK:", repr(out))


def test_convert_hf():
    tj = {
        "model": {
            "type": "BPE",
            "vocab": {"<pad>": 0, "a": 1, "b": 2},
            "merges": [["a", "b"], ["ab", "a"]],
        },
        "added_tokens": [
            {"content": "<|im_start|>", "id": 3},
            {"content": "<|im_end|>", "id": 4},
        ],
    }
    tj_path = os.path.join(TMP, "tokenizer.json")
    with open(tj_path, "w", encoding="utf-8") as f:
        import json
        json.dump(tj, f)
    convert_hf_tokenizer(tj_path, TMP + "_conv")
    tok = ByteLevelBPE(TMP + "_conv")
    assert tok.token_to_id["<|im_start|>"] == 3
    assert tok.specials["<|im_end|>"] == 4
    print("convert OK")


if __name__ == "__main__":
    test_bpe_roundtrip()
    test_specials()
    test_template()
    test_convert_hf()
