import json
import os
import re


def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\xa1"), ord("\xac") + 1)) +
          list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class ByteLevelBPE:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "vocab.txt"), encoding="utf-8") as f:
            self.ids_to_tokens = [line.rstrip("\n") for line in f]
        self.token_to_id = {t: i for i, t in enumerate(self.ids_to_tokens)}
        merges = []
        with open(os.path.join(model_dir, "merges.txt"), encoding="utf-8") as f:
            for line in f:
                pair = line.rstrip("\n")
                if not pair:
                    continue
                a, b = pair.split()
                merges.append((a, b))
        self.merges = merges
        self.merge_rank = {p: i for i, p in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.unicode_to_byte = {v: k for k, v in self.byte_encoder.items()}
        self.specials = {}
        sp_path = os.path.join(model_dir, "special_tokens.json")
        if os.path.exists(sp_path):
            with open(sp_path, encoding="utf-8") as f:
                self.specials = {k: int(v) for k, v in json.load(f).items()}
        self.template = None
        tpl_path = os.path.join(model_dir, "template.txt")
        if os.path.exists(tpl_path):
            with open(tpl_path, encoding="utf-8") as f:
                self.template = f.read()

    def encode_piece(self, text):
        chars = [self.byte_encoder[b] for b in text.encode("utf-8")]
        if not chars:
            return []
        word = self._bpe(chars)
        out = []
        for t in word:
            tid = self.token_to_id.get(t)
            if tid is None:
                tid = self.token_to_id.get(self.byte_encoder.get(ord(t)) if len(t) == 1 else 3)
            out.append(tid if tid is not None else 3)
        return out

    def _bpe(self, word):
        if len(word) == 1:
            return list(word)
        pairs = list(zip(word[:-1], word[1:]))
        while True:
            bigram = min(pairs, key=lambda p: self.merge_rank.get(p, 1 << 30))
            if bigram not in self.merge_rank:
                break
            first, second = bigram
            new = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = new
            if len(word) == 1:
                break
            pairs = list(zip(word[:-1], word[1:]))
        return list(word)

    def encode(self, text):
        ids = []
        specials = sorted(self.specials.keys(), key=len, reverse=True)
        if not specials:
            return self.encode_piece(text)
        pos = 0
        n = len(text)
        while pos < n:
            hit = None
            for s in specials:
                if text.startswith(s, pos):
                    hit = s
                    break
            if hit:
                ids.append(self.specials[hit])
                pos += len(hit)
            else:
                nxt = len(text)
                for s in specials:
                    j = text.find(s, pos)
                    if j != -1 and j < nxt:
                        nxt = j
                if nxt > pos:
                    ids.extend(self.encode_piece(text[pos:nxt]))
                pos = nxt
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            t = self.ids_to_tokens[i]
            if len(t) == 1 and t in self.unicode_to_byte:
                chars.append(self.unicode_to_byte[t])
            else:
                for c in t:
                    if c in self.unicode_to_byte:
                        chars.append(self.unicode_to_byte[c])
        return bytes(chars).decode("utf-8", errors="replace")

    def apply_template(self, messages, add_generation_prompt=False):
        if self.template is None:
            raise ValueError("model has no chat template")
        return render_template(self.template, messages,
                               {"add_generation_prompt": add_generation_prompt})


# --------------------------------------------------------------------------
# minimal Jinja-subset for chat templates
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"(\{\{[^}]*\}\}|\{%[^%]*%\}|\{#[^#]*#\})")
_EXPR_SPLIT = re.compile(r"^(.*?)(\|\s*(\w+)\s*)$")


def _resolve(path, env):
    cur = env
    for part in path.strip().split("."):
        if not part:
            continue
        m = re.match(r"(\w+)((?:\[[^\]]*\])*)", part)
        if not m:
            return None
        name = m.group(1)
        if isinstance(cur, dict) and name not in cur:
            return None
        cur = cur[name]
        for idx in re.finditer(r"\[(.*?)\]", m.group(2)):
            k = idx.group(1).strip()
            if len(k) >= 2 and k[0] in "'\"" and k[-1] == k[0]:
                k = k[1:-1]
            else:
                try:
                    k = int(k)
                except ValueError:
                    k = _resolve(k, env)
            try:
                cur = cur[k]
            except (KeyError, IndexError, TypeError):
                return None
    return cur


def _value_of(expr, env):
    expr = expr.strip()
    m = _EXPR_SPLIT.match(expr)
    if m:
        expr = m.group(1)
    if expr.startswith("'") and expr.endswith("'") and len(expr) > 1:
        v = expr[1:-1]
    elif expr.startswith('"') and expr.endswith('"') and len(expr) > 1:
        v = expr[1:-1]
    else:
        try:
            v = int(expr)
        except ValueError:
            v = _resolve(expr, env)
    if m and m.group(3) == "trim" and isinstance(v, str):
        v = v.strip()
    return v


def _truthy(v):
    if v is None:
        return False
    if isinstance(v, str):
        return len(v) > 0
    if isinstance(v, (list, dict, tuple)):
        return len(v) > 0
    return bool(v)


def _cond_true(text, env):
    t = text.strip()
    if len(t) > 1 and t[0] == "(" and t[-1] == ")":
        return _cond_true(t[1:-1], env)
    if " and " in t:
        return all(_cond_true(p, env) for p in t.split(" and "))
    if " or " in t:
        return any(_cond_true(p, env) for p in t.split(" or "))
    m = re.match(r"^(.*?)\s*==\s*(.*)$", t)
    if m:
        a = _value_of(m.group(1), env)
        b = _value_of(m.group(2), env)
        if isinstance(a, str) and isinstance(b, str):
            return a == b
        return a == b
    m = re.match(r"^(.*?)\s*!=\s*(.*)$", t)
    if m:
        a = _value_of(m.group(1), env)
        b = _value_of(m.group(2), env)
        return a != b
    m = re.match(r"^(.*?)\s+is\s+(not\s+)?defined\s*$", t)
    if m:
        defined = _resolve(m.group(1).strip(), env) is not None
        return (not defined) if m.group(2) else defined
    m = re.match(r"^not\s+(.+)$", t)
    if m:
        return not _cond_true(m.group(1), env)
    return _truthy(_resolve(t, env))


def _parse(tokens, pos, stoppers=None):
    stoppers = stoppers or set()
    nodes = []
    while pos < len(tokens):
        tok = tokens[pos]
        if tok.startswith("{#"):
            pos += 1
            continue
        if tok.startswith("{{"):
            nodes.append(("expr", tok[2:-2].strip()))
            pos += 1
            continue
        if tok.startswith("{%"):
            s = tok[2:-2].strip()
            if s in stoppers or ("elif" in stoppers and s.startswith("elif ")):
                return nodes, pos
            if s.startswith("if "):
                branches = []
                body, pos = _parse(tokens, pos + 1, {"endif", "else", "elif"})
                branches.append((s[3:], body))
                while pos < len(tokens):
                    st = tokens[pos][2:-2].strip() if tokens[pos].startswith("{%") else None
                    if st == "else":
                        else_body, pos = _parse(tokens, pos + 1, {"endif", "else", "elif"})
                        nodes.append(("if", branches, else_body))
                        break
                    if st and st.startswith("elif "):
                        body, pos = _parse(tokens, pos + 1, {"endif", "else", "elif"})
                        branches.append((st[5:], body))
                    else:
                        nodes.append(("if", branches, []))
                        break
                continue
            if s.startswith("for "):
                m = re.match(r"for\s+(\w+)\s+in\s+(.+)$", s)
                if m:
                    body, pos = _parse(tokens, pos + 1, {"endfor"})
                    nodes.append(("for", m.group(1), m.group(2).strip(), body))
                else:
                    pos += 1
                continue
            if s.startswith("set "):
                m = re.match(r"set\s+(\w+)\s*=\s*(.+)$", s)
                if m:
                    nodes.append(("set", m.group(1), m.group(2).strip()))
                pos += 1
                continue
            pos += 1
            continue
        nodes.append(("text", tok))
        pos += 1
    return nodes, pos


def _eval_nodes(nodes, env, out):
    for node in nodes:
        kind = node[0]
        if kind == "text":
            out.append(node[1])
        elif kind == "expr":
            expr = node[1]
            plus = expr.split(" + ")
            parts = []
            for p in plus:
                v = _value_of(p, env)
                if v is None:
                    parts = None
                    break
                parts.append(str(v) if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False, separators=(",", ":")))
            if parts is not None:
                out.append("".join(parts))
        elif kind == "if":
            taken = False
            for cond, body in node[1]:
                if _cond_true(cond, env):
                    _eval_nodes(body, env, out)
                    taken = True
                    break
            if not taken:
                _eval_nodes(node[2], env, out)
        elif kind == "for":
            var, expr, body = node[1], node[2], node[3]
            seq = _resolve(expr, env) or []
            n = len(seq)
            for idx, item in enumerate(seq):
                env[var] = item
                env["loop"] = {"index": idx + 1, "last": idx == n - 1}
                env["loop0"] = {"index": idx, "last": idx == n - 1}
                _eval_nodes(body, env, out)
            env.pop(var, None)
            env.pop("loop", None)
            env.pop("loop0", None)
        elif kind == "set":
            env[node[1]] = _value_of(node[2], env)


def render_template(template, messages, env=None):
    env = dict(env or {})
    env["messages"] = messages
    nodes, _ = _parse([t for t in _TOKEN_RE.split(template) if t], 0)
    out = []
    _eval_nodes(nodes, env, out)
    return "".join(out)


# --------------------------------------------------------------------------
# HF tokenizer.json conversion
# --------------------------------------------------------------------------

def convert_hf_tokenizer(tokenizer_json_path, out_dir, tokenizer_config_path=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(tokenizer_json_path, encoding="utf-8") as f:
        tj = json.load(f)
    model = tj.get("model", {})
    vocab = model.get("vocab", {})
    merges = model.get("merges", [])
    added = tj.get("added_tokens", [])
    tokens = list(vocab.keys())
    for at in added:
        if at["content"] not in vocab and at.get("id") is not None:
            tokens.insert(min(at["id"], len(tokens)), at["content"])
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t + "\n")
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for m in merges:
            if isinstance(m, list):
                f.write(" ".join(m) + "\n")
            else:
                f.write(m + "\n")
    specials = {}
    for at in added:
        if at["content"] not in vocab and at.get("id") is not None:
            specials[at["content"]] = at["id"]
    with open(os.path.join(out_dir, "special_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(specials, f)
    template = None
    if tokenizer_config_path and os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, encoding="utf-8") as f:
            tc = json.load(f)
        template = tc.get("chat_template")
    if not template:
        template = (
            "{% for message in messages %}{% if message['role'] == 'system' %}"
            "{{ message['content'] }}{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n"
            "{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
    with open(os.path.join(out_dir, "template.txt"), "w", encoding="utf-8") as f:
        f.write(template)
