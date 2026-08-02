"""Strict JSON output via token-level logit masking.

A character-level automaton walks the partial JSON buffer; the allowed next
character set follows the schema node. Candidate tokens are simulated
against the automaton, and invalid ones are masked out before sampling.

Fast path: per-token structural flags (contains quote/brace/comma/digit...)
turn O(vocab * len) simulation into vectorized masks; only the few tokens
containing structural characters are fully simulated.

Schema subset: {type: object|array|string|number|integer|boolean|null|any},
properties, required, items, enum. A type of [T, "null"] is treated as
optional T.
"""

import copy

import numpy as np

_WS = " \t\r\n"
_DIGITS = "0123456789"
_NUM_CHARS = _DIGITS + "-.eE+"


def _norm(schema):
    if schema is None:
        return {"type": "any"}
    if not isinstance(schema, dict):
        raise ValueError("grammar schema must be an object")
    t = schema.get("type")
    if isinstance(t, list):
        types = [x for x in t if x != "null"]
        if len(types) == 1:
            s = dict(schema)
            s["type"] = types[0]
            s["nullable"] = True
            return s
        raise ValueError("union types not supported (only optional null)")
    s = dict(schema)
    s.setdefault("type", "any")
    return s


class JsonGrammar:
    def __init__(self, schema):
        self.schema = _norm(schema)
        self._f_tok = None

    # -- automaton --------------------------------------------------------

    def _root(self):
        return {"kind": "root", "node": self.schema, "st": "val",
                "key": None, "seen": set(), "required": set(self.schema.get("required") or []),
                "child": None}

    def _machine(self, buf):
        m = {"stack": [self._root()], "top": None, "lit": "",
             "lit_targets": (), "num_last": ""}
        for ch in buf:
            if not self._consume(m, ch):
                # Buffer is already invalid (e.g. model ran without a
                # grammar before this turn) - degrade to permissive.
                m = {"stack": [self._root()], "top": None, "lit": "",
                     "lit_targets": (), "num_last": ""}
                break
        return m

    def _consume(self, m, ch):
        top = m["top"]
        if top == "str":
            return self._in_str(m, ch)
        if top == "esc":
            if ch in '"\\/bfnrtu':
                m["top"] = "str"
                return True
            return False
        if top == "lit":
            return self._in_lit(m, ch)
        if top == "num":
            return self._in_num(m, ch)
        return self._in_struct(m, ch)

    def _prim_done(self, m):
        """A scalar value just completed inside the containing frame."""
        frame = m["stack"][-1]
        if frame["kind"] == "root":
            frame["st"] = "done"
        else:
            frame["st"] = "after_val"
        m["top"] = None

    def _in_str(self, m, ch):
        if ch == '"':
            if m["str_is_key"]:
                frame = m["stack"][-1]
                key = m["sbuf"]
                frame["key"] = key
                frame["seen"].add(key)
                node = frame["node"]
                props = node.get("properties") or {} if isinstance(node, dict) else {}
                frame["child"] = props.get(key) if props else None
                frame["st"] = "colon"
                m["top"] = None
            else:
                self._prim_done(m)
            return True
        if ch == "\\":
            m["top"] = "esc"
            return True
        m["sbuf"] += ch
        return True

    def _lit_targets(self, node):
        t = node.get("type") if node else "any"
        if t == "boolean":
            return ("true", "false")
        if t == "null":
            return ("null",)
        if t == "enum":
            return tuple(str(v) for v in (node.get("enum") or ()))
        return ("true", "false", "null")

    def _in_lit(self, m, ch):
        if ch in _WS:
            return False
        m["lit"] += ch
        targets = m.get("lit_targets") or ()
        matched = False
        completed = False
        for t in targets:
            if t.startswith(m["lit"]):
                matched = True
                if len(t) == len(m["lit"]):
                    completed = True
        if not matched:
            return False
        if completed:
            self._prim_done(m)
        return True

    def _in_num(self, m, ch):
        if ch in _WS or ch in ",}]":
            self._prim_done(m)
            return self._in_struct(m, ch)
        prev = m["num_last"]
        if ch in _DIGITS:
            m["num_last"] = ch
            return True
        if ch == "-" and prev == "":
            m["num_last"] = "-"
            return True
        if ch in ".eE" and prev not in (".", "e", "E", "-"):
            m["num_last"] = ch
            return True
        if ch == "+" and prev in ("e", "E"):
            m["num_last"] = "+"
            return True
        return False

    def _close_container(self, m, ch):
        frame = m["stack"][-1]
        if frame["kind"] == "obj" and ch == "}":
            if not frame["required"].issubset(frame["seen"]):
                return False
            m["stack"].pop()
            self._prim_done(m)
            return True
        if frame["kind"] == "arr" and ch == "]":
            m["stack"].pop()
            self._prim_done(m)
            return True
        return False

    def _node_for(self, frame):
        if frame["kind"] == "obj" and frame["st"] == "val":
            return frame["child"] or {}
        return frame["node"]

    def _in_struct(self, m, ch):
        frame = m["stack"][-1]
        st = frame["st"]
        if st == "key":
            if ch in _WS:
                return True
            if ch == '"':
                frame["st"] = "keystr"
                m["top"] = "str"
                m["str_is_key"] = True
                m["sbuf"] = ""
                return True
            if ch == "}":
                return self._close_container(m, ch)
            return False
        if st == "colon":
            if ch in _WS:
                return True
            if ch == ":":
                frame["st"] = "val"
                return True
            return False
        if st in ("val", "elem", "done"):
            if st == "done":
                return ch in _WS
            if ch in _WS:
                return True
            node = self._node_for(frame)
            t = node.get("type") if node else "any"
            if ch == "{":
                if t not in ("any", "object"):
                    return False
                sub = node if isinstance(node, dict) else {}
                props = sub.get("properties") or {} if t == "object" else {}
                m["stack"].append({"kind": "obj", "node": sub, "st": "key",
                                   "key": None, "seen": set(),
                                   "required": set(sub.get("required") or []),
                                   "child": None})
                return True
            if ch == "[":
                if t not in ("any", "array"):
                    return False
                sub = node if isinstance(node, dict) else {}
                m["stack"].append({"kind": "arr", "node": sub, "st": "elem",
                                   "key": None, "seen": set(),
                                   "required": set(),
                                   "child": None})
                return True
            if ch == '"':
                if t not in ("any", "string"):
                    return False
                m["top"] = "str"
                m["str_is_key"] = False
                m["sbuf"] = ""
                return True
            if ch == "-" or ch in _DIGITS:
                if t not in ("any", "number", "integer"):
                    return False
                m["top"] = "num"
                m["num_last"] = ch
                return True
            if ch in "tfn":
                if t == "enum":
                    if not any(str(v).startswith(ch) for v in (node.get("enum") or ())):
                        return False
                elif t not in ("any", "boolean", "null"):
                    return False
                m["top"] = "lit"
                m["lit"] = ch
                m["lit_targets"] = self._lit_targets(node)
                return True
            return False
        if st == "after_val":
            if ch in _WS:
                return True
            if ch == ",":
                frame["st"] = "key" if frame["kind"] == "obj" else "elem"
                return True
            if frame["kind"] in ("obj", "arr"):
                return self._close_container(m, ch)
            return False
        return False

    # -- token masking ----------------------------------------------------

    def _flags(self, tokenizer):
        if self._f_tok is tokenizer:
            return self._flags_cache
        n = len(tokenizer.ids_to_tokens)
        texts = [tokenizer.decode([i]) for i in range(n)]
        self._flags_cache = {
            "n": n,
            "texts": texts,
            "first": np.array([ord(t[0]) if t else 0 for t in texts], dtype=np.uint32),
            "len_one": np.array([len(t) == 1 for t in texts], dtype=bool),
            "empty": np.array([not t for t in texts], dtype=bool),
            "has_quote": np.array(['"' in t for t in texts], dtype=bool),
            "has_bslash": np.array(["\\" in t for t in texts], dtype=bool),
            "num_ok": np.array([bool(t) and all(c in _NUM_CHARS for c in t) for t in texts],
                               dtype=bool),
        }
        self._f_tok = tokenizer
        return self._flags_cache

    def _mask_structural(self, m, flags):
        frame = m["stack"][-1]
        st = frame["st"]
        chars = set()
        if st in ("val", "elem"):
            chars.update(_WS)
            node = self._node_for(frame)
            t = node.get("type") if node else "any"
            if t == "string":
                chars.add('"')
            elif t in ("number", "integer"):
                chars.update(_DIGITS)
                chars.add("-")
            elif t == "boolean":
                chars.update("tf")
            elif t == "null":
                chars.add("n")
            elif t == "object":
                chars.add("{")
            elif t == "array":
                chars.add("[")
            else:
                chars.update('{["-')
                chars.update(_DIGITS)
                chars.update("tfn")
        elif st == "key":
            chars.update(_WS)
            chars.add('"')
            if frame["required"].issubset(frame["seen"]):
                chars.add("}")
        elif st == "colon":
            chars.update(_WS)
            chars.add(":")
        elif st == "after_val":
            chars.update(_WS)
            chars.add(",")
            if frame["kind"] in ("obj", "arr"):
                chars.add("}" if frame["kind"] == "obj" else "]")
        elif st == "done":
            chars.update(_WS)
        n = flags["n"]
        out = np.zeros(n, dtype=bool)
        first = flags["first"]
        ok_first = np.fromiter((chr(c) in chars for c in first), dtype=bool, count=n)
        single = ok_first & flags["len_one"]
        out[single] = True
        multi = np.flatnonzero(ok_first & ~flags["len_one"])
        for i in multi:
            m2 = copy.deepcopy(m)
            if self._token_ok(m2, flags["texts"][i]):
                out[i] = True
        return out

    def _sim_candidates(self, m, flags, starts):
        """Simulate every token whose first char ord is in `starts`; return mask."""
        out = np.zeros(flags["n"], dtype=bool)
        ok = np.fromiter((int(c) in starts for c in flags["first"]), dtype=bool,
                         count=flags["n"])
        for i in np.flatnonzero(ok):
            m2 = copy.deepcopy(m)
            if self._token_ok(m2, flags["texts"][i]):
                out[i] = True
        return out

    def allowed_mask(self, buf, tokenizer):
        flags = self._flags(tokenizer)
        m = self._machine(buf)
        top = m["top"]
        terms = _WS + ",}]"
        if top == "str":
            base = ~flags["has_quote"] & ~flags["has_bslash"] & ~flags["empty"]
            cand = np.flatnonzero(flags["has_quote"] | flags["has_bslash"])
            for i in cand:
                m2 = copy.deepcopy(m)
                if self._token_ok(m2, flags["texts"][i]):
                    base[i] = True
            return base
        if top == "esc":
            out = np.zeros(flags["n"], dtype=bool)
            esc = set('"\\/bfnrtu')
            ok = np.fromiter((chr(c) in esc for c in flags["first"]), dtype=bool,
                             count=flags["n"])
            cands = np.flatnonzero(ok)
            for i in cands:
                m2 = copy.deepcopy(m)
                if self._token_ok(m2, flags["texts"][i]):
                    out[i] = True
            return out
        if top == "lit":
            targets = m.get("lit_targets") or ()
            prefix = m["lit"]
            nxt = {ord(t[len(prefix)]) for t in targets
                   if len(t) > len(prefix) and t.startswith(prefix)}
            starts = nxt | {ord(c) for c in terms}
            return self._sim_candidates(m, flags, starts)
        if top == "num":
            out = flags["num_ok"].copy()
            if len(m.get("num_last") or ""):
                # a completed number may be followed by terminators
                out |= self._sim_candidates(m, flags, {ord(c) for c in terms})
            return out
        return self._mask_structural(m, flags)

    def _token_ok(self, m, text):
        for ch in text:
            if not self._consume(m, ch):
                return False
        return True
