"""Continuous batching: one engine serves many concurrent chat requests.

A request becomes a Session (prompt ids + per-slot KV + sampler state) and a
single scheduler thread round-robins one work unit per visit:
  - prefill: a chunk of prompt positions (fairness for long prompts)
  - decode: one token (or one speculative step)
so N chats all make progress instead of queueing behind each other.

The SessionPool keeps finished sessions so the next request with a matching
prompt prefix reuses the KV (cross-turn prefix reuse survives batching).

Env knobs:
  ANDROIDLLM_BATCH_MAX   max concurrent slots (default 4)
  ANDROIDLLM_BATCH_MEM_MB  KV budget for all slots+pool (default 700)
"""
import os
import threading
import time
from collections import deque


def kv_bytes_per_session(engine, ctx_len=None):
    """Per-session KV cache size: layers * kv_heads * head_dim * (k+v) * 2B."""
    ctx = ctx_len or engine.ctx_len
    n = engine.canon["layers"]
    kh = engine.canon["kv_heads"]
    hd = engine.canon["head_dim"]
    return ctx * kh * hd * 2 * 2 * n


class Session:
    """One generation in progress. Created fresh or reused from the pool.
    Invariant: kv filled 0..pos-1, token = id at position pos."""

    def __init__(self, engine, prompt_ids, max_new_tokens=64, temperature=0.8,
                 top_p=0.9, min_p=0.0, stop_ids=(), grammar=None,
                 prefill_chunk=32):
        self.engine = engine
        self.prompt_ids = list(prompt_ids)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.stop_ids = tuple(stop_ids)
        self.grammar = grammar
        self.prefill_chunk = max(1, int(prefill_chunk))
        self.kv = None
        self.pos = 0
        self.token = None
        self.prefilled = False
        self.done = False
        self.generated = []
        self.buf_parts = []
        self.draft_kv = None
        self.error = None
        self.started = time.time()
        self.using_spec = (engine.draft is not None and engine.spec_k > 0
                           and grammar is None)

    # -- pool reuse ------------------------------------------------------

    def reuse_kv(self, pfx):
        """Continue from a pooled KV that already holds prompt[0..pfx-1].
        kv must be set before calling (taken from a pooled session)."""
        self.prefilled = True
        self.pos = pfx
        self.token = self.prompt_ids[pfx]
        self.using_spec = (self.engine.draft is not None
                           and self.engine.spec_k > 0 and self.grammar is None)

    def common_prefix(self, other_ids):
        n = min(len(self.prompt_ids), len(other_ids), self.engine.ctx_len)
        p = 0
        while p < n and self.prompt_ids[p] == other_ids[p]:
            p += 1
        return p

    # -- work units ------------------------------------------------------

    def step(self):
        """Advance one work unit. Returns (finished, new_tokens)."""
        e = self.engine
        if self.done:
            return True, []
        if not self.prefilled:
            return self._prefill()
        if len(self.generated) >= self.max_new_tokens:
            self.done = True
            return True, []
        if self.token in self.stop_ids:
            self.done = True
            return True, []
        if self.using_spec:
            emitted, self.token, self.pos, n_draft = e._spec_step(
                self.kv, self.draft_kv, self.pos, self.token,
                self.temperature, self.top_p, self.min_p)
            for i, t in enumerate(emitted):
                if t in self.stop_ids:
                    self.token = t
                    break
                if len(self.generated) >= self.max_new_tokens:
                    break
                self.generated.append(t)
                self._count(t, i, n_draft)
            if len(self.generated) >= self.max_new_tokens:
                self.done = True
            return self.done, self.generated[-len(emitted):]
        self.token = e._step(self.kv, self.pos, self.token,
                             self.temperature, self.top_p, self.min_p,
                             self.grammar, self.buf_parts)
        self.pos += 1
        if self.token in self.stop_ids:
            self.done = True
            return True, []
        self.generated.append(self.token)
        if self.grammar is not None:
            self.buf_parts.append(self.engine.tokenizer.decode([self.token]))
        self._count(self.token, 0, 0)
        return False, [self.token]

    def _count(self, tok, i, n_draft):
        if self.using_spec:
            if i < n_draft:
                self.engine.stats["spec_accepted"] += 1
            else:
                self.engine.stats["spec_bonus"] += 1
        self.engine.stats["tokens_served"] += 1

    def _prefill(self):
        e = self.engine
        end = min(self.pos + self.prefill_chunk, len(self.prompt_ids) - 1)
        while self.pos < end:
            e._forward(e.model.embed[self.prompt_ids[self.pos]].reshape(
                1, e.model.hidden), self.kv, self.pos)
            self.pos += 1
        if self.pos < len(self.prompt_ids) - 1:
            return False, []
        self.token = self.prompt_ids[-1]
        self.pos = len(self.prompt_ids) - 1
        self.prefilled = True
        if self.using_spec:
            self.draft_kv = e.draft.model.prepare_kv(e.ctx_len)
            for i in range(len(self.prompt_ids)):
                e.draft._forward(e.draft.model.embed[self.prompt_ids[i]].reshape(
                    1, e.draft.model.hidden), self.draft_kv, i)
        return False, []


class SessionPool:
    """Finished sessions kept for KV prefix reuse. Acquire pops the session
    whose prompt shares the longest prefix with the new prompt."""

    def __init__(self, engine, max_pooled=4):
        self.engine = engine
        self.max_pooled = max_pooled
        self._sessions = []
        self._lock = threading.Lock()

    def acquire(self, prompt_ids):
        with self._lock:
            best = None
            best_p = 1
            for i, s in enumerate(self._sessions):
                p = s.common_prefix(prompt_ids)
                if p > best_p:
                    best_p = p
                    best = i
            if best is None:
                s = Session(self.engine, prompt_ids)
                s.kv = self.engine.model.prepare_kv(self.engine.ctx_len)
                return s, 0
            s = self._sessions.pop(best)
            s.prompt_ids = list(prompt_ids)
            s.max_new_tokens = 64
            s.grammar = None
            s.generated = []
            s.buf_parts = []
            s.error = None
            s.done = False
            s.started = time.time()
            s.reuse_kv(min(best_p, len(prompt_ids) - 1))
            return s, best_p

    def put(self, sess):
        if sess.error or sess.kv is None:
            return
        with self._lock:
            self._sessions.append(sess)
            if len(self._sessions) > self.max_pooled:
                self._sessions.pop(0)

    def __len__(self):
        with self._lock:
            return len(self._sessions)


class BatchScheduler:
    """Round-robin scheduler thread. submit() returns True when a slot was
    taken; the done callback fires on the scheduler thread."""

    def __init__(self, engine, max_slots=4):
        self.engine = engine
        kv = kv_bytes_per_session(engine)
        budget = int(os.environ.get("ANDROIDLLM_BATCH_MEM_MB", "700")) * 1048576
        cap = max(1, budget // max(kv, 1))
        self.max_slots = min(max(1, int(max_slots)), cap)
        self._active = deque()
        self._cond = threading.Condition()
        self._stop = False
        self.stats = {"active": 0, "queued": 0, "max": self.max_slots,
                      "completed": 0, "stepped": 0, "rejected": 0}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, session, done_cb):
        with self._cond:
            if len(self._active) >= self.max_slots:
                self.stats["rejected"] += 1
                return False
            self._active.append((session, done_cb))
            self.stats["queued"] += 1
            self._cond.notify()
        return True

    def active_count(self):
        with self._cond:
            return len(self._active)

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def _run(self):
        e = self.engine
        while True:
            with self._cond:
                while not self._active and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                slot = self._active.popleft()
                self.stats["queued"] = max(0, self.stats["queued"] - 1)
            sess, done_cb = slot
            finished = False
            try:
                finished, _toks = sess.step()
            except Exception as exc:
                sess.error = str(exc)
                sess.done = True
                finished = True
            self.stats["stepped"] += 1
            if finished:
                self.stats["completed"] += 1
                try:
                    done_cb(sess)
                except Exception:
                    pass
            else:
                with self._cond:
                    self._active.append(slot)
                    self.stats["queued"] += 1
            if e.throttle_ms > 0:
                time.sleep(e.throttle_ms / 1000.0)

    def snapshot(self):
        with self._cond:
            active = len(self._active)
            return {"active": active, "queued": self.stats["queued"],
                    "max": self.max_slots, "completed": self.stats["completed"],
                    "rejected": self.stats["rejected"]}
