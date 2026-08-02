"""Device spec scanning for androidllm model selection.

Reads /proc + statvfs + Android props directly (Termux-friendly, stdlib
only). Degrades gracefully to None/unknown on desktop dev boxes, and
accepts manual overrides via specs_from_text().
"""
import os
import platform
import re

_HOME = os.path.expanduser("~")
_DEFAULT_DIR = os.environ.get("ANDROIDLLM_DIR", os.path.join(_HOME, "androidllm"))


def _read_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except OSError:
        return []


def ram_gb():
    for line in _read_lines("/proc/meminfo"):
        if line.startswith("MemTotal:"):
            try:
                return round(int(line.split()[1]) // 1024 / 1024, 1)
            except ValueError:
                return None
    return None


def cpu_info():
    cores = 0
    model = None
    for line in _read_lines("/proc/cpuinfo"):
        if line.startswith("processor"):
            cores += 1
        elif model is None and (
                line.startswith("model name") or line.startswith("Hardware")):
            model = line.split(":", 1)[1].strip()
    return (cores or None), model


def disk_free_gb(path=None):
    import shutil
    target = path or _DEFAULT_DIR
    try:
        if hasattr(os, "statvfs"):
            st = os.statvfs(target)
            return round(st.f_bavail * st.f_frsize / 1e9, 1)
    except OSError:
        pass
    try:
        _, _, free = shutil.disk_usage(target)
        return round(free / 1e9, 1)
    except OSError:
        try:
            _, _, free = shutil.disk_usage("/")
            return round(free / 1e9, 1)
        except OSError:
            return None


def device_model():
    for line in _read_lines("/system/build.prop"):
        if line.startswith("ro.product.model") or line.startswith("ro.product.device"):
            return line.split("=", 1)[1].strip()
    return None


def battery():
    for d in ("/sys/class/power_supply/battery", "/sys/class/power_supply/BAT0"):
        try:
            cap_path = os.path.join(d, "capacity")
            with open(cap_path, encoding="utf-8") as f:
                cap = f.read().strip()
            if cap.isdigit():
                return int(cap)
        except OSError:
            continue
    return None


def device_specs(env=None):
    """Full spec snapshot as a dict (JSON-safe)."""
    cores, cpu = cpu_info()
    specs = {
        "ram_gb": ram_gb(),
        "cores": cores,
        "cpu": cpu,
        "device": device_model(),
        "disk_free_gb": disk_free_gb(),
        "platform": platform.system(),
        "arch": platform.machine(),
        "battery_pct": battery(),
        "androidllm_dir": env.get("ANDROIDLLM_DIR", _DEFAULT_DIR) if env else _DEFAULT_DIR,
    }
    if specs["disk_free_gb"] is None:
        specs["disk_free_gb"] = 0.0
    return specs


_GB = re.compile(r"(\d+(?:\.\d+)?)\s*(gb|g|mb)\b", re.IGNORECASE)
_RAM_KW = re.compile(r"\b(?:ram|memory|mem)\b", re.IGNORECASE)
_DISK_KW = re.compile(r"\b(?:storage|disk|space|rom|free)\b", re.IGNORECASE)


def specs_from_text(text):
    """Manual override: parse '8gb ram 32gb storage' style input.
    Numbers tagged by a ram/storage keyword (before or after) win;
    untagged numbers fill ram then disk."""
    text = (text or "").strip()
    specs = device_specs()
    if not text:
        return specs
    entries = []
    for m in _GB.finditer(text):
        val = float(m.group(1))
        if m.group(2).lower() == "mb":
            val /= 1024
        after_pre = re.match(r"[^0-9]*", text[m.end():m.end() + 24]).group(0)
        kw = None
        if _RAM_KW.search(after_pre):
            kw = "ram"
        elif _DISK_KW.search(after_pre):
            kw = "disk"
        else:
            before = text[max(0, m.start() - 20):m.start()]
            if re.search(r"\b(?:ram|memory|mem)\s*$", before, re.IGNORECASE):
                kw = "ram"
            elif re.search(r"\b(?:storage|disk|space|rom|free)\s*$", before, re.IGNORECASE):
                kw = "disk"
        entries.append((val, kw))
    if not entries:
        return specs
    ram = next((v for v, k in entries if k == "ram"), None)
    disk = next((v for v, k in entries if k == "disk"), None)
    for v, k in entries:
        if k is None and ram is None:
            ram = v
        elif k is None and disk is None:
            disk = v
    if ram is not None:
        specs["ram_gb"] = ram
    if disk is not None:
        specs["disk_free_gb"] = disk
    specs["manual"] = True
    return specs


def describe(specs):
    bits = []
    if specs.get("device"):
        bits.append(specs["device"])
    if specs.get("ram_gb"):
        bits.append("{} GB RAM".format(specs["ram_gb"]))
    if specs.get("cores"):
        bits.append(f"{specs['cores']} cores")
    if specs.get("cpu"):
        bits.append(specs["cpu"])
    bits.append("%s GB free" % (specs.get("disk_free_gb") or 0))
    return " / ".join(bits)
