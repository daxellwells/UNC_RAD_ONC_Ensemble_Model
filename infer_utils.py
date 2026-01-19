import json
import os
from pathlib import Path
import pickle
from typing import Any
import re
import shutil


def discover_niftis(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    files = []
    for p in input_dir.iterdir():
        if p.is_file():
            name = p.name.lower()
            if name.endswith(".nii.gz") or name.endswith(".nii"):
                files.append(p)

    files.sort(key=lambda x: x.name.lower())
    return files

def case_id_from_path(p: Path) -> str:
    name = p.name
    lname = name.lower()
    if lname.endswith(".nii.gz"):
        return name[:-7]
    if lname.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI file: {p}")

def build_datalist_dict(image_paths: list[Path], base_dir: str | Path) -> dict:
    base_dir = Path(base_dir).resolve()
    testing = []
    for img in image_paths:
        img = img.resolve()
        rel = img.relative_to(base_dir).as_posix()
        testing.append({
            "image": rel,
            "id": case_id_from_path(img),
        })
    return {"testing": testing}

def write_datalist_json(datalist: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(datalist, f, indent=2)
        f.write("\n")

def generate_datalist(job_dir: str | Path = "./infer_job") -> Path:
    job_dir = Path(job_dir).resolve()
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    datalist_path = job_dir / "datalist.json"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    imgs = discover_niftis(input_dir)
    if not imgs:
        raise RuntimeError(f"No NIfTI files found in: {input_dir}")

    datalist = build_datalist_dict(imgs, base_dir=job_dir)
    write_datalist_json(datalist, datalist_path)
    return datalist_path

def _safe_copy(src, dst):
    # copy bytes only; metadata on /mnt/c can throw PermissionError (utime/copystat)
    shutil.copyfile(src, dst)
    try:
        shutil.copystat(src, dst, follow_symlinks=False)
    except PermissionError:
        pass
    except OSError:
        pass


def copy_outputs(work_dir: Path, job_dir: Path) -> None:
    """
    Copy ensemble outputs to infer_job/output for the "drop files in -> outputs out" UX.
    Copies:
      - work_dir/ensemble_output/**/*   (if ensemble_output exists)
      - otherwise: any immediate NIfTI outputs under work_dir/ (fallback)
    """
    work_dir = Path(work_dir)
    job_dir = Path(job_dir)

    src = work_dir / "ensemble_output"
    dst = job_dir / "output"
    dst.mkdir(parents=True, exist_ok=True)

    def _copy_tree(src_dir: Path) -> int:
        count = 0
        for p in src_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(src_dir)
                out = dst / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                _safe_copy(p, out)
                count += 1
        return count

    def _flatten_if_needed(src_dir: Path) -> Path:
        """
        MONAI Auto3DSeg often writes into:
          work_dir/ensemble_output/input/*.nii.gz
        For the desired UX (infer_job/output/*.nii.gz), treat that `input/` as the real root.
        """
        input_dir = src_dir / "input"
        if not input_dir.is_dir():
            return src_dir

        # If this layout matches the common pattern, flatten it.
        # - no files directly under src_dir
        # - either only 'input' subdir exists OR we don't care and still flatten input
        has_top_files = any(p.is_file() for p in src_dir.iterdir())
        if not has_top_files:
            return input_dir

        # Conservative: if there are other top-level files, keep structure.
        return src_dir

    def _migrate_old_nested_output(dst_dir: Path) -> int:
        """
        If previous runs created infer_job/output/input/*.nii.gz, move those files up to infer_job/output/.
        """
        moved = 0
        nested = dst_dir / "input"
        if not nested.is_dir():
            return 0

        for p in nested.rglob("*"):
            if not p.is_file():
                continue
            if not (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
                continue
            target = dst_dir / p.name
            # If a file already exists at the target, keep the existing one.
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            _safe_copy(p, target)
            moved += 1

        return moved

    copied = 0

    if src.exists() and src.is_dir():
        # Clean up old nested outputs from prior runs (best-effort).
        migrated = _migrate_old_nested_output(dst)
        if migrated:
            print(f"[INFO] Migrated {migrated} existing NIfTI file(s) from {dst / 'input'} -> {dst}")

        src_eff = _flatten_if_needed(src)
        copied = _copy_tree(src_eff)
        if src_eff != src:
            print(f"[INFO] Flattened outputs: {src_eff} -> {dst}")
        print(f"[INFO] Copied {copied} file(s) from {src_eff} -> {dst}")
        return

    # Fallback: sometimes outputs are written elsewhere; copy likely NIfTI outputs.
    # (Keeps it conservative: only .nii / .nii.gz)
    for p in work_dir.rglob("*"):
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            rel = p.relative_to(work_dir)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            _safe_copy(p, out)
            copied += 1

    print(f"[WARN] {src} not found. Fallback copied {copied} NIfTI file(s) from {work_dir} -> {dst}")

def _get_template_path(obj: Any) -> str | None:
    if isinstance(obj, dict):
        return obj.get("template_path")
    if hasattr(obj, "template_path"):
        return getattr(obj, "template_path")
    return None


def _set_template_path(obj: Any, new_value: str) -> bool:
    if isinstance(obj, dict) and "template_path" in obj:
        obj["template_path"] = new_value
        return True
    if hasattr(obj, "template_path"):
        try:
            setattr(obj, "template_path", new_value)
            return True
        except Exception:
            return False
    return False


def relocate_work_dir(
    work_dir: str | Path,
    *,
    backup: bool = True,
    create_templates_dir: bool = True,
    verbose: bool = True,
) -> int:
    """
    Patch Auto3DSeg algo_object.pkl files so template_path points to
    <work_dir>/algorithm_templates on *this* machine.

    Returns: number of pkls patched.
    """
    work_dir = Path(work_dir).resolve()
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir does not exist: {work_dir}")

    algo_templates = (work_dir / "algorithm_templates").resolve()
    if not algo_templates.exists():
        if not create_templates_dir:
            raise FileNotFoundError(f"algorithm_templates not found: {algo_templates}")
        algo_templates.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[INFO] Created: {algo_templates}")

    new_template_path = algo_templates.as_posix()

    pkls = sorted(work_dir.rglob("algo_object.pkl"))
    if not pkls:
        if verbose:
            print(f"[INFO] No algo_object.pkl files found under: {work_dir}")
        return 0

    patched = 0
    for pkl_path in pkls:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        old = _get_template_path(obj)
        if old is None:
            # Not expected, but keep going.
            if verbose:
                print(f"[WARN] No template_path in: {pkl_path}")
            continue

        if old == new_template_path:
            continue

        if verbose:
            print(f"[INFO] Relocating template_path in: {pkl_path}")
            print(f"       old: {old}")
            print(f"       new: {new_template_path}")

        if backup:
            bak = pkl_path.with_suffix(pkl_path.suffix + ".bak")
            if not bak.exists():  # avoid endlessly overwriting backups
                shutil.copy2(pkl_path, bak)

        ok = _set_template_path(obj, new_template_path)
        if not ok:
            raise RuntimeError(f"Failed to set template_path in {pkl_path}")

        with open(pkl_path, "wb") as f:
            pickle.dump(obj, f)

        patched += 1

    if verbose:
        print(f"[INFO] relocate_work_dir: patched {patched} algo_object.pkl file(s).")

    return patched

def patch_monai_get_score(verbose: bool = True) -> None:
    """
    Monkey-patch MONAI Auto3DSeg BundleAlgo.get_score() so it always reads
    progress.yaml from the *local* output_path, ignoring any stale ckpt_path.

    Supports both layouts:
      - <output_path>/model/progress.yaml
      - <output_path>/model_fold*/progress.yaml

    Also supports progress.yaml being dict or list.
    """
    import math
    from pathlib import Path
    from monai.bundle import ConfigParser
    from monai.apps.auto3dseg import bundle_gen

    METRIC_KEYS = (
        "best_metric", "best_mean_dice", "best_dice", "best_score",
        "metric", "val_mean_dice", "mean_dice"
    )

    def _extract_from_dict(d: dict) -> float | None:
        for k in METRIC_KEYS:
            v = d.get(k, None)
            if isinstance(v, (int, float)):
                return float(v)

        for k in METRIC_KEYS:
            v = d.get(k, None)
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, (int, float)):
                        return float(vv)

        for container_key in ("metrics", "metric"):
            v = d.get(container_key, None)
            if isinstance(v, dict):
                for k in METRIC_KEYS:
                    vv = v.get(k, None)
                    if isinstance(vv, (int, float)):
                        return float(vv)

        return None

    def _extract_from_list(lst: list) -> float | None:
        best = None
        for item in lst:
            if isinstance(item, dict):
                v = _extract_from_dict(item)
                if v is not None:
                    best = v  # prefer later entries
        if best is not None:
            return best

        candidates = []
        for item in lst:
            if isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(v, (int, float)) and any(mk in str(k).lower() for mk in ("dice", "metric", "score")):
                        candidates.append(float(v))
        if candidates:
            return max(candidates)

        return None

    def _find_progress_files(output_path: Path) -> list[Path]:
        """
        Return candidate progress.yaml paths in priority order.
        """
        candidates: list[Path] = []

        # 1) Common
        p = output_path / "model" / "progress.yaml"
        if p.exists():
            candidates.append(p)

        # 2) Fold-based
        for fold_dir in sorted(output_path.glob("model_fold*")):
            if fold_dir.is_dir():
                pf = fold_dir / "progress.yaml"
                if pf.exists():
                    candidates.append(pf)

        # 3) Last resort: any progress.yaml under output_path (bounded)
        if not candidates:
            for pf in sorted(output_path.rglob("progress.yaml")):
                candidates.append(pf)

        return candidates

    def fixed_get_score(self):
        output_path = getattr(self, "output_path", None)
        if not output_path:
            # No safe way to locate locally; don't crash the run.
            return -math.inf

        op = Path(str(output_path)).resolve()
        progress_files = _find_progress_files(op)

        if verbose:
            # keep it light; only print if we're debugging hard
            pass

        for progress in progress_files:
            try:
                d = ConfigParser.load_config_file(str(progress))
            except Exception:
                continue

            score = None
            if isinstance(d, dict):
                score = _extract_from_dict(d)
            elif isinstance(d, list):
                score = _extract_from_list(d)

            if score is not None:
                return float(score)

        # Could not parse any progress.yaml -> treat as unusable for BestN
        return -math.inf

    bundle_gen.BundleAlgo.get_score = fixed_get_score

    if verbose:
        print("[INFO] Patched MONAI BundleAlgo.get_score() to use local progress.yaml (model/ or model_fold*/; dict/list safe)")


def disable_bundle_file_logging(work_dir: str | Path, *, verbose: bool = True) -> int:
    """
    Patches Auto3DSeg bundle-generated scripts/*/infer.py to disable file logging.
    Safe to run every time (idempotent-ish).
    Returns the number of files modified.
    """
    work_dir = Path(work_dir)
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir not found: {work_dir}")

    # Match the two-line block that triggers your crash:
    #   CONFIG["handlers"]["file"]["filename"] = ...
    #   logging.config.dictConfig(CONFIG)
    block_re = re.compile(
        r"""
        ^\s*CONFIG\["handlers"\]\["file"\]\["filename"\]\s*=\s*parser\.get_parsed_content\("infer"\)\["log_output_file"\]\s*\n
        ^\s*logging\.config\.dictConfig\(CONFIG\)\s*\n
        """,
        re.MULTILINE | re.VERBOSE,
    )

    replacement = (
        '        # Disable file logging for inference (portable across OS & DDP)\n'
        '        CONFIG["handlers"].pop("file", None)\n'
        '        logging.basicConfig(stream=sys.stdout, level=logging.INFO)\n'
    )

    modified = 0
    for p in work_dir.rglob("scripts/infer.py"):
        try:
            txt = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = p.read_text(encoding="utf-8", errors="ignore")

        if "Disable file logging for inference" in txt:
            continue  # already patched

        new_txt, n = block_re.subn(replacement, txt, count=1)
        if n == 0:
            # If the exact block isn't found (MONAI version differences), skip without breaking.
            if verbose:
                print(f"[patch] skipped (pattern not found): {p}")
            continue

        p.write_text(new_txt, encoding="utf-8")
        modified += 1
        if verbose:
            print(f"[patch] disabled file logging: {p}")

    if verbose:
        print(f"[patch] total modified infer.py files: {modified}")
    return modified

def rewrite_bundle_paths(
    work_dir: str | Path,
    *,
    datalist_path: str | Path,
    base_dir: str | Path,
    verbose: bool = True,
) -> int:
    """
    Rewrites data_list_file_path and data_file_base_dir in bundle YAML/JSON configs
    in a cross-platform, YAML-safe way.

    - YAML: use single quotes + forward slashes to avoid escape parsing.
    - JSON: keep as JSON string (backslashes escaped as needed by json.dumps).
    """
    work_dir = Path(work_dir)
    datalist_path = str(Path(datalist_path).resolve())
    base_dir = str(Path(base_dir).resolve())

    def _posix(p: str) -> str:
        return p.replace("\\", "/")

    def _yaml_quote(p: str) -> str:
        # single-quoted YAML; escape single quote by doubling it
        return "'" + _posix(p).replace("'", "''") + "'"

    def _json_quote(p: str) -> str:
        # json.dumps returns a valid JSON string literal including surrounding quotes
        return json.dumps(p)

    modified = 0

    for p in work_dir.rglob("*"):
        suf = p.suffix.lower()
        if suf not in (".yaml", ".yml", ".json"):
            continue

        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if "data_list_file_path" not in txt and "data_file_base_dir" not in txt:
            continue

        new = txt

        if suf in (".yaml", ".yml"):
            new = re.sub(
                r'^(?P<k>\s*data_list_file_path\s*:\s*)(?P<v>.*)$',
                lambda m: m.group("k") + _yaml_quote(datalist_path),
                new,
                flags=re.MULTILINE,
            )
            new = re.sub(
                r'^(?P<k>\s*data_file_base_dir\s*:\s*)(?P<v>.*)$',
                lambda m: m.group("k") + _yaml_quote(base_dir),
                new,
                flags=re.MULTILINE,
            )
        else:  # .json
            new = re.sub(
                r'("data_list_file_path"\s*:\s*)(".*?"|\'.*?\'|\S+)(\s*[,\}])',
                r"\1" + _json_quote(datalist_path) + r"\3",
                new,
                flags=re.DOTALL,
            )
            new = re.sub(
                r'("data_file_base_dir"\s*:\s*)(".*?"|\'.*?\'|\S+)(\s*[,\}])',
                r"\1" + _json_quote(base_dir) + r"\3",
                new,
                flags=re.DOTALL,
            )

        if new != txt:
            p.write_text(new, encoding="utf-8")
            modified += 1
            if verbose:
                print(f"[patch] rewrote paths in: {p}")

    if verbose:
        print(f"[patch] total config files modified: {modified}")
    return modified

def patch_bundle_ckpt_resolution(work_dir: str | Path, *, verbose: bool = True) -> int:
    """
    Patches Auto3DSeg bundle-generated scripts/*/infer.py so ckpt_name is resolved locally
    if the config contains a stale absolute path from another machine.

    Safe to run every time (idempotent).
    Returns number of files modified.
    """
    work_dir = Path(work_dir)
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir not found: {work_dir}")

    modified = 0

    # Inject helper only once per file
    helper_src = (
        "\n\ndef _resolve_ckpt_path(ckpt_name: str) -> str:\n"
        "    \"\"\"Resolve checkpoint path robustly across machines/OS.\n"
        "    If ckpt_name doesn't exist, search under this algo folder for the same filename,\n"
        "    then fall back to common checkpoint names.\n"
        "    \"\"\"\n"
        "    p = Path(str(ckpt_name))\n"
        "    if p.exists():\n"
        "        return str(p)\n"
        "\n"
        "    bundle_root = Path(__file__).resolve().parents[1]  # .../<algo>/\n"
        "    target_name = p.name if p.name else \"best_metric_model.pt\"\n"
        "\n"
        "    # 1) expected location\n"
        "    candidates = list(bundle_root.glob(f\"model_fold*/{target_name}\"))\n"
        "    if candidates:\n"
        "        return str(candidates[0])\n"
        "\n"
        "    # 2) common fallback names\n"
        "    for nm in (\"best_metric_model.pt\", \"model.pt\", \"final_model.pt\"):\n"
        "        candidates = list(bundle_root.glob(f\"model_fold*/{nm}\"))\n"
        "        if candidates:\n"
        "            return str(candidates[0])\n"
        "\n"
        "    # 3) last resort: search anywhere under algo root\n"
        "    candidates = list(bundle_root.rglob(target_name))\n"
        "    if candidates:\n"
        "        return str(candidates[0])\n"
        "\n"
        "    return str(p)\n"
    )

    # Replace the ckpt assignment line
    ckpt_line_re = re.compile(
        r'^(?P<indent>\s*)ckpt_name\s*=\s*parser\.get_parsed_content\("infer"\)\["ckpt_name"\]\s*$',
        re.MULTILINE,
    )

    for p in work_dir.rglob("scripts/infer.py"):
        txt = p.read_text(encoding="utf-8", errors="ignore")

        # Skip if already patched
        if "_resolve_ckpt_path(" in txt:
            continue

        # Must contain the standard ckpt assignment to patch
        if not ckpt_line_re.search(txt):
            if verbose:
                print(f"[patch] skipped (ckpt pattern not found): {p}")
            continue

        # Ensure Path is imported in that file (it is in stock MONAI infer.py)
        # but if MONAI changes it, we can be defensive:
        if "from pathlib import Path" not in txt:
            # try inserting after first "import" block
            txt = txt.replace("import sys\n", "import sys\nfrom pathlib import Path\n", 1)

        # Inject helper after the line: "from .train import CONFIG, pre_operation" (or non-package variant)
        insert_point = None
        for marker in (
            "from .train import CONFIG, pre_operation",
            "from train import CONFIG, pre_operation",
        ):
            idx = txt.find(marker)
            if idx != -1:
                # insert after end of that line
                insert_point = txt.find("\n", idx)
                if insert_point != -1:
                    insert_point += 1
                break

        if insert_point is None:
            # fallback: insert after imports (after first blank line following imports)
            m = re.search(r"\n\s*\n", txt)
            insert_point = m.end() if m else 0

        new_txt = txt[:insert_point] + helper_src + txt[insert_point:]

        # Now replace ckpt_name assignment to call resolver
        new_txt = ckpt_line_re.sub(
            r'\g<indent>ckpt_name = _resolve_ckpt_path(parser.get_parsed_content("infer")["ckpt_name"])',
            new_txt,
            count=1,
        )

        if new_txt != txt:
            p.write_text(new_txt, encoding="utf-8")
            modified += 1
            if verbose:
                print(f"[patch] enabled local ckpt resolution: {p}")

    if verbose:
        print(f"[patch] total modified infer.py files (ckpt): {modified}")
    return modified

def _discover_stale_roots_from_cache(work_dir: Path) -> set[str]:
    """
    Try to discover stale absolute roots like:
      /home/dwells/shared_Onc_Rad/work_dir_gloria_3
    from work_dir/cache.yaml and work_dir/input.yaml.

    Returns a set of roots (POSIX style).
    """
    roots: set[str] = set()
    candidates = [work_dir / "cache.yaml", work_dir / "input.yaml"]

    # Heuristic: capture '/home/.../work_dir_<something>' up to (but not including) '/segresnet_' or '/swinunetr_'
    pat = re.compile(r"(/home/[^ \n\r\t'\"\\]+?/work_dir[^ \n\r\t'\"\\]*)(?=/(?:segresnet_|swinunetr_)|[\s'\"\\]|$)")

    for f in candidates:
        if not f.exists():
            continue
        s = f.read_text(encoding="utf-8", errors="ignore")
        for m in pat.finditer(s):
            roots.add(m.group(1))
    return roots


def _recursive_replace_strings(obj: Any, repl: dict[str, str]) -> Any:
    """
    Recursively replace substrings in any string fields inside obj.
    Works for dict/list/tuple and objects with __dict__.
    """
    if isinstance(obj, str):
        out = obj
        for old, new in repl.items():
            if old in out:
                out = out.replace(old, new)
        return out

    if isinstance(obj, list):
        return [_recursive_replace_strings(x, repl) for x in obj]

    if isinstance(obj, tuple):
        return tuple(_recursive_replace_strings(x, repl) for x in obj)

    if isinstance(obj, dict):
        # keys can be strings too
        return { _recursive_replace_strings(k, repl): _recursive_replace_strings(v, repl) for k, v in obj.items() }

    # best-effort: patch attributes on simple objects
    if hasattr(obj, "__dict__"):
        for k, v in list(obj.__dict__.items()):
            obj.__dict__[k] = _recursive_replace_strings(v, repl)
        return obj

    return obj


def patch_segresnet_stale_workdir_paths(
    work_dir: str | Path,
    *,
    verbose: bool = True,
    backup: bool = True,
) -> dict[str, int]:
    """
    Minimal, targeted fix for SegResNet: rewrite stale Linux work_dir roots
    inside segresnet_*/algo_object.pkl and segresnet_*/configs/hyper_parameters.yaml.

    Safe to run every time (idempotent). Returns counts.
    """
    work_dir = Path(work_dir).resolve()
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir not found: {work_dir}")

    # Discover old roots from cache/input (nice-to-have, but not sufficient alone)
    old_roots = _discover_stale_roots_from_cache(work_dir)
    if verbose:
        if old_roots:
            print("[patch] discovered stale roots:")
            for r in sorted(old_roots):
                print(f"        {r}")
        else:
            print("[patch] no stale roots discovered from cache/input; will still patch known segresnet root")

    # Map old roots -> local work_dir (POSIX form works well even on Windows for many MONAI internals)
    new_root = work_dir.as_posix()
    repl = {r: new_root for r in old_roots}

    # Always include the known training root (critical for segresnet pkls)
    KNOWN_OLD_ROOT = "/home/dwells/shared_Onc_Rad/work_dir_gloria_3"
    repl[KNOWN_OLD_ROOT] = new_root

    # 1) Patch segresnet algo_object.pkl (prefer raw-byte gating)
    pkl_modified = 0
    needle = KNOWN_OLD_ROOT.encode("utf-8")

    for pkl_path in sorted(work_dir.glob("segresnet_*/algo_object.pkl")):
        raw = pkl_path.read_bytes()

        # Fast skip + confirms whether we should be patching this pickle at all
        if needle not in raw:
            if verbose:
                print(f"[patch] pkl skip (no stale bytes): {pkl_path}")
            continue

        obj = pickle.loads(raw)
        new_obj = _recursive_replace_strings(obj, repl)
        new_raw = pickle.dumps(new_obj)

        if new_raw != raw:
            if backup:
                bak_dir = work_dir / "_bak"
                bak_dir.mkdir(exist_ok=True)
                bak = bak_dir / f"{pkl_path.parent.name}_{pkl_path.name}.bak"
                if not bak.exists():
                    bak.write_bytes(raw)

            pkl_path.write_bytes(new_raw)
            pkl_modified += 1
            if verbose:
                print(f"[patch] patched pkl: {pkl_path}")
        else:
            # Useful signal: pickle had stale bytes but traversal didn't change anything
            if verbose:
                print(f"[patch] WARNING: pkl had stale bytes but no changes were made: {pkl_path}")

    # 2) Patch segresnet hyper_parameters.yaml (text replace only)
    hp_modified = 0
    for hp_path in sorted(work_dir.glob("segresnet_*/configs/hyper_parameters.yaml")):
        txt = hp_path.read_text(encoding="utf-8", errors="ignore")
        new_txt = txt
        for old, new in repl.items():
            new_txt = new_txt.replace(old, new)
        if new_txt != txt:
            if backup:
                bak_dir = work_dir / "_bak"
                bak_dir.mkdir(exist_ok=True)
                bak = bak_dir / f"{hp_path.parent.name}_{hp_path.name}.bak"
                if not bak.exists():
                    bak.write_text(txt, encoding="utf-8")
            hp_path.write_text(new_txt, encoding="utf-8")
            hp_modified += 1
            if verbose:
                print(f"[patch] patched hyper_parameters: {hp_path}")

    if verbose:
        print(f"[patch] segresnet stale-path fix complete: pkls={pkl_modified}, hyper_parameters={hp_modified}")

    return {"pkls": pkl_modified, "hyper_parameters": hp_modified}

def patch_bundle_script_stale_paths(
    work_dir: str | Path,
    *,
    verbose: bool = True,
    backup: bool = True,
) -> dict[str, int]:
    """
    Patch stale absolute paths inside bundle Python scripts (e.g., infer.py/train.py),
    especially the common '/home/dwells/...' leftovers that cause PermissionError on WSL/Linux.

    Idempotent and safe to run every time.
    """
    work_dir = Path(work_dir).resolve()
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir not found: {work_dir}")

    old_roots = _discover_stale_roots_from_cache(work_dir)
    new_root = work_dir.as_posix()

    # Replace any discovered stale roots -> this work_dir
    repl = {r: new_root for r in old_roots}

    # Extra “known bad” roots observed in your logs
    repl.setdefault("/home/dwells/shared_Onc_Rad/work_dir_gloria_3", new_root)
    repl.setdefault("/home/dwells", str(Path.home().as_posix()))  # safer than failing mkdir

    modified = 0
    scripts = sorted(work_dir.glob("**/scripts/*.py"))
    for p in scripts:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        new_txt = txt
        for old, new in repl.items():
            new_txt = new_txt.replace(old, new)

        if new_txt != txt:
            if backup:
                bak_dir = work_dir / "_bak"
                bak_dir.mkdir(exist_ok=True)
                bak = bak_dir / (p.as_posix().replace("/", "__") + ".bak")
                if not bak.exists():
                    bak.write_text(txt, encoding="utf-8")

            p.write_text(new_txt, encoding="utf-8")
            modified += 1
            if verbose:
                print(f"[patch] patched script: {p}")

    if verbose:
        print(f"[patch] bundle script stale-path fix complete: scripts={modified}")
    return {"scripts": modified}



# =========================
# Portable "self-fix" layer
# =========================

_AUTOSEG_HINTS = (
    "work_dir", "algorithm_templates", "model_fold", "progress.yaml",
    "configs/", "scripts/", "ckpt", "checkpoint", "infer.py", "train.py",
)

# POSIX: capture up to the work_dir* folder name
_RE_POSIX_WORKDIR_ROOT = re.compile(r"(/[^\s'\"\n\r\t]+?/work_dir[^/\s'\"\n\r\t]+)")
# Windows: capture up to the work_dir* folder name
_RE_WIN_WORKDIR_ROOT = re.compile(r"([A-Za-z]:\\[^\s'\"\n\r\t]+?\\work_dir[^\\\s'\"\n\r\t]+)")


def _looks_like_autoseg_path(s: str) -> bool:
    s_low = s.lower()
    return any(h.lower() in s_low for h in _AUTOSEG_HINTS)


def _discover_stale_roots_from_text_tree(work_dir: Path, *, max_bytes: int = 4_000_000) -> set[str]:
    roots: set[str] = set()
    exts = {".py", ".yaml", ".yml", ".json", ".txt"}
    for p in work_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        try:
            # avoid huge binaries accidentally ending with .txt
            if p.stat().st_size > max_bytes:
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if not _looks_like_autoseg_path(txt):
            continue

        for m in _RE_POSIX_WORKDIR_ROOT.finditer(txt):
            roots.add(m.group(1))
        for m in _RE_WIN_WORKDIR_ROOT.finditer(txt):
            roots.add(m.group(1))
    return roots


def _discover_stale_roots_from_algo_pickles(work_dir: Path, *, max_bytes: int = 200_000_000) -> set[str]:
    roots: set[str] = set()

    # NOTE: We're not unpickling here (safe + fast). We only scan bytes for recognizable roots.
    # This catches stale '/home/<user>/.../work_dir_*' that often shows up in algo_object.pkl.
    re_bytes = re.compile(rb"(/home/[^\x00\s]+?/work_dir[^/\x00\s]+)")
    for pkl_path in work_dir.rglob("algo_object.pkl"):
        try:
            sz = pkl_path.stat().st_size
            if sz > max_bytes:
                continue
            raw = pkl_path.read_bytes()
        except Exception:
            continue

        if b"work_dir" not in raw and b"/home/" not in raw:
            continue

        for m in re_bytes.finditer(raw):
            try:
                roots.add(m.group(1).decode("utf-8", errors="ignore"))
            except Exception:
                pass

    return roots


def discover_stale_workdir_roots(work_dir: str | Path, *, verbose: bool = True) -> set[str]:
    """Discover absolute roots like '/home/<user>/.../work_dir_*' embedded in configs/scripts/pickles."""
    work_dir = Path(work_dir).resolve()

    roots: set[str] = set()

    # 1) cache/input (already implemented elsewhere)
    try:
        roots |= set(_discover_stale_roots_from_cache(work_dir))
    except Exception:
        pass

    # 2) text tree
    roots |= _discover_stale_roots_from_text_tree(work_dir)

    # 3) pickle byte-scan (DISABLED)
    # Scanning binary pickles for roots can produce garbage strings and is unnecessary.

    # normalize trivial variants
    roots = {r.rstrip("/\\") for r in roots if r}

    if verbose:
        if roots:
            print("[self-fix] discovered candidate stale roots:")
            for r in sorted(roots):
                print(f"          {r}")
        else:
            print("[self-fix] no candidate stale roots discovered")

    return roots


def _root_should_patch(root: str, *, new_root: str) -> bool:
    """Patch a root if it's not already our current work_dir and it's not a valid/writable target."""
    # If the root already points inside current work_dir, keep it.
    try:
        # string contains check is fine; we don't want to resolve foreign roots
        if new_root and root.replace('\\', '/').startswith(new_root.rstrip('/') + '/'):
            return False
    except Exception:
        pass

    # If it exists and is writable, prefer NOT to patch (avoid breaking a valid Linux setup).
    try:
        # os.path.exists handles both POSIX and Windows paths on Windows; on Linux, Windows roots will be False.
        if os.path.exists(root):
            # For directories, check write access; for files, check parent directory.
            target = root
            if os.path.isfile(root):
                target = os.path.dirname(root)
            if os.access(target, os.W_OK):
                return False
            # Exists but not writable -> patch (common WSL permission issue when trying to create /home/<otheruser>/...)
            return True
    except Exception:
        # If we can't even stat it, treat as patchable.
        return True

    # Doesn't exist -> patch.
    return True


def build_workdir_replacements(
    stale_roots: set[str],
    work_dir: str | Path,
    *,
    verbose: bool = True,
) -> dict[str, str]:
    work_dir = Path(work_dir).resolve()
    new_root = work_dir.as_posix()

    repl: dict[str, str] = {}
    for r in sorted(stale_roots):
        if not r:
            continue
        if _root_should_patch(r, new_root=new_root):
            repl[r] = new_root

    if verbose:
        if repl:
            print("[self-fix] replacement map:")
            for old, new in repl.items():
                print(f"          {old}  ->  {new}")
        else:
            print("[self-fix] no roots require patching after safety checks")

    return repl


def _maybe_backup_text(path: Path, original_text: str, *, bak_dir: Path) -> None:
    bak_dir.mkdir(exist_ok=True)
    bak = bak_dir / (path.as_posix().replace("/", "__") + ".bak")
    if not bak.exists():
        bak.write_text(original_text, encoding="utf-8")


def patch_text_tree_stale_roots(
    work_dir: str | Path,
    repl: dict[str, str],
    *,
    exts: tuple[str, ...] = (".py", ".yaml", ".yml", ".json"),
    verbose: bool = True,
    backup: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    """Patch stale roots across scripts/configs in work_dir tree."""
    work_dir = Path(work_dir).resolve()
    modified = 0
    if not repl:
        return {"text_files": 0}

    bak_dir = work_dir / "_bak"
    for p in work_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in set(e.lower() for e in exts):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if not _looks_like_autoseg_path(txt):
            continue

        new_txt = txt
        for old, new in repl.items():
            new_txt = new_txt.replace(old, new)

        if new_txt != txt:
            modified += 1
            if verbose:
                print(f"[self-fix] patch text: {p}")
            if not dry_run:
                if backup:
                    _maybe_backup_text(p, txt, bak_dir=bak_dir)
                p.write_text(new_txt, encoding="utf-8")

    return {"text_files": modified}


def patch_algo_object_pickles_stale_roots(
    work_dir: str | Path,
    repl: dict[str, str],
    *,
    verbose: bool = True,
    backup: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Patch stale roots inside algo_object.pkl files by *unpickle -> rewrite -> repickle*.

    Why not raw byte replacement?
      Pickle strings are length-prefixed; changing byte lengths without updating the
      pickle stream corrupts the file (e.g. UnpicklingError: invalid load key).

    This routine:
      1) gates on whether any old-root bytes exist in the file (fast skip),
      2) loads the pickle safely,
      3) recursively rewrites string fields,
      4) dumps back to bytes and writes (with optional backup).

    Returns a dict with count of pkls modified.
    """
    work_dir = Path(work_dir).resolve()
    if not repl:
        return {"algo_pkls": 0}

    bak_dir = work_dir / "_bak"
    modified = 0

    # Fast "needle" set for gating
    needles = [old.encode("utf-8", errors="ignore") for old in repl.keys() if old]

    for pkl_path in sorted(work_dir.rglob("algo_object.pkl")):
        try:
            raw = pkl_path.read_bytes()
        except Exception:
            continue

        # Gate: skip pkls that clearly don't contain any stale-root bytes
        if needles and not any(n in raw for n in needles):
            continue

        # Load -> rewrite -> dump
        try:
            obj = pickle.loads(raw)
        except Exception as e:
            if verbose:
                print(f"[self-fix] WARNING: could not unpickle (skipping): {pkl_path} ({e})")
            continue

        new_obj = _recursive_replace_strings(obj, repl)

        try:
            new_raw = pickle.dumps(new_obj, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            if verbose:
                print(f"[self-fix] WARNING: could not repickle (skipping): {pkl_path} ({e})")
            continue

        if new_raw == raw:
            continue

        modified += 1
        if verbose:
            print(f"[self-fix] patch pkl: {pkl_path}")

        if dry_run:
            continue

        if backup:
            bak_dir.mkdir(exist_ok=True)
            bak = bak_dir / f"{pkl_path.parent.name}_{pkl_path.name}.bak"
            if not bak.exists():
                bak.write_bytes(raw)

        pkl_path.write_bytes(new_raw)

    return {"algo_pkls": modified}


def self_fix(
    work_dir: str | Path,
    *,
    datalist_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    patch_scripts: bool = True,
    patch_pkls: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
    backup: bool = True,
) -> dict[str, Any]:
    """One-shot portability repair pass. Call this BEFORE EnsembleRunner.run()."""
    work_dir_p = Path(work_dir).resolve()
    if not work_dir_p.exists():
        raise FileNotFoundError(f"work_dir not found: {work_dir_p}")

    summary: dict[str, Any] = {"work_dir": str(work_dir_p), "dry_run": dry_run}

    # 0) Patch algorithm_templates references to local work_dir paths
    try:
        summary["relocate_work_dir"] = relocate_work_dir(work_dir_p, create_templates_dir=True, verbose=verbose)
    except Exception as e:
        summary["relocate_work_dir_error"] = str(e)

    # 1) Rewrite datalist/base_dir paths in bundle configs if requested
    if datalist_path is not None and base_dir is not None:
        try:
            rewrite_bundle_paths(
                work_dir=work_dir_p,
                datalist_path=Path(datalist_path),
                base_dir=Path(base_dir),
                verbose=verbose,
            )
            summary["rewrite_bundle_paths"] = True
        except Exception as e:
            summary["rewrite_bundle_paths_error"] = str(e)

    # 2) Fix checkpoint resolution & disable file logging (both can contain absolute stale paths)
    try:
        patch_bundle_ckpt_resolution(work_dir_p)
        summary["patch_bundle_ckpt_resolution"] = True
    except Exception as e:
        summary["patch_bundle_ckpt_resolution_error"] = str(e)

    try:
        disable_bundle_file_logging(work_dir=work_dir_p, verbose=verbose)
        summary["disable_bundle_file_logging"] = True
    except Exception as e:
        summary["disable_bundle_file_logging_error"] = str(e)

    # 3) Discover stale roots and build a safe replacement map
    stale_roots = discover_stale_workdir_roots(work_dir_p, verbose=verbose)
    repl = build_workdir_replacements(stale_roots, work_dir_p, verbose=verbose)
    summary["stale_roots_found"] = sorted(stale_roots)
    summary["replacements"] = dict(repl)

    # 4) Patch scripts/configs and algo pickles
    if patch_scripts:
        summary.update(
            patch_text_tree_stale_roots(
                work_dir_p, repl, verbose=verbose, backup=backup, dry_run=dry_run
            )
        )
    if patch_pkls:
        summary.update(
            patch_algo_object_pickles_stale_roots(
                work_dir_p, repl, verbose=verbose, backup=backup, dry_run=dry_run
            )
        )

    # 5) Patch BundleAlgo.get_score so BestN scoring uses local progress.yaml
    try:
        patch_monai_get_score(verbose=verbose)
        summary["patch_monai_get_score"] = True
    except Exception as e:
        summary["patch_monai_get_score_error"] = str(e)

    if verbose:
        print("[self-fix] summary:", summary)

    return summary
