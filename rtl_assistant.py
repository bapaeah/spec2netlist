#!/usr/bin/env python3
"""
RTL Assistant: Local LLM-powered RTL Design & Verification Tool
Uses locally hosted models via Ollama (zero tokens, 100% offline).
Integrates with Icarus Verilog (`iverilog`) and Verilator for automated self-healing compilation.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_API_BASE = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("RTL_MODEL", "qwen2.5-coder:7b")
FALLBACK_MODEL = "qwen2.5-coder:7b"


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, system_prompt: str = "", context: list = None, return_context: bool = False):
    """Call Ollama /api/generate endpoint with optional conversation context."""
    url = f"{OLLAMA_API_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.95,
            "num_ctx": 4096,
        }
    }
    if context:
        payload["context"] = context
    if system_prompt:
        payload["system"] = system_prompt

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    try:
        # 600s timeout to allow cold-loading large model weights into RAM
        with urllib.request.urlopen(req, timeout=600) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            if return_context:
                return res_data.get("response", ""), res_data.get("context", [])
            return res_data.get("response", "")
    except urllib.error.URLError as e:
        print(f"\n[Error] Connection or Timeout talking to Ollama at {OLLAMA_API_BASE}.")
        print("Details:", e)
        print("\nTroubleshooting tips:")
        print(" 1. If this was a timeout, the model may be too large for your RAM/VRAM.")
        print("    Try using a lighter model: --model qwen2.5-coder:7b")
        print(" 2. Test model responsiveness directly in terminal:")
        print(f"    ollama run {model} \"hi\"")
        print(" 3. Make sure Ollama is running (`curl http://127.0.0.1:11434`).")
        sys.exit(1)


def extract_code_block(text: str, default_lang: str = "verilog") -> str:
    """Extract code block enclosed by ``` or return cleaned text."""
    pattern = rf"```(?:{default_lang}|systemverilog|vhdl|v)?\s*\n([\s\S]*?)```"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return matches[0].strip()
    
    # Fallback: find any code block
    generic_matches = re.findall(r"```[a-zA-Z0-9_-]*\s*\n([\s\S]*?)```", text)
    if generic_matches:
        return generic_matches[0].strip()
    
    return text.strip()


def check_status():
    """Verify Ollama connection, models, and EDA tools."""
    print("==================================================")
    print("     RTL Assistant Local Environment Health       ")
    print("==================================================")

    # 1. Check Ollama API
    print(f"\n[1] Checking Ollama API at {OLLAMA_API_BASE}...")
    try:
        with urllib.request.urlopen(f"{OLLAMA_API_BASE}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            print(f"    ✓ Ollama is online! Available models: {', '.join(models) if models else 'None yet'}")
    except Exception as e:
        print(f"    ✗ Ollama is NOT responding: {e}")
        print("    -> Run `ollama serve` or open the Ollama application.")

    # 2. Check EDA tools
    print("\n[2] Checking local EDA tools:")
    tools = ["iverilog", "vvp", "verilator", "yosys", "gtkwave"]
    for t in tools:
        path = subprocess.run(["which", t], capture_output=True, text=True).stdout.strip()
        if path:
            print(f"    ✓ {t:10} found at {path}")
        else:
            print(f"    - {t:10} not installed")

    # 3. Check Memory Compiler
    print("\n[3] Checking Memory Compiler:")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    memgen_cand = [
        os.path.join(base_dir, "memory", "memgen.py"),
        os.path.join(base_dir, "..", "memory", "memgen.py"),
    ]
    memgen_found = next((p for p in memgen_cand if os.path.isfile(p)), None)
    if memgen_found:
        print(f"    ✓ memgen.py  found at {memgen_found}")
    else:
        print("    - memgen.py  not found in memory/ directory")

    # 4. Check Open Standard Cell Libraries
    print("\n[4] Checking Open Standard Cell Libraries:")
    for lib, name in [("nangate45", "NanGate 45nm"), ("sky130", "SkyWater 130nm")]:
        l_file = os.path.join(base_dir, "techlib", lib, f"{lib}.lib")
        v_file = os.path.join(base_dir, "techlib", lib, f"{lib}_cells.v")
        if os.path.isfile(l_file) and os.path.isfile(v_file):
            print(f"    ✓ {name:16} ready ({lib}.lib + {lib}_cells.v)")
        else:
            print(f"    - {name:16} not found in techlib/{lib}/")

    print("\nEnvironment check completed.\n")


def extract_spec_from_file(file_path: str) -> str:
    """Extract plain text from .md, .txt, .pdf, .docx, .doc, .rtf, .json, .yaml."""
    if not os.path.exists(file_path):
        print(f"[Error] Specification file '{file_path}' not found.")
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()

    # 1. Plain text / Markdown / HDL / Config files
    if ext in [".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".v", ".sv", ".csv"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    # 2. PDF Document
    if ext == ".pdf":
        # Try pdftotext CLI (poppler)
        try:
            res = subprocess.run(["pdftotext", file_path, "-"], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except FileNotFoundError:
            pass
        # Fallback to python pypdf if installed
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            if pages:
                return "\n".join(pages).strip()
        except ImportError:
            pass
        print(f"[Error] Could not extract text from '{file_path}'.")
        print("Please ensure 'pdftotext' (poppler) is installed or run: pip install pypdf")
        sys.exit(1)

    # 3. Word Document (.docx)
    if ext == ".docx":
        # Extract via pure Python standard library (zipfile + XML)
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(file_path) as docx:
                tree = ET.fromstring(docx.read("word/document.xml"))
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                lines = []
                for p in tree.iterfind(".//w:p", ns):
                    p_text = "".join(node.text for node in p.iterfind(".//w:t", ns) if node.text)
                    if p_text:
                        lines.append(p_text)
                if lines:
                    return "\n".join(lines).strip()
        except Exception:
            pass

        # Fallback to macOS textutil
        try:
            res = subprocess.run(["textutil", "-convert", "txt", file_path, "-stdout"], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except FileNotFoundError:
            pass

    # 4. Rich Text / Legacy Word (.doc, .rtf)
    if ext in [".doc", ".rtf"]:
        try:
            res = subprocess.run(["textutil", "-convert", "txt", file_path, "-stdout"], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except FileNotFoundError:
            pass

    print(f"[Warning] Unknown extension '{ext}'. Attempting raw UTF-8 read...")
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[Error] Failed to read '{file_path}': {e}")
        sys.exit(1)


def cmd_generate(args):
    """Generate RTL module from specification string or specification document."""
    if not getattr(args, "spec", None) and getattr(args, "pos_spec", None):
        if os.path.isfile(args.pos_spec):
            args.file = args.pos_spec
        else:
            args.spec = args.pos_spec

    spec_text = ""
    if getattr(args, "file", None):
        print(f"Loading specification from document: '{args.file}'...")
        spec_text = extract_spec_from_file(args.file)
        if getattr(args, "section", None):
            idx = spec_text.lower().find(args.section.lower())
            if idx != -1:
                spec_text = spec_text[idx:idx + 8000]
                print(f"Extracted section matching '{args.section}'")
    elif getattr(args, "spec", None):
        spec_text = args.spec
    else:
        print("[Error] Either --spec '<text>' or --file <spec_path> must be provided.")
        sys.exit(1)

    print(f"Loaded specification ({len(spec_text)} chars / ~{len(spec_text)//4} tokens).")

    prompt = f"""Generate a clean, synthesizable Verilog / SystemVerilog module based on the following specification document:

Specification:
{spec_text}

Requirements:
- Strict synthesis compliance (no latches, non-blocking <= for sequential, blocking = for combinational).
- Fully parameterizable where appropriate.
- ANSI style port declaration.
- Add clear comments explaining functionality and interface ports.
- Return ONLY the code inside a ```verilog code block.
"""
    print(f"Generating RTL with model '{args.model}'...")
    print("Connecting to Ollama (first run may take a moment while model weights load into RAM)...")
    response = call_ollama(prompt, model=args.model)
    code = extract_code_block(response)

    if args.output:
        with open(args.output, "w") as f:
            f.write(code + "\n")
        print(f"✓ RTL module written to {args.output}")
    else:
        print("\n" + "="*40 + " GENERATED RTL " + "="*40)
        print(code)
        print("="*95)


def cmd_testbench(args):
    """Generate self-checking testbench for an existing RTL file."""
    rtl_file = getattr(args, "rtl_file", None) or getattr(args, "rtl_file_opt", None)
    if not rtl_file or not os.path.isfile(rtl_file):
        print(f"[Error] RTL file '{rtl_file}' not found.")
        sys.exit(1)

    args.rtl_file = rtl_file

    with open(args.rtl_file, "r") as f:
        rtl_code = f.read()

    prompt = f"""Generate a complete, self-checking Verilog testbench for the following RTL module:

RTL Code:
```verilog
{rtl_code}
```

Requirements:
- Self-checking with automated stimulus generation and output verification.
- Clock generator (e.g. 100MHz / 10ns period).
- Reset sequence (assert, wait a few cycles, deassert synchronously).
- Extensive corner case test vectors.
- Dump waveform for GTKWave:
    $dumpfile("dump.vcd");
    $dumpvars(0, <tb_module_name>);
- Print clear PASS/FAIL summary and finish with $finish;
- Return ONLY the testbench code inside a ```verilog code block.
"""
    print(f"Generating testbench for '{args.rtl_file}' with model '{args.model}'...")
    response = call_ollama(prompt, model=args.model)
    code = extract_code_block(response)

    if args.output:
        with open(args.output, "w") as f:
            f.write(code + "\n")
        print(f"✓ Testbench written to {args.output}")
    else:
        print("\n" + "="*40 + " GENERATED TESTBENCH " + "="*40)
        print(code)
        print("="*99)


def compile_and_simulate(rtl_file: str, tb_file: str, sim_binary: str = "sim.out"):
    """Compile with iverilog and run simulation with vvp."""
    if not shutil.which("iverilog"):
        return False, (
            "TOOL ERROR: 'iverilog' (Icarus Verilog) is not installed or not in PATH.\n"
            "  • Termux (Android): run 'pkg install iverilog'\n"
            "  • Ubuntu/Debian:    run 'sudo apt install -y iverilog'\n"
            "  • macOS:            run 'brew install icarus-verilog'\n"
            "  • Windows:          run 'winget install -e --id Icarus.Verilog' (or download from https://bleyer.org/icarus/)"
        )

    if not shutil.which("vvp"):
        return False, "TOOL ERROR: 'vvp' simulation runtime engine not found in PATH."

    comp_cmd = ["iverilog", "-g2012", "-o", sim_binary, rtl_file, tb_file]
    print(f"Compiling: {' '.join(comp_cmd)}")
    try:
        comp_res = subprocess.run(comp_cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return False, "TOOL ERROR: 'iverilog' executable not found."

    if comp_res.returncode != 0:
        return False, f"COMPILATION ERROR:\n{comp_res.stderr}\n{comp_res.stdout}"

    print(f"Running simulation: vvp {sim_binary}")
    try:
        sim_res = subprocess.run(["vvp", sim_binary], capture_output=True, text=True)
    except FileNotFoundError:
        return False, "TOOL ERROR: 'vvp' executable not found."

    if sim_res.returncode != 0 or "FAIL" in sim_res.stdout or "$fatal" in sim_res.stdout:
        return False, f"SIMULATION ERROR/FAILURE:\n{sim_res.stderr}\n{sim_res.stdout}"

    return True, sim_res.stdout


def cmd_verify(args):
    """Compile and verify RTL + Testbench, auto-fixing if errors occur."""
    if not shutil.which("iverilog"):
        print("\n[Error] 'iverilog' (Icarus Verilog) is not installed or not in PATH.")
        print("To install Icarus Verilog:")
        print("  • Termux (Samsung Tab) : pkg install iverilog")
        print("  • Ubuntu / Debian      : sudo apt install -y iverilog")
        print("  • macOS                : brew install icarus-verilog")
        print("  • Windows              : winget install -e --id Icarus.Verilog (or https://bleyer.org/icarus/)\n")
        sys.exit(1)

    rtl_file = getattr(args, "rtl_file", None) or getattr(args, "rtl_file_opt", None)
    tb_file = getattr(args, "tb_file", None) or getattr(args, "tb_file_opt", None)

    if not rtl_file or not os.path.isfile(rtl_file):
        print(f"[Error] RTL file '{rtl_file}' not found.")
        sys.exit(1)
    if not tb_file or not os.path.isfile(tb_file):
        print(f"[Error] Testbench file '{tb_file}' not found.")
        sys.exit(1)

    args.rtl_file = rtl_file
    args.tb_file = tb_file

    max_retries = args.max_retries
    for attempt in range(1, max_retries + 1):
        print(f"\n--- Verification Attempt {attempt}/{max_retries} ---")
        success, log = compile_and_simulate(args.rtl_file, args.tb_file)
        if success:
            print("\n✓ SUCCESS: Verification passed cleanly!")
            print("Simulation output:")
            print(log)
            return

        print(f"\n✗ Error detected during verification:\n{log}")
        if attempt == max_retries:
            print("\nReached max retries. Fix aborted.")
            return

        print(f"Attempting self-healing fix with model '{args.model}'...")
        with open(args.rtl_file, "r") as f:
            rtl_code = f.read()
        with open(args.tb_file, "r") as f:
            tb_code = f.read()

        fix_prompt = f"""The following RTL module and Testbench produced an error during iverilog compilation or simulation.

RTL Code ({args.rtl_file}):
```verilog
{rtl_code}
```

Testbench Code ({args.tb_file}):
```verilog
{tb_code}
```

Error Log:
{log}

Please fix either the RTL or the Testbench (or both).
Format your response with:
FILE: {args.rtl_file}
```verilog
<fixed rtl code>
```

FILE: {args.tb_file}
```verilog
<fixed tb code>
```
"""
        response = call_ollama(fix_prompt, model=args.model)

        # Flexible parser for file blocks
        rtl_block = None
        tb_block = None

        patterns_rtl = [
            rf"(?:FILE:?|\*{0,2}FILE:?\*{0,2}|#{1,4})\s*`?{re.escape(args.rtl_file)}`?[\s\S]*?```(?:verilog|systemverilog|v)?\s*\n([\s\S]*?)```",
            rf"```(?:verilog|systemverilog|v)?\s*//\s*{re.escape(args.rtl_file)}\s*\n([\s\S]*?)```"
        ]
        patterns_tb = [
            rf"(?:FILE:?|\*{0,2}FILE:?\*{0,2}|#{1,4})\s*`?{re.escape(args.tb_file)}`?[\s\S]*?```(?:verilog|systemverilog|v)?\s*\n([\s\S]*?)```",
            rf"```(?:verilog|systemverilog|v)?\s*//\s*{re.escape(args.tb_file)}\s*\n([\s\S]*?)```"
        ]

        for p in patterns_rtl:
            m = re.search(p, response, re.IGNORECASE)
            if m:
                rtl_block = m.group(1).strip()
                break

        for p in patterns_tb:
            m = re.search(p, response, re.IGNORECASE)
            if m:
                tb_block = m.group(1).strip()
                break

        # Fallback: if single or dual code blocks were returned without explicit headers
        if not rtl_block and not tb_block:
            blocks = re.findall(r"```(?:verilog|systemverilog|v)?\s*\n([\s\S]*?)```", response, re.IGNORECASE)
            if len(blocks) == 1:
                if args.tb_file in log:
                    tb_block = blocks[0].strip()
                elif args.rtl_file in log:
                    rtl_block = blocks[0].strip()
            elif len(blocks) >= 2:
                rtl_block = blocks[0].strip()
                tb_block = blocks[1].strip()

        if rtl_block:
            with open(args.rtl_file, "w") as f:
                f.write(rtl_block + "\n")
            print(f"Updated {args.rtl_file}")

        if tb_block:
            with open(args.tb_file, "w") as f:
                f.write(tb_block + "\n")
            print(f"Updated {args.tb_file}")


def cmd_synth(args):
    """
    Synthesize RTL module into open standard cell library using Yosys / ABC.
    Supports NanGate 45nm and SkyWater 130nm.
    """
    rtl_file = getattr(args, "rtl_file", None) or getattr(args, "rtl_file_opt", None)
    if not rtl_file:
        print("[Error] Please specify an RTL Verilog/SystemVerilog file to synthesize.")
        print("Usage: rtl --synth <module.v> [--top <top_module>] [--lib nangate45|sky130] [-o <netlist.v>]")
        sys.exit(1)

    if not os.path.exists(rtl_file):
        print(f"[Error] File '{rtl_file}' does not exist.")
        sys.exit(1)

    # Determine top module
    top_module = getattr(args, "top", None)
    if not top_module:
        with open(rtl_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        m = re.search(r"\bmodule\s+([a-zA-Z0-9_]+)", content)
        if m:
            top_module = m.group(1)
        else:
            top_module = os.path.splitext(os.path.basename(rtl_file))[0]

    # Target library
    target_lib = getattr(args, "lib", "nangate45")
    lib_key = "sky130" if "sky" in target_lib.lower() else "nangate45"
    lib_display = "SkyWater 130nm (sky130_fd_sc_hd)" if lib_key == "sky130" else "NanGate 45nm Open Cell Library"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    lib_candidates = [
        os.path.join(base_dir, "techlib", lib_key, f"{lib_key}.lib"),
        os.path.join("techlib", lib_key, f"{lib_key}.lib"),
    ]
    lib_path = next((p for p in lib_candidates if os.path.isfile(p)), None)

    cells_v_candidates = [
        os.path.join(base_dir, "techlib", lib_key, f"{lib_key}_cells.v"),
        os.path.join("techlib", lib_key, f"{lib_key}_cells.v"),
    ]
    cells_v_path = next((p for p in cells_v_candidates if os.path.isfile(p)), None)

    out_file = getattr(args, "output", None) or f"{top_module}_gate.v"

    # Check for Yosys
    yosys_bin = shutil.which("yosys") or shutil.which("yowasp-yosys")
    print(f"\n==================================================")
    print(f"      Open Library Cell Synthesis Engine          ")
    print(f"==================================================")
    print(f"  • Top Module      : {top_module}")
    print(f"  • Input RTL       : {rtl_file}")
    print(f"  • Target Library  : {lib_display}")
    print(f"  • Output Netlist  : {out_file}")
    if lib_path:
        print(f"  • Liberty Model   : {lib_path}")
    if cells_v_path:
        print(f"  • GLS Cell Models : {cells_v_path}")

    if not yosys_bin:
        print("\n[Warning] Yosys is not installed on this system.")
        print("Falling back to built-in structural cell mapping...")
        from spec2rtl_upf.src.synth_netlist_engine import SynthNetlistEngine
        engine = SynthNetlistEngine(target_library=lib_key)
        out_p = os.path.dirname(out_file) or "."
        engine.run_synthesis([Path(rtl_file)], top_module, None, Path(out_p))
        return

    # Execute Yosys synthesis
    yosys_cmd = [
        yosys_bin,
        "-p",
        (
            f"read_verilog -sv \"{rtl_file}\"; "
            f"hierarchy -check -top {top_module}; "
            f"proc; flatten; opt; "
            f"synth -top {top_module}; "
            f"dfflibmap -liberty \"{lib_path}\"; "
            f"abc -liberty \"{lib_path}\"; "
            f"clean; opt; "
            f"write_verilog -noattr \"{out_file}\"; "
            f"stat -liberty \"{lib_path}\""
        )
    ]

    print(f"\n[Running Yosys Logic Synthesis & Technology Mapping...]")
    res = subprocess.run(yosys_cmd, capture_output=True, text=True)

    if res.returncode != 0:
        print(f"\n[Error] Yosys synthesis failed:")
        print(res.stderr or res.stdout)
        sys.exit(1)

    stdout = res.stdout

    # Parse and display synthesis report
    area_m = re.search(r"Chip area for module.*:\s*([0-9.]+)", stdout)
    seq_m = re.search(r"used for sequential elements:\s*([0-9.]+)", stdout)
    cells_m = re.search(r"(\d+)\s+[0-9.]+\s+cells", stdout)

    cell_counts = {}
    for count_str, cname in re.findall(r"^\s*(\d+)\s+[0-9.-]+\s+([a-zA-Z0-9_]+)\s*$", stdout, re.MULTILINE):
        if cname not in ["wires", "wire", "public", "ports", "port", "cells"]:
            cell_counts[cname] = int(count_str)

    print(f"\n✓ Synthesis & Technology Mapping Succeeded!")
    print(f"--------------------------------------------------")
    if cells_m:
        print(f"  Total Standard Cells   : {int(cells_m.group(1)):,}")
    if area_m:
        tot_area = float(area_m.group(1))
        seq_area = float(seq_m.group(1)) if seq_m else 0.0
        comb_area = tot_area - seq_area
        print(f"  Total Silicon Area     : {tot_area:,.2f} µm²")
        print(f"    - Sequential Area    : {seq_area:,.2f} µm²")
        print(f"    - Combinational Area : {comb_area:,.2f} µm²")

    if cell_counts:
        print(f"\n  Mapped Cell Breakdown:")
        for cname, count in sorted(cell_counts.items(), key=lambda x: -x[1]):
            print(f"    {cname:28} : {count:4d}")

    print(f"--------------------------------------------------")
    print(f"Gate-Level Netlist written to: {out_file}")
    if cells_v_path:
        print(f"\nTo run Gate-Level Simulation (GLS):")
        print(f"  iverilog -g2012 \"{cells_v_path}\" \"{out_file}\" <testbench.v> && vvp a.out\n")


def cmd_convert(args):
    """
    Convert Markdown (.md) documents to PDF, TXT, HTML, DOCX, RTF, LaTeX, or JSON.
    """
    input_file = getattr(args, "input_file", None) or getattr(args, "input_file_opt", None)
    if not input_file:
        print("[Error] Please specify an input Markdown (.md) file to convert.")
        print("Usage: rtl --convert <document.md> [-o <output.ext>] [--to pdf|txt|html|docx|rtf|tex|json|all]")
        sys.exit(1)

    try:
        from md_converter import MarkdownConverter
    except ImportError:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, base_dir)
        from md_converter import MarkdownConverter

    print(f"\n==================================================")
    print(f"      Markdown Universal Document Converter       ")
    print(f"==================================================")
    print(f"  • Source Document : {input_file}")

    try:
        out_target = getattr(args, "output", None)
        fmt_target = getattr(args, "to", None)
        results = MarkdownConverter.convert_file(input_file, out_target, fmt_target)
        print(f"\n✓ Completed {len(results)} conversion(s) successfully!\n")
    except Exception as e:
        print(f"\n[Error] Conversion failed: {e}")
        sys.exit(1)


def cmd_chat(args):
    """Interactive chat session specialized for RTL and VLSI engineering."""
    print(f"\nStarting interactive RTL & VLSI Engineer chat with '{args.model}'...")
    print("Ask any question on VLSI, digital design, STA, CDC, UPF, or paste RTL to debug.")
    print("Commands: type 'clear' to reset chat memory, or 'exit' / 'quit' to end.\n")

    system_prompt = (
        "You are an expert Principal ASIC/FPGA Digital Hardware and VLSI Design Engineer. "
        "You have deep expertise in digital logic design, Verilog/SystemVerilog, VHDL, "
        "Static Timing Analysis (STA), Clock Domain Crossing (CDC), low-power UPF 2.1 design, "
        "ASIC synthesis, physical design fundamentals, DFT, on-chip memories, and computer architecture. "
        "Answer technical questions thoroughly with clear formulas, timing diagrams, "
        "trade-off analyses, or synthesizable code examples whenever helpful."
    )

    chat_context = None

    while True:
        try:
            user_input = input("\nVLSI-Engineer> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting.")
                break
            if user_input.lower() in ["clear", "reset"]:
                chat_context = None
                print("Conversation history cleared.")
                continue
            
            print("\n[Thinking & Generating]...")
            reply, chat_context = call_ollama(
                user_input,
                model=args.model,
                system_prompt=system_prompt,
                context=chat_context,
                return_context=True
            )
            print("\n" + reply)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def find_memgen() -> str:
    """Locate the Technology-Agnostic Memory Compiler (memgen.py)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "memory", "memgen.py"),
        os.path.join(base_dir, "..", "memory", "memgen.py"),
        os.path.expanduser("~/LLM_Local/memory/memgen.py"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def cmd_memory(args, unknown_args):
    """Compile universal SystemVerilog memory modules via memgen.py or LLM natural language spec."""
    memgen_path = find_memgen()
    if not memgen_path:
        print("[Error] Technology-Agnostic Memory Compiler (memgen.py) not found in 'memory/' directory.")
        sys.exit(1)

    mem_dir = os.path.dirname(memgen_path)

    # 1. Preset listing
    if getattr(args, "list_presets", False):
        subprocess.run([sys.executable, memgen_path, "--list-presets"], cwd=mem_dir)
        return

    # 2. Natural language prompt parsing via Ollama
    spec_query = getattr(args, "spec", None) or getattr(args, "prompt", None) or getattr(args, "pos_spec", None)
    if spec_query:
        print(f"\n[AI Memory Compiler] Analyzing natural language specification with '{args.model}':")
        print(f"  \"{spec_query}\"\n")

        prompt = f"""You are an expert VLSI memory compiler engineer.
Analyze the following natural language memory specification request and extract the parameters for the SystemVerilog memory compiler (memgen.py).
Return a strict JSON object (and nothing else).

Specification: "{spec_query}"

Allowed JSON fields:
- "type": "rom", "spram", "sdp", or "tdp" (default "spram")
- "size": string capacity like "4KB", "16KB", "64KB", "1MB", "1Kx32", "2Kx64", "512x8" (or null)
- "depth": integer number of words or null
- "width": integer data bit-width (default 32)
- "bus": "none", "axi5", "riscv-ibus", "ahb-lite", or "wishbone" (default "none")
- "byte_enable": boolean (true if byte-write enable requested, else false)
- "name": string valid Verilog module name
- "init_file": string hex filename for ROM or null

JSON response:"""
        llm_reply = call_ollama(prompt, model=args.model)
        m = re.search(r"\{[\s\S]*?\}", llm_reply)
        params = {}
        if m:
            try:
                params = json.loads(m.group(0))
            except Exception as e:
                print(f"[Warning] Could not parse LLM JSON ({e}), raw output: {llm_reply}")

        cmd = [sys.executable, memgen_path]
        if params.get("type"):
            cmd += ["--type", str(params["type"]).lower()]
        if params.get("size"):
            cmd += ["--size", str(params["size"])]
        if params.get("depth"):
            cmd += ["--depth", str(params["depth"])]
        if params.get("width"):
            cmd += ["--width", str(params["width"])]
        if params.get("name"):
            cmd += ["--name", str(params["name"])]
        elif getattr(args, "name", None):
            cmd += ["--name", args.name]
        if params.get("byte_enable"):
            cmd.append("--byte-enable")
        if params.get("bus") and params["bus"] != "none":
            cmd += ["--bus", str(params["bus"]).lower()]
        if params.get("init_file"):
            cmd += ["--init-file", str(params["init_file"])]

        out_dir = getattr(args, "out_dir", None) or "generated_mem"
        cmd += ["--out-dir", out_dir]

        print(f"Executing memgen:")
        print(f"  {' '.join(cmd)}\n")
        res = subprocess.run(cmd, cwd=mem_dir)
        if res.returncode == 0:
            print(f"\n✓ Memory compilation complete! Target SystemVerilog wrappers generated in: {out_dir}")
        return

    # 3. Direct CLI invocation pass-through
    full_args = [sys.executable, memgen_path]
    for k, v in vars(args).items():
        if k in ["command", "model", "spec", "prompt", "list_presets"] or v is None:
            continue
        if isinstance(v, bool):
            if v:
                full_args.append(f"--{k.replace('_', '-')}")
        else:
            full_args.extend([f"--{k.replace('_', '-')}", str(v)])
    full_args.extend(unknown_args)

    subprocess.run(full_args, cwd=mem_dir)


def cmd_update(args=None):
    """Update package, rebuild models from Modelfiles, and run tests."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("==========================================================")
    print("           Updating LLM_Local Environment                 ")
    print("==========================================================")

    # 1. Update Python package installation
    print("\n[1/4] Updating package registration in editable mode...")
    pip_res = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=base_dir, capture_output=True, text=True)
    if pip_res.returncode == 0:
        print("      ✓ Package registration updated (rtl, rtl-assistant, memgen, spec2rtl).")
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--user"], cwd=base_dir, capture_output=True)
        print("      ✓ Package registration updated in user directory.")

    # 2. Rebuild local LLM models from Modelfiles
    print("\n[2/4] Updating local RTL models from Modelfiles...")
    mf_rtl = os.path.join(base_dir, "Modelfile.rtl")
    mf_fast = os.path.join(base_dir, "Modelfile.rtl-fast")
    if os.path.isfile(mf_rtl):
        print("      Building rtl-coder (7B)...")
        res = subprocess.run(["ollama", "create", "rtl-coder", "-f", mf_rtl], cwd=base_dir, capture_output=True, text=True)
        if res.returncode == 0:
            print("      ✓ rtl-coder (7B) rebuilt successfully.")
        else:
            print(f"      ! Note: Ensure Ollama is running (`ollama serve`).")
    if os.path.isfile(mf_fast):
        print("      Building rtl-coder-fast (3B)...")
        res = subprocess.run(["ollama", "create", "rtl-coder-fast", "-f", mf_fast], cwd=base_dir, capture_output=True, text=True)
        if res.returncode == 0:
            print("      ✓ rtl-coder-fast (3B) rebuilt successfully.")

    # 3. Run Memory Compiler unit tests
    print("\n[3/4] Running memory compiler test suite...")
    test_dir = os.path.join(base_dir, "memory", "tests")
    if os.path.isdir(test_dir):
        t_res = subprocess.run([sys.executable, "-m", "unittest", "discover", test_dir], cwd=base_dir, capture_output=True, text=True)
        if t_res.returncode == 0:
            print("      ✓ Memory compiler unit tests passed.")
        else:
            print("      ! Memory compiler tests finished.")

    # 4. Verify health
    print("\n[4/4] Verifying environment health...")
    check_status()
    print("==========================================================")
    print("          LLM_Local Update Complete!                      ")
    print("==========================================================\n")


def cmd_soc(args, unknown_args=None):
    """Run full Spec2RTL + UPF + CDC + SDC pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "spec2rtl_upf"),
        os.path.join(base_dir, "..", "Spec2rtl_upf"),
        os.path.expanduser("~/LLM_Local/spec2rtl_upf"),
        os.path.expanduser("~/Spec2rtl_upf"),
        "/Users/bappi/Spec2rtl_upf"
    ]
    soc_dir = next((c for c in candidates if os.path.isdir(c)), None)
    if not soc_dir:
        print("[Error] Spec2rtl_upf directory not found.")
        sys.exit(1)

    full_args = (getattr(args, "soc_args", []) or []) + (unknown_args or [])
    main_py = os.path.join(soc_dir, "main.py")
    if os.name == "nt":  # Windows
        cmd = ["python", main_py, "--llm"] + full_args
    else:
        run_script = os.path.join(soc_dir, "run_local.sh")
        if os.path.isfile(run_script):
            cmd = [run_script] + full_args
        else:
            cmd = [sys.executable, main_py, "--llm"] + full_args

    print(f"Launching Spec2RTL + AutoUPF pipeline via local LLM: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=soc_dir)


def print_unified_help():
    print("""
================================================================================
  rtl: 1 Unified Command for RTL, VLSI, Memory Compiler & SoC Engineering
================================================================================

USAGE:
  rtl [SWITCH] [OPTIONS...]

SWITCHES (Choose any one):
  --rtl, -r          Generate synthesizable RTL from spec or document
  --tb, -t           Generate self-checking testbench for an RTL file
  --verify, -v       Autonomous verify & self-healing simulation loop (RTL + TB)
  --synth            Synthesize RTL with Open Standard Cell Library (NanGate45 / Sky130)
  --convert, -c      Convert Markdown (.md) to PDF, TXT, HTML, DOCX, RTF, LaTeX, JSON
  --mem, --memory    Technology-Agnostic Memory Compiler (AI spec or direct)
  --soc              Full SoC CAD Flow (RTL + CDC + UPF 2.1 + SDC + Gate Netlist)
  --chat             Start interactive VLSI / RTL Engineer chat session
  --check            Check local Ollama daemon, models, and EDA tooling status
  --update           Update environment, re-link package & rebuild local models

EXAMPLES:
  # 1. Synthesize RTL module from prompt or document:
  rtl --rtl --spec "4-bit priority encoder with active-low enable" -o encoder.v
  rtl --rtl --file spec.docx -o module.v

  # 2. Generate self-checking testbench:
  rtl --tb encoder.v -o encoder_tb.v

  # 3. Simulate & Auto-Fix Errors (Self-Healing Loop):
  rtl --verify encoder.v encoder_tb.v

  # 4. Synthesize with Open Library Cells (Yosys + Liberty Technology Mapping):
  rtl --synth encoder.v --lib nangate45 -o encoder_gate.v
  rtl --synth encoder.v --lib sky130 -o encoder_gate.v

  # 5. Convert Markdown Documents (PDF, TXT, HTML, DOCX, RTF, LaTeX, JSON):
  rtl --convert spec.md -o spec.pdf
  rtl --convert spec.md -o spec.txt
  rtl --convert spec.md --to docx
  rtl --convert spec.md --to all

  # 6. Generate On-Chip Memory (AI Natural Language or Direct Parameters):
  rtl --mem --spec "16KB Simple Dual-Port RAM with 64-bit width and Wishbone bus"
  rtl --mem --type spram --size 4KB --bus axi5 -o generated_mem
  rtl --mem --list-presets

  # 6. Full SoC CAD & Low-Power Intent Pipeline:
  rtl --soc --spec benchmarks/H264D-U1000_cfg_regs_spec.md --rtl-only
  rtl --soc --spec benchmarks/H264D-U1000_cfg_regs_spec.md --all
  rtl --soc --spec market_specs/ --rtl-only

  # 7. Interactive VLSI / Timing / CDC / UPF Chat:
  rtl --chat

  # 8. Environment Health Check:
  rtl --check

  # 9. Update Environment & Rebuild Models:
  rtl --update
================================================================================
""")


def build_subparsers():
    """Build individual subcommand parsers."""
    p_status = argparse.ArgumentParser(prog="rtl --check", description="Check local status")
    p_update = argparse.ArgumentParser(prog="rtl --update", description="Update environment & rebuild models")

    p_gen = argparse.ArgumentParser(prog="rtl --rtl", description="Generate synthesizable RTL")
    p_gen.add_argument("pos_spec", nargs="?", default="", help="Specification string (optional positional)")
    p_gen.add_argument("--spec", "-s", help="Natural language RTL specification")
    p_gen.add_argument("--file", "-f", help="Path to specification file (.md, .txt, .pdf, .docx, .json, .yaml)")
    p_gen.add_argument("--section", help="Extract only specific section from document (optional)")
    p_gen.add_argument("--output", "-o", help="Output Verilog file path (e.g. module.v)")
    p_gen.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")

    p_tb = argparse.ArgumentParser(prog="rtl --tb", description="Generate self-checking testbench")
    p_tb.add_argument("rtl_file", nargs="?", default="", help="Path to RTL Verilog file")
    p_tb.add_argument("--rtl-file", "-r", dest="rtl_file_opt", help="Path to RTL Verilog file")
    p_tb.add_argument("--output", "-o", help="Output testbench file path (e.g. tb_module.v)")
    p_tb.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")

    p_verify = argparse.ArgumentParser(prog="rtl --verify", description="Verify RTL + TB with self-healing")
    p_verify.add_argument("rtl_file", nargs="?", default="", help="Path to RTL Verilog file")
    p_verify.add_argument("tb_file", nargs="?", default="", help="Path to testbench file")
    p_verify.add_argument("--rtl-file", dest="rtl_file_opt", help="Path to RTL Verilog file")
    p_verify.add_argument("--tb-file", dest="tb_file_opt", help="Path to testbench file")
    p_verify.add_argument("--max-retries", type=int, default=3, help="Max self-healing retry iterations")
    p_verify.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")

    p_synth = argparse.ArgumentParser(prog="rtl --synth", description="Synthesize RTL with Open Standard Cell Library (Yosys)")
    p_synth.add_argument("rtl_file", nargs="?", default="", help="Path to RTL Verilog file")
    p_synth.add_argument("--rtl-file", "-r", dest="rtl_file_opt", help="Path to RTL Verilog file")
    p_synth.add_argument("--top", "-t", help="Top module name")
    p_synth.add_argument("--lib", "-l", choices=["nangate45", "sky130"], default="nangate45", help="Target open standard cell library (default: nangate45)")
    p_synth.add_argument("--output", "-o", help="Output netlist Verilog file (e.g. module_gate.v)")

    p_chat = argparse.ArgumentParser(prog="rtl --chat", description="Interactive VLSI chat")
    p_chat.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")

    p_mem = argparse.ArgumentParser(prog="rtl --mem", description="Technology-Agnostic Memory Compiler")
    p_mem.add_argument("pos_spec", nargs="?", default="", help="Memory specification (optional positional)")
    p_mem.add_argument("--spec", "-s", help="Natural language memory specification (AI-parsed)")
    p_mem.add_argument("--prompt", help="Alias for --spec")
    p_mem.add_argument("--type", "-t", choices=["rom", "spram", "sdp", "tdp"], help="Memory type (default: spram)")
    p_mem.add_argument("--size", help="Capacity string (e.g. 4KB, 16KB, 1Kx32, 2Kx64, 512x8, 36Kb)")
    p_mem.add_argument("--depth", "-d", type=int, help="Explicit depth in words")
    p_mem.add_argument("--width", "-w", type=int, help="Explicit data width in bits (default: 32)")
    p_mem.add_argument("--name", "-n", type=str, help="Top module name")
    p_mem.add_argument("--bus", choices=["none", "axi5", "riscv-ibus", "ahb-lite", "wishbone"], help="Bus interface")
    p_mem.add_argument("--byte-enable", "-b", action="store_true", help="Enable byte-write enable")
    p_mem.add_argument("--init-file", "-i", type=str, help="Path to hex init file for ROM")
    p_mem.add_argument("--out-dir", "-o", type=str, help="Output directory (default: generated_mem)")
    p_mem.add_argument("--list-presets", action="store_true", help="List universal memory size presets")
    p_mem.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")

    p_convert = argparse.ArgumentParser(prog="rtl --convert", description="Convert Markdown (.md) to PDF, TXT, HTML, DOCX, RTF, LaTeX, or JSON")
    p_convert.add_argument("input_file", nargs="?", default="", help="Path to input Markdown (.md) file")
    p_convert.add_argument("--file", "-f", dest="input_file_opt", help="Path to input Markdown file")
    p_convert.add_argument("--output", "-o", help="Path to output file (e.g. spec.pdf, notes.txt, doc.docx)")
    p_convert.add_argument("--to", "-t", choices=["pdf", "txt", "html", "docx", "rtf", "tex", "json", "all"], help="Target format extension (default: inferred from -o or pdf)")

    p_soc = argparse.ArgumentParser(prog="rtl --soc", description="Full Spec2RTL + UPF + CDC + SDC pipeline")
    p_soc.add_argument("soc_args", nargs=argparse.REMAINDER, help="Arguments passed to Spec2rtl_upf main.py")

    return {
        "check": p_status,
        "update": p_update,
        "generate": p_gen,
        "testbench": p_tb,
        "verify": p_verify,
        "synth": p_synth,
        "convert": p_convert,
        "chat": p_chat,
        "memory": p_mem,
        "soc": p_soc,
    }


def main():
    raw_args = sys.argv[1:]

    if not raw_args or raw_args in [["-h"], ["--help"]]:
        print_unified_help()
        sys.exit(0)

    parsers = build_subparsers()
    KNOWN_COMMANDS = set(parsers.keys())

    SWITCH_MAP = {
        "--rtl": "generate",
        "-r": "generate",
        "--gen": "generate",
        "--generate": "generate",
        "--synth": "synth",
        "--synthesize": "synth",
        "--convert": "convert",
        "-c": "convert",
        "--conv": "convert",
        "--tb": "testbench",
        "-t": "testbench",
        "--testbench": "testbench",
        "--verify": "verify",
        "-v": "verify",
        "--sim": "verify",
        "--mem": "memory",
        "--memory": "memory",
        "--soc": "soc",
        "--chat": "chat",
        "--check": "check",
        "--status": "check",
        "--update": "update",
    }

    # 1. Parse --mode <mode> if passed
    mode_target = None
    filtered_args = []
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg == "--mode" and i + 1 < len(raw_args):
            mval = raw_args[i + 1].lower()
            if mval in ["rtl", "gen", "generate"]:
                mode_target = "generate"
            elif mval in ["synth", "synthesize"]:
                mode_target = "synth"
            elif mval in ["convert", "conv"]:
                mode_target = "convert"
            elif mval in ["tb", "testbench"]:
                mode_target = "testbench"
            elif mval in ["verify", "sim"]:
                mode_target = "verify"
            elif mval in ["mem", "memory"]:
                mode_target = "memory"
            elif mval in ["soc"]:
                mode_target = "soc"
            elif mval in ["chat"]:
                mode_target = "chat"
            elif mval in ["check", "status"]:
                mode_target = "check"
            elif mval in ["update"]:
                mode_target = "update"
            i += 2
            continue
        filtered_args.append(arg)
        i += 1

    # 2. Extract direct switch flag (e.g. --rtl, --mem, --tb, etc.)
    target_cmd = mode_target
    dispatch_args = []
    for arg in filtered_args:
        if not target_cmd and arg in SWITCH_MAP:
            target_cmd = SWITCH_MAP[arg]
        else:
            dispatch_args.append(arg)

    # 3. Check for traditional subcommands (generate, memory, chat, etc.)
    if not target_cmd and dispatch_args and dispatch_args[0] in KNOWN_COMMANDS:
        target_cmd = dispatch_args[0]
        dispatch_args = dispatch_args[1:]

    # 4. Intelligent Auto-Detection Heuristic
    if not target_cmd:
        # Synthesis flags
        if any(a in dispatch_args for a in ["--synth", "--synthesize"]):
            target_cmd = "synth"
        # Convert flags
        elif any(a in dispatch_args for a in ["--convert", "--conv"]):
            target_cmd = "convert"
        elif any(a.endswith(".md") for a in dispatch_args if not a.startswith("-")) and any(a in dispatch_args for a in ["--to", "-t"]):
            target_cmd = "convert"
        # Memory flags
        elif any(a in dispatch_args for a in ["--list-presets", "--bus", "--type", "--depth", "--width", "--byte-enable"]):
            target_cmd = "memory"
        # SoC flags
        elif any(a in dispatch_args for a in ["--all", "--rtl-only", "--config", "--filelist", "--upf-only"]):
            target_cmd = "soc"
        # Two Verilog files -> verify
        elif len([a for a in dispatch_args if a.endswith((".v", ".sv")) and not a.startswith("-")]) == 2:
            target_cmd = "verify"
        # One Verilog file without spec -> testbench
        elif len([a for a in dispatch_args if a.endswith((".v", ".sv")) and not a.startswith("-")]) == 1 and not any(a in dispatch_args for a in ["--spec", "-s", "--file", "-f", "-o", "--output"]):
            target_cmd = "testbench"
        # Spec or file provided -> default to RTL generation
        elif any(a in dispatch_args for a in ["--spec", "-s", "--file", "-f"]):
            target_cmd = "generate"
        else:
            print_unified_help()
            sys.exit(1)

    target_parser = parsers[target_cmd]
    sub_args, sub_unknown = target_parser.parse_known_args(dispatch_args)

    if target_cmd == "check":
        check_status()
    elif target_cmd == "update":
        cmd_update(sub_args)
    elif target_cmd == "generate":
        cmd_generate(sub_args)
    elif target_cmd == "testbench":
        cmd_testbench(sub_args)
    elif target_cmd == "verify":
        cmd_verify(sub_args)
    elif target_cmd == "synth":
        cmd_synth(sub_args)
    elif target_cmd == "convert":
        cmd_convert(sub_args)
    elif target_cmd == "memory":
        cmd_memory(sub_args, sub_unknown)
    elif target_cmd == "chat":
        cmd_chat(sub_args)
    elif target_cmd == "soc":
        cmd_soc(sub_args, sub_unknown)


if __name__ == "__main__":
    main()
