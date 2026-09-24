# implement code quality metric for LLMs
import json
import pickle
from radon.metrics import h_visit, mi_visit
from radon.visitors import ComplexityVisitor
import ast
import tempfile
import sys
import os, signal, time, subprocess, psutil
import shutil
import math
import re
from pathlib import Path
import psutil
import tokenize
import io
from collections import defaultdict, Counter
import keyword

_KILL_SIG = signal.SIGKILL if os.name != "nt" else signal.SIGTERM  

TEMP_DIR='./temp'


# ----------No need for pip----------
BUILTIN_PKGS = {
    'sys', 'os', 'math', 'json', 'datetime', 'time', 'itertools', 'collections',
    'random', 're', 'typing', 'pathlib', 'subprocess', 'logging', 'functools',
    'statistics', 'shutil', 'dataclasses', 'decimal', 'fractions', 'heapq', 'string'
}

def split_list_to_dict(src, n = 5):
    total = len(src)
    base  = total // n
    extra = total % n

    parts, start = {}, 0
    for idx in range(0, n):
        end = start + base + (1 if idx <= extra else 0)
        parts[idx] = src[start:end]
        start = end
    return parts

def _kill_proc_tree(root: psutil.Process, sig=_KILL_SIG, timeout=3):
    """
    kill root and subprocess, similar to `kill -9 -- -pgid`
    """
    children = root.children(recursive=True)
    for p in children:
        try:
            p.send_signal(sig)
        except psutil.NoSuchProcess:
            pass
    root.send_signal(sig)
    psutil.wait_procs(children + [root], timeout=timeout)

def kill_tree_pid(pid: int, sig=_KILL_SIG, timeout: float = 3.0):
    """kill pid and subprocess"""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for p in children:
        try:
            p.send_signal(sig)
        except psutil.NoSuchProcess:
            pass
    try:
        parent.send_signal(sig)
    except psutil.NoSuchProcess:
        pass
    psutil.wait_procs(children + [parent], timeout=timeout)


def _py_of_env(env_dir):
    """Return Path to python executable inside the given venv directory."""
    env_dir = Path(env_dir)
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return env_dir / sub

def _sanitize_nonascii_bytes(src: str) -> str:
    _NON_ASCII_BYTES_RE = re.compile(
        r"""b(['"])([^'"\n]*?[^\x00-\x7f][^'"\n]*?)\1""",
        flags=re.ASCII,
    )
    def repl(m):
        quote = m.group(1)
        body  = m.group(2)
        return f"{quote}{body}{quote}"
    return _NON_ASCII_BYTES_RE.sub(repl, src)

def safe_tokenize(code):
    """tokenize the given code"""
    tokens = []
    try:
        tokens_iter = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in tokens_iter:
            tokens.append(token)
    except (tokenize.TokenError, IndentationError) as e:
        # tokenize error, run line by line
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            try:
                line_tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
                for token in line_tokens:
                    if token.type != tokenize.ENDMARKER:
                        new_token = tokenize.TokenInfo(
                            token.type, token.string, 
                            (line_num, token.start[1]), 
                            (line_num, token.end[1]), 
                            line
                        )
                        tokens.append(new_token)
            except:
                words = re.findall(r'\b\w+\b|[^\w\s]', line)
                for word in words:
                    tokens.append(tokenize.TokenInfo(
                        tokenize.NAME if word.isidentifier() else tokenize.OP,
                        word, (line_num, 0), (line_num, len(word)), line
                    ))
    return tokens

# ---------- whether the target lib is in stdlib (can use BUILTIN_PKGS = set(sys.stdlib_module_names) in Python 3.10) ----------
def missing_modules(pkgs,venvs_dir_P):
    python_bin = venvs_dir_P / ("Scripts" if os.name == "nt" else "bin") / "python"
    check_code = r"""
import importlib.util, json, sys
mods = json.loads(sys.stdin.read())
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print(json.dumps(missing))
    """

    proc = subprocess.run(
        [str(python_bin), "-c", check_code],
        input=json.dumps(list(pkgs)),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"python subprocess failed:\n{proc.stderr}")
        return pkgs
    missing=set(json.loads(proc.stdout))
    return missing

def detect_imports(py_file,installed=set(),venvs_dir_P=None, package_name_path='./ppl_utils/tmp_lib_name.pkl'):
    """extract the `import` from the pyfile and return all packages need to be install"""
    # tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    raw = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(raw, filename=str(py_file))
    except SyntaxError as e:
        if "bytes can only contain" not in str(e):
            raise
        cleaned = _sanitize_nonascii_bytes(raw)
        tree = ast.parse(cleaned, filename=str(py_file))
    pkgs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pkgs.add(alias.name.split('.')[0].lower())
        elif isinstance(node, ast.ImportFrom) and node.module:
            pkgs.add(node.module.split('.')[0].lower())
    # pkgs={'yaml'}#'smtplib','numpy',
    third_party=missing_modules(pkgs,venvs_dir_P)

    with open(package_name_path, 'rb') as f:
        package_name_dict = pickle.load(f)

    new_packages = {package_name_dict['import2pip'].get(item, item) for item in third_party}
    return {p for p in new_packages if p not in installed}


def ensure_base_venv(base_venv_dir):
    """
    if no base environment, then create one
    """
    if os.path.exists(base_venv_dir):
        return base_venv_dir
    subprocess.check_call([sys.executable, "-m", "venv", "--copies", str(base_venv_dir)])

    py = Path(base_venv_dir) / ("Scripts" if os.name == "nt" else "bin") / "python"
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(py), "-m", "pip", "install", "numpy", "pandas"])
    return base_venv_dir

def new_venv(venv_dir,base_venv_dir):
    # subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    # py = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    # subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    base_dir = ensure_base_venv(base_venv_dir)
    if venv_dir.exists():
        # shutil.rmtree(venv_dir) 
        pass
    else:
        shutil.copytree(base_dir, venv_dir, symlinks=True)
    py_P = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    return py_P


def pip_install(python, packages):
    """install packages in the virtual environment"""
    if packages:
        subprocess.check_call([str(python), "-m", "pip", "install", *sorted(packages)])


def installed_in_venv(venv_dir):
    """
    return the installed libs in `venv_dir`
    """
    if venv_dir is None:
        return set()

    python_bin = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    if not os.path.exists(str(python_bin)):
        return set()
    try:
        output = subprocess.check_output(
            [str(python_bin), "-m", "pip", "list", "--format=json"],
            text=True
        )
        pkgs_json = json.loads(output)
        return {pkg["name"].lower() for pkg in pkgs_json}
    except subprocess.CalledProcessError as e:
        return set()

def extract_keywords(scenario_path,service_path):
    # add lib names manually
    # service_list=list(service_keywords.keys())
    # with open('./tmp_name', "w") as f:
    #     for line in service_list:
    #         f.write(f"{line}\n")
    # with open('tmp_name', encoding="utf-8") as f1, \
    #      open('tmp_lib_name', encoding="utf-8") as f2:
    #     keys   = [line.rstrip("\n\r") for line in f1]
    #     values = [line.rstrip("\n\r") for line in f2]
    # tmp_dict=dict(zip(keys, values))
    # for key,value in service_keywords.items():
    #     if tmp_dict[key] not in value:
    #         service_keywords[key].append(tmp_dict[key])
    if os.path.exists(service_path):
        with open(service_path, 'rb') as f:#input,bug type,params
            service_dict = pickle.load(f)
    else:
        with open(scenario_path, 'rb') as f:#input,bug type,params
            scenario_dict = pickle.load(f)
        service_dict={}
        for scenario in scenario_dict.keys():
            for service in scenario_dict[scenario]['Service'].keys():
                # if service=='Seldon Core':
                #     print(1)
                if service not in service_dict.keys():
                    service_dict[service]=[]
                if service.lower() not in service_dict[service]:
                    service_dict[service].append(service.lower())
                pattern =  re.compile(r"github\.com/[^/\s'\"“”]+/([A-Za-z0-9]+)(?=[/\-\?\'\"“”#\s]|$)",
                flags=re.IGNORECASE)
                # re.compile(
                # r'https?://github\.com/[^/\s]+/'
                # r'([A-Za-z0-9_.-]+)'
                # r'(?=[/\s\'"【，]|$)')
                url_keyword=pattern.findall(scenario_dict[scenario]['Service'][service]['URL'])
                for ukey in url_keyword:
                    if ukey.lower() not in service_dict[service]:
                        service_dict[service].append(ukey.lower())
                if '-' in service:
                    for _tmp in service.split('-'):
                        if _tmp.lower() not in service_dict[service]:
                            service_dict[service].append(_tmp.lower())
                elif ' ' in service:
                    for _tmp in service.split(' '):
                        if _tmp.lower() not in service_dict[service]:
                            service_dict[service].append(_tmp.lower())
                elif '.' in service:
                    for _tmp in service.split('.')[:-1]:
                        if _tmp.lower() not in service_dict[service]:
                            service_dict[service].append(_tmp.lower())
        with open(service_path, 'wb') as f:
            pickle.dump(service_dict, f)
    return service_dict


# --------------- Generatability ---------------
def contain_or_not(input_code,keywords_list):
    # fsrc = "\n".join(input_code)
    if isinstance(input_code, list):
        input_code = "\n".join(input_code)
    for service_keywords in keywords_list:
        if service_keywords in input_code:
            return 1
    return 0

def calculate_contain_ratio(response_list,service,keywords_pkl):
    if isinstance(keywords_pkl,str):
        with open(keywords_pkl, 'rb') as f:#input,bug type,params
            service_keywords_dict = pickle.load(f)
    else:
        service_keywords_dict=keywords_pkl
    service_keywords=service_keywords_dict[service]
    contain_sum=0
    for response in response_list:
        code_list=response.split('\n')[1:-2]# remove the start and the end
        contain_sum+=contain_or_not(code_list,service_keywords)
    return contain_sum/len(response_list)


# --------------- Maintainability ---------------
def _estimate_halstead(code):
    """Plan B for halstead_volum"""
    operators = set()
    operands = set()
    operator_count = defaultdict(int)
    operand_count = defaultdict(int)
    
    python_operators = {
        '+', '-', '*', '/', '//', '%', '**',
        '=', '+=', '-=', '*=', '/=', '//=', '%=', '**=',
        '==', '!=', '<', '>', '<=', '>=',
        'and', 'or', 'not', 'in', 'is',
        '&', '|', '^', '~', '<<', '>>',
        '(', ')', '[', ']', '{', '}',
        ',', ':', ';', '.', '->', 
        'if', 'elif', 'else', 'for', 'while', 'def', 'class',
        'try', 'except', 'finally', 'with', 'as', 'import', 'from',
        'return', 'yield', 'break', 'continue', 'pass', 'lambda'
    }
    
    tokens = safe_tokenize(code)
    
    for token in tokens:
        if token.type == tokenize.ENDMARKER:
            continue
            
        token_str = token.string.strip()
        if not token_str:
            continue
            
        if (token.type == tokenize.OP or 
            token_str in python_operators or
            keyword.iskeyword(token_str)):
            
            operators.add(token_str)
            operator_count[token_str] += 1
        elif token.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
            if not keyword.iskeyword(token_str):
                operands.add(token_str)
                operand_count[token_str] += 1
    

    n1 = len(operators)
    n2 = len(operands)
    N1 = sum(operator_count.values())
    N2 = sum(operand_count.values())
    
    if n1 == 0 and n2 == 0:
        return 1.0
    
    vocabulary = n1 + n2
    length = N1 + N2
    
    if vocabulary <= 0:
        return 1.0
    
    volume = length * math.log2(vocabulary) if vocabulary > 1 else length
    return max(1.0, volume)


def calculate_halstead_volume(input_code):
    """
    Calculate Halstead Volume.
    """
    # src = "\n".join(input_code)
    if isinstance(input_code, list):
        src = "\n".join(input_code)
    else:
        src = input_code
    hv_log=[]
    try:
        report = h_visit(src)
        return report.total.volume,hv_log
    except Exception as e:
        # print(e)
        hv_log.append(str(e))

    # Plan B: estimate
    # try:
    return _estimate_halstead(src),hv_log
    # except Exception as e:
    #     # print(e)
    #     hv_log.append(str(e))
    #     return None,hv_log

def _estimate_complexity_ast(code):
    """Plan B: estimate with AST"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        if "bytes can only contain" not in str(e):
            raise
        cleaned = _sanitize_nonascii_bytes(code)
        tree = ast.parse(cleaned)
    try:
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.Lambda):
                complexity += 1
            elif isinstance(node, ast.ListComp):
                complexity += 1
            elif isinstance(node, ast.DictComp):
                complexity += 1
            elif isinstance(node, ast.SetComp):
                complexity += 1
            elif isinstance(node, ast.GeneratorExp):
                complexity += 1
        
        return max(1, complexity)
    except SyntaxError:
        raise  

def _estimate_complexity_re(src):
    """
    Plan C: re estimate
    """
    complexity = 1
    
    basic_patterns = [
        r'\bif\b', r'\belif\b', r'\bfor\b', r'\bwhile\b',
        r'\band\b', r'\bor\b', r'\btry\b', r'\bexcept\b'
    ]
    
    for pattern in basic_patterns:
        matches = re.findall(pattern, src, re.IGNORECASE)
        complexity += len(matches)
    
    return max(1, min(complexity, 50))# limit the max value

def calculate_cyclomatic_complexity(input_code):
    """
    Calculate Cyclomatic Complexity (total across functions/classes).
    """
    # src = "\n".join(input_code)
    if isinstance(input_code, list):
        src = "\n".join(input_code)
    else:
        src = input_code
    cc_log=[]
    try:
        visitor = ComplexityVisitor.from_code(src)
        total_cc = sum(block.complexity for block in visitor.blocks)
        return max(1, total_cc),cc_log
    except Exception as e:
        # print(e)
        cc_log.append(str(e))

    # Plan B: Estimate
    try:
        estimate_cc=_estimate_complexity_ast(src)
        return estimate_cc,cc_log
    except Exception as e:
        # print(e)
        cc_log.append(str(e))
        estimate_cc=_estimate_complexity_re(src)
        return estimate_cc,cc_log

def _calculate_mi_manually(code,halstead_volume,cyclomatic_complexity):
    """
    MI = MAX(0,(171 - 5.2 * ln(Halstead Volume) - 0.23 * (Cyclomatic Complexity) - 16.2 * ln(Lines of Code))*100 / 171)
    
    Ref:
    Microsoft Learn - Code metrics - Maintainability index range and meaning
    https://learn.microsoft.com/en-us/visualstudio/code-quality/code-metrics-maintainability-index-range-and-meaning?view=vs-2022
    """

    if halstead_volume==None:
        halstead_volume = calculate_halstead_volume(code)
    if cyclomatic_complexity==None:
        cyclomatic_complexity = calculate_cyclomatic_complexity(code)
    lines_of_code = _count_lines_of_code(code)
    
    if halstead_volume==None or cyclomatic_complexity==None:
        return None
    if halstead_volume <= 0:
        halstead_volume = 1
    if lines_of_code <= 0:
        lines_of_code = 1
    
    try:
        raw_mi = (171 - 
                 5.2 * math.log(halstead_volume) - 
                 0.23 * cyclomatic_complexity - 
                 16.2 * math.log(lines_of_code))
        mi = max(0, (raw_mi * 100) / 171)
        return max(0.0, min(100.0, mi))
        
    except Exception as e:
        print(e)
        return None


def _count_lines_of_code(code):
    """Count the code lines"""
    lines = code.split('\n')
    loc = 0
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            loc += 1
    
    return max(1, loc)

def calculate_maintainability_index(input_code,hv=None,cc=None):
    """
    Calculate Maintainability Index using Halstead V, Cyclomatic C, and SLOC/comments.
    https://radon.readthedocs.io/en/latest/api.html
    """
    # src = "\n".join(input_code)
    src = input_code
    mi_log=[]
    try:
        mi_score = mi_visit(src, True)
        return mi_score,mi_log
    except Exception as e:
        # print(e)
        mi_log.append(str(e))
    
    # Plan B: Manually
    # try:
    return _calculate_mi_manually(src,hv,cc),mi_log
    # except Exception as e:
    #     mi_log.append(str(e))
    #     return None,mi_log


# --------------- Readability ---------------
# TODO: validate
def linter_readability_score(input_code, code_path=None, pylint_binary = "pylint", timeout=30):
    """
    Uses Pylint's global evaluation score (0-10) as a proxy for readability:
    higher → more readable.  Returns a float in [0, 1].
    """
    if isinstance(input_code, list):
        input_code = "\n".join(input_code)
    else:
        input_code = input_code

    # write snippet to a temporary file
    if code_path!=None:
        tmp_path=code_path
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=TEMP_DIR, delete=False) as tf:
            tf.write(input_code)
            tmp_path = tf.name
    try:
        # run pylint quietly, keep the global score line
        proc = subprocess.run(
            [pylint_binary, "--score=y", "-sn", "--reports=n", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log=proc.stdout + proc.stderr
        # search for “rated at x/10”
        m = re.search(r"rated at\s+([0-9.]+)/10", proc.stdout)
        raw = float(m.group(1)) if m else 0.0
        return max(0.0, min(1.0, raw / 10.0)),log
    except Exception as e:
        print(e)
        return None,e
    # finally:
    #     os.remove(tmp_path)

def calculate_posnett_readability(input_code):
    """
    Compute the readability probability defined by Posnett-Hindle-Devanbu (MSR 2011):
        z = 8.87 - 0.033 * V + 0.40 * L - 1.5 * H
        score = 1 / (1 + e^(-z))
    where
        V = Halstead Volume
        L = number of non-empty lines
        H = byte-level Shannon entropy
    Returns a float in [0,1] (higher = more readable).
    """
    # normalise input
    if isinstance(input_code, list):
        code_str = "\n".join(input_code)
    else:
        code_str = input_code

    # 1) Lines of code (non-blank)
    lines = [ln for ln in code_str.splitlines() if ln.strip()]
    L = len(lines)

    # 2) Halstead Volume V
    try:
        V = h_visit(code_str).total.volume
    except Exception:
        V = 0.0  # fallback on parse failure

    # 3) Byte-level entropy H
    byte_counts = Counter(code_str.encode())
    total = sum(byte_counts.values())
    H = 0.0
    if total:
        H = -sum((c / total) * math.log2(c / total) for c in byte_counts.values())

    # logistic model
    z = 8.87 - 0.033 * V + 0.40 * L - 1.5 * H
    return 1.0 / (1.0 + math.exp(-z))
