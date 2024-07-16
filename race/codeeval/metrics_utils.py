import numpy as np
import importlib.util
import os
import ast
import inspect
import re
from radon.metrics import mi_visit
from race.codeeval.human_eval.execution import swallow_io


class LCOMCalculator:
    def __init__(self, cls):
        self.cls = cls
        self.methods = [getattr(cls, func) for func in dir(cls) if callable(getattr(cls, func)) and not func.startswith("__")]
        self.instance_variables = self._get_instance_variables()
        self.access_matrix = self._build_access_matrix()

    def _get_instance_variables(self):
        # Try to create an instance of the class to inspect instance variables
        instance = self._create_instance()
        if instance is None:
            return []
        return [var for var in dir(instance) if not callable(getattr(instance, var)) and not var.startswith("__")]

    def _create_instance(self):
        try:
            # Get the constructor parameters
            signature = inspect.signature(self.cls)
            parameters = signature.parameters
            
            # Prepare arguments with default or placeholder values
            args = {}
            for name, param in parameters.items():
                if param.default != inspect.Parameter.empty:
                    args[name] = param.default
                elif param.annotation != inspect.Parameter.empty:
                    args[name] = self._get_default_value_for_type(param.annotation)
                else:
                    args[name] = None  # Fallback to None if no type or default is provided
            
            # Create an instance using the prepared arguments
            return self.cls(**args)
        except Exception as e:
            print(f"Failed to create an instance of {self.cls.__name__}: {e}")
            return None
    
    def _get_default_value_for_type(self, annotation):
        # Define default values for common types
        if annotation in {int, float}:
            return 0
        elif annotation == str:
            return ""
        elif annotation == bool:
            return False
        elif annotation in {list, set, tuple}:
            return type(annotation)()
        elif annotation == dict:
            return {}
        else:
            return None

    def _build_access_matrix(self):
        access_matrix = []
        for method in self.methods:
            method_vars = []
            for var in self.instance_variables:
                if self._method_accesses_variable(method, var):
                    method_vars.append(1)
                else:
                    method_vars.append(0)
            access_matrix.append(method_vars)
        return access_matrix

    def _method_accesses_variable(self, method, var):
        # Check if the method's code object references the variable
        code = method.__code__
        return var in code.co_names

    def calculate_lcom(self):
        method_count = len(self.methods)
        variable_count = len(self.instance_variables)
        if method_count == 0 or variable_count == 0:
            return 0

        access_sum = sum(sum(row) for row in self.access_matrix)
        lcom = 1 - (access_sum / (method_count * variable_count))
        return lcom
    

class VariableNameExtractor(ast.NodeVisitor):
    def __init__(self):
        self.variable_names = set()

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variable_names.add(target.id)
            elif isinstance(target, ast.Tuple):
                for elem in target.elts:
                    if isinstance(elem, ast.Name):
                        self.variable_names.add(elem.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            self.variable_names.add(arg.arg)
        self.generic_visit(node)

    def visit_For(self, node):
        if isinstance(node.target, ast.Name):
            self.variable_names.add(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elem in node.target.elts:
                if isinstance(elem, ast.Name):
                    self.variable_names.add(elem.id)
        self.generic_visit(node)

    def extract(self, tree):
        self.visit(tree)
        return self.variable_names

def extract_variable_names(code):
    try:
        code_bytes = bytes(code, "utf8")
        tree = ast.parse(code_bytes)
    except SyntaxError:
        return []

    extractor = VariableNameExtractor()
    return extractor.extract(tree)


def extract_class_names(code):
    try:
        code_bytes = bytes(code, "utf8")
        tree = ast.parse(code_bytes)
    except SyntaxError:
        return []

    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    return class_names


def extract_function_names(code):
    try:
        code_bytes = bytes(code, "utf8")
        tree = ast.parse(code_bytes)
    except SyntaxError:
        return []

    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)

    return names


def extract_function_bodies(code):
    try:
        code_bytes = bytes(code, "utf8")
        tree = ast.parse(code_bytes)
    except SyntaxError:
        return []

    function_bodies = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            function_body = ast.get_source_segment(code, node)

            function_bodies.append((function_name, function_body))

    return function_bodies


# readibility
def metrics_for_readability_camel(code):
    pattern = re.compile(r'^[a-z]+(?:[A-Z][a-z]*)*$')

    names = extract_function_names(code)
    names += extract_variable_names(code)
    if len(names) == 0:
        return False
    for name in names:
        if not bool(pattern.match(name)):
            return False
        
    return True


def metrics_for_readability_snake(code):
    pattern = re.compile(r'^[a-z]+(?:_[a-z]+)*$')

    names = extract_function_names(code)
    names += extract_variable_names(code)
    if len(names) == 0:
        return False
    for name in names:
        if not bool(pattern.match(name)):
            return False
    
    return True


def metrics_for_readability_function_camel(code):
    pattern = re.compile(r'^[a-z]+(?:[A-Z][a-z]*)*$')

    names = extract_function_names(code)
    if len(names) == 0:
        return False
    for name in names:
        if not bool(pattern.match(name)):
            return False
        
    return True


def metrics_for_readability_function_snake(code):
    pattern = re.compile(r'^[a-z]+(?:_[a-z]+)*$')

    names = extract_function_names(code)
    if len(names) == 0:
        return False
    for name in names:
        if not bool(pattern.match(name)):
            return False
    
    return True


def metrics_for_readability_var_camel(code):
    pattern = re.compile(r'^[a-z]+(?:[A-Z][a-z]*)*$')

    names = extract_variable_names(code)
    if len(names) == 0:
        return False
    for name in names:
        if not bool(pattern.match(name)):
            return False
        
    return True


def metrics_for_readability_var_snake(code):
    pattern = re.compile(r'^[a-z]+(?:_[a-z]+)*$')

    names = extract_variable_names(code)
    if len(names) == 0:
        return False
    for name in names:
        if not bool(pattern.match(name)):
            return False
    
    return True


def metrics_for_readability_length(code, max_line_length=100, max_function_length=40):
    splitted_code = code.split('\n')
    for line in splitted_code:
        if len(line) > max_line_length:
            return False
    
    functions = extract_function_bodies(code)
    if len(functions) == 0:
        return False
    for _, function_body in functions:
        if len(function_body.split('\n')) > max_function_length:
            return False
        
    return True


def metrics_for_readability_comment_by_function(code):
    try:
        code_bytes = bytes(code, "utf8")
        tree = ast.parse(code_bytes)
    except SyntaxError:
        return False

    if isinstance(tree, ast.Module) and tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
        return True

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str)):
                return False

    return True


def metrics_for_readability_comment_by_line(code, threshold=0.75): 
    def backup_metrics_for_readability_comment_by_line(code):
        def has_comment(line):
            line = line.strip()
            if not line:
                return True
            if line.startswith('#'):
                return True
            if '#' in line:
                code_part = line.split('#')[0].strip()
                if code_part:
                    return True
            return False

        lines = code.split('\n')
        lines_without_comments = [line for line in lines if not has_comment(line)]
        if len(lines_without_comments) / len(lines) < 1-threshold:
            return True

        return False

    def merge_lines(code):
        class LineMerger(ast.NodeVisitor):
            def __init__(self):
                self.line_map = {}
                self.current_line = 1

            def visit(self, node):
                if hasattr(node, 'lineno'):
                    self.line_map[node.lineno] = self.current_line
                self.generic_visit(node)

            def generic_visit(self, node):
                if hasattr(node, 'lineno'):
                    self.current_line = node.lineno
                super().generic_visit(node)

        class LineMergerTransformer(ast.NodeTransformer):
            def __init__(self, line_map):
                self.line_map = line_map

            def visit(self, node):
                if hasattr(node, 'lineno'):
                    node.lineno = self.line_map.get(node.lineno, node.lineno)
                return super().visit(node)

        tree = ast.parse(code)
        line_merger = LineMerger()
        line_merger.visit(tree)

        transformer = LineMergerTransformer(line_merger.line_map)
        new_tree = transformer.visit(tree)

        return ast.fix_missing_locations(new_tree)
    
    try:
        comment_line_cnt = len(re.findall(r'\s#+\s', code))
        match = re.search(r'""".*?"""', code, flags=re.DOTALL)
        if match:
            comment_line_cnt += 1
            
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL).strip()
        
        merged_code = merge_lines(code)
        merged_code = ast.unparse(merged_code)
        
        total_line_cnt = len(merged_code.split('\n'))
        
        if comment_line_cnt / total_line_cnt >= threshold:
            return True

        return False
    except:
        return backup_metrics_for_readability_comment_by_line(code)


def metrics_for_readability_arg_count(code):
    class FunctionArgCounter(ast.NodeVisitor):
        def __init__(self):
            self.function_args = {}
        
        def visit_FunctionDef(self, node):
            arg_count = len(node.args.args)
            self.function_args[node.name] = arg_count
            self.generic_visit(node)
    
    try:
        code_bytes = bytes(code, "utf8")
        tree = ast.parse(code_bytes)
    except SyntaxError:
        return False

    arg_counter = FunctionArgCounter()
    arg_counter.visit(tree)

    return arg_counter.function_args


# maintainability
def metrics_for_maintainability_cohesion(code):
    local_scope = {}
    with swallow_io():
        exec(code, {}, local_scope)  # TODO, unsafe
    
    class_names = extract_class_names(code)
    lcom_results = {}
    
    for class_name in class_names:
        cls = local_scope[class_name]
        lcom_calculator = LCOMCalculator(cls)
        lcom_results[class_name] = lcom_calculator.calculate_lcom()
    
    if len(lcom_results.values()) == 0:
        return -1
    else:
        return round(sum(lcom_results.values()) / len(lcom_results.values()), 2)


def metrics_for_maintainability_loop(code, loop_type='for'):
    class LoopCounter(ast.NodeVisitor):
        def __init__(self):
            self.while_count = 0
            self.for_count = 0
        
        def visit_While(self, node):
            self.while_count += 1
            self.generic_visit(node)
        
        def visit_For(self, node):
            self.for_count += 1
            self.generic_visit(node)
    
    try:
        code_bytes = bytes(code, "utf8")
        tree = ast.parse(code_bytes)
    except SyntaxError:
        return False

    loop_counter = LoopCounter()
    loop_counter.visit(tree)
    
    if loop_type == 'for':
        return loop_counter.while_count == 0
    elif loop_type == 'while':
        return loop_counter.for_count == 0


def metrics_for_maintainability_mi(code):
    return round(mi_visit(code, True), 2)


def metrics_for_maintainability_module_count(code):
    return len(extract_function_names(code))


###########################################################################################


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def compute_metrics_from_results(results, k_list=[1, 5]):
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist()
        for k in ks
        if (total >= k).all()
    }
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }
    detail_metrics = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k