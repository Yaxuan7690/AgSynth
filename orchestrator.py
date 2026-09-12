"""
orchestrator.py
Orchestrator for multi-agent collaboration (4+1 Architecture)
"""

import ast
import base64
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

from agents import (
    ConceptDesignerAgent,
    PainterAgent,
    QAGeneratorAgent,
    VisualSpecifierAgent,
)
from audit_agent import FullProcessAuditorAgent


class Orchestrator:
    """
    Orchestrator - Coordinate 5 agents to generate questions
    """

    def __init__(self):
        """Initialize the five generation and quality-control agents."""
        self.concept_designer = ConceptDesignerAgent()
        self.visual_specifier = VisualSpecifierAgent()
        self.painter = PainterAgent()
        self.qa_generator = QAGeneratorAgent()
        self.auditor = FullProcessAuditorAgent()

    def execute_python_code(self, code: str, save_path: str = None) -> None:
        """
        Execute Python code to generate image

        Args:
            code: Python code string
            save_path: Image save path

        Raises:
            Exception: If code execution fails
        """
        if not save_path:
            raise ValueError("A destination path is required for rendered images.")

        if not isinstance(code, str) or not code.strip():
            raise ValueError("Generated painter code is empty.")

        self._validate_generated_code(code)

        output_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(output_dir, exist_ok=True)

        encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
        runner = (
            "import base64; "
            f"code=base64.b64decode({encoded_code!r}).decode('utf-8'); "
            f"save_path={save_path!r}; "
            "exec(compile(code, '<generated_painter>', 'exec'), "
            "{'__name__': '__generated_painter__', 'save_path': save_path})"
        )

        try:
            with tempfile.TemporaryDirectory(prefix="agsynth-mpl-") as mpl_config_dir:
                render_env = {
                    key: value
                    for key, value in os.environ.items()
                    if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"}
                }
                render_env["MPLBACKEND"] = "Agg"

                render_env["MPLCONFIGDIR"] = mpl_config_dir
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", runner],
                    capture_output=True,
                    text=True,
                    timeout=45,
                    check=False,
                    cwd=output_dir,
                    env=render_env,
                )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Code execution timed out after 45 seconds.") from e

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown execution error").strip()
            raise RuntimeError(f"Code execution failed: {detail[-800:]}\n\nCode:\n{code[:500]}...")

    @staticmethod
    def _validate_generated_code(code: str) -> None:
        """Reject non-plotting imports and dangerous operations before execution."""
        try:
            tree = ast.parse(code, filename="<generated_painter>")
        except SyntaxError as e:
            raise ValueError(f"Generated code has invalid Python syntax: {e}") from e

        allowed_roots = {"matplotlib", "numpy", "math", "random", "re", "collections"}
        blocked_names = {
            "eval",
            "exec",
            "compile",
            "open",
            "input",
            "__import__",
            "globals",
            "locals",
            "vars",
            "getattr",
            "setattr",
            "delattr",
        }
        blocked_attributes = {
            "system",
            "popen",
            "remove",
            "unlink",
            "rmdir",
            "rename",
            "replace",
            "run",
            "Popen",
            "call",
            "check_call",
            "check_output",
            "connect",
            "send",
            "urlopen",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                modules = []

            for module in modules:
                root = module.split(".", 1)[0]
                if root not in allowed_roots:
                    raise ValueError(f"Generated code imports a forbidden module: {module}")

            if isinstance(node, ast.Name) and node.id in blocked_names:
                raise ValueError(f"Generated code uses a forbidden operation: {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in blocked_attributes:
                raise ValueError(f"Generated code uses a forbidden operation: {node.attr}")

            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "savefig"
            ):
                uses_destination = (
                    bool(node.args)
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "save_path"
                ) or any(
                    keyword.arg in {"fname", "filename"}
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "save_path"
                    for keyword in node.keywords
                )
                if not uses_destination:
                    raise ValueError("Generated code must save the image to save_path only.")

    @staticmethod
    def _rule_based_concept_check(concept: dict, difficulty: str) -> Optional[str]:
        """Concept design basic check

        Returns:
        - None: Passed
        - str: Error message
        """
        required_fields = [
            "core_logic_kernel",
            "difficulty_requirements",
            "concept_design",
            "expected_answer",
        ]

        for field in required_fields:
            if field not in concept or not concept[field]:
                return f"Missing required field: {field}"

        if len(str(concept.get("core_logic_kernel", ""))) < 20:
            return "core_logic_kernel too short (< 20 chars)"

        concept_design = concept.get("concept_design", {})
        if not isinstance(concept_design, dict):
            return "concept_design must be a dict"

        required_sub_fields = ["logic_description", "data_specifications"]
        for field in required_sub_fields:
            if field not in concept_design or not concept_design[field]:
                return f"concept_design missing field: {field}"

            if len(str(concept_design[field])) < 10:
                return f"concept_design.{field} too short (< 10 chars)"

        return None

    @staticmethod
    def _validate_data_consistency(concept_design: dict, visual_spec: dict) -> Optional[str]:
        """Validate data consistency between concept and visual spec

        Returns:
        - None: Passed
        - str: Error message
        """
        if not isinstance(concept_design, dict) or not isinstance(visual_spec, str):
            return "Concept design and visual specification must use the expected types."

        concept_block = concept_design.get("concept_design", {})
        if not isinstance(concept_block, dict):
            return "concept_design must be a dictionary."

        spec_lower = visual_spec.lower()
        locked_fields = concept_design.get("locked_data_fields", {})
        if locked_fields and not isinstance(locked_fields, dict):
            return "locked_data_fields must be a dictionary."

        for field_name, value in (locked_fields or {}).items():
            value_text = str(value).strip()
            if value_text and value_text.lower() not in spec_lower:
                return (
                    f"Visual specification is missing locked value for {field_name}: {value_text}"
                )

        source_text = " ".join(
            str(concept_block.get(name, ""))
            for name in ("data_specifications", "constraints", "logic_description")
        )
        numeric_values = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", source_text)
        missing_values = [
            value for value in dict.fromkeys(numeric_values) if value not in spec_lower
        ]
        if missing_values:
            return "Visual specification is missing concept data values: " + ", ".join(
                missing_values[:8]
            )

        return None

    @staticmethod
    def _rule_based_visual_check(visual_spec) -> Optional[str]:
        """Visual specification basic check (Natural language format)

        Returns:
        - None: Passed
        - str: Error message
        """
        if not visual_spec or not isinstance(visual_spec, str):
            return "visual_spec must be a non-empty string"

        if len(visual_spec) < 100:
            return f"visual_spec too short (< 100 chars, got {len(visual_spec)}). Must include positions, colors, and sizes."
        if len(visual_spec) > 1500:
            return f"visual_spec exceeds the 1500-character paper limit (got {len(visual_spec)})."

        desc_lower = visual_spec.lower()
        essential_keywords = [
            ("position", "coordinate", "at (", "at(", "x", "y"),
            ("color", "red", "blue", "green", "#", "rgb"),
        ]

        missing_info = []
        for keywords in essential_keywords:
            if not any(kw in desc_lower for kw in keywords):
                missing_info.append(keywords[0])

        if missing_info:
            return f"visual_spec missing essential info: {', '.join(missing_info)}. Must specify positions and colors."

        return None

    @staticmethod
    def _rule_based_code_check(code: str) -> Optional[str]:
        """Internal component of the AgSynth generation pipeline."""
        if not code or len(code.strip()) < 50:
            return "Code too short (< 50 chars). Ensure complete drawing code was generated."

        if "plt.savefig" not in code:
            return "Missing plt.savefig() statement. Add plt.savefig(save_path) at the end."

        if "plt.close" not in code:
            return "Missing plt.close() statement. Call plt.close() after saving the image."

        if "if __name__" in code:
            return "Code contains an if __name__ block and will not execute through exec(). Remove the guard and write drawing code directly."

        return None

    @staticmethod
    def _rule_based_qa_check(qa: dict, difficulty: str) -> Optional[str]:
        """QA basic check

        Returns:
        - None: Passed
        - str: Error message
        """
        question = qa.get("question_text", "")
        answer = qa.get("answer_text", "")
        solution_steps = qa.get("solution_steps", [])

        if not question or len(question) < 50:
            return "Question text empty or too short (< 50 chars)"

        if not answer or len(answer) < 1:
            return "Answer text empty"

        if not isinstance(solution_steps, list) or len(solution_steps) < 2:
            return "solution_steps must contain at least two steps"
        if any(not isinstance(step, str) or not step.strip() for step in solution_steps):
            return "solution_steps must be a list of non-empty strings"

        return None

    def run_with_retry(self, stage_name: str, func, max_attempts: int = 3, *args, **kwargs) -> Any:
        """Run a pipeline stage with feedback-driven retries.

        Args:
            stage_name: Name shown in retry logs
            func: Function to execute
            max_attempts: Maximum number of attempts
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result

        Raises:
            Exception: If all attempts fail
        """
        last_error = None
        feedback = ""

        for attempt in range(1, max_attempts + 1):
            try:
                result = func(*args, feedback=feedback, **kwargs)

                return result

            except Exception as e:
                last_error = e
                error_msg = str(e)
                feedback = error_msg

                if attempt < max_attempts:
                    print(f"{stage_name} attempt {attempt} failed: {error_msg}")
                    print(f"Retrying with feedback ({attempt + 1}/{max_attempts})...")
                else:
                    raise Exception(
                        f"{stage_name} failed after {max_attempts} attempts: {error_msg}"
                    )

        raise last_error

    @staticmethod
    def _ensure_image_ok(image_path: str) -> None:
        """Ensure image file exists and is valid"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        file_size = os.path.getsize(image_path)
        if file_size < 1024:
            raise ValueError(f"Image file too small ({file_size} bytes): {image_path}")

    def run(
        self,
        seed_questions: List[Dict[str, Any]],
        output_image_dir: str,
        image_filename: str,
        difficulty: str,
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        print(f"\nStarting generation: {image_filename}")
        print(f"   Difficulty: {difficulty}")
        print(f"   Seed count: {len(seed_questions)}")

        seed_image_paths = [
            seed.get("image", "")
            for seed in seed_questions
            if isinstance(seed, dict) and seed.get("image")
        ]
        missing_seed_images = [path for path in seed_image_paths if not os.path.isfile(path)]
        if not seed_image_paths or missing_seed_images:
            raise FileNotFoundError(
                f"Seed image input is missing: {missing_seed_images or 'no image path provided'}"
            )

        def concept_design_with_check(feedback: str = ""):
            concept_design = self.concept_designer.generate(
                seed_questions=seed_questions,
                difficulty=difficulty,
                feedback=feedback,
            )

            error = self._rule_based_concept_check(concept_design, difficulty)
            if error:
                raise ValueError(f"Concept check failed: {error}")

            audit_pass, audit_feedback = self.auditor.audit_stage1_concept(
                concept_design=concept_design,
                seed_questions=seed_questions,
                difficulty=difficulty,
            )
            if not audit_pass:
                raise ValueError(f"Concept audit failed:\n{audit_feedback}")

            return concept_design

        concept_design = self.run_with_retry(
            stage_name="ConceptDesigner", func=concept_design_with_check, max_attempts=3
        )

        print("Stage 1 complete: Concept Design")

        print("Stage 1 audit passed")

        def visual_spec_with_check(feedback: str = ""):
            visual_spec = self.visual_specifier.generate(
                seed_questions=seed_questions,
                concept_design=concept_design,
                difficulty=difficulty,
                feedback=feedback,
            )

            error = self._rule_based_visual_check(visual_spec)
            if error:
                raise ValueError(f"Visual spec check failed: {error}")

            consistency_error = self._validate_data_consistency(concept_design, visual_spec)
            if consistency_error:
                raise ValueError(f"Data consistency check failed: {consistency_error}")

            audit_pass, audit_feedback = self.auditor.audit_stage2_visual(
                visual_spec=visual_spec,
                concept_design=concept_design,
                seed_image_paths=seed_image_paths,
            )
            if not audit_pass:
                raise ValueError(f"Visual audit failed:\n{audit_feedback}")

            return visual_spec

        visual_spec = self.run_with_retry(
            stage_name="VisualSpecifier", func=visual_spec_with_check, max_attempts=3
        )

        print("Stage 2 complete: Visual Specification")

        print("Stage 2 audit passed")

        previous_code = ""

        def paint_with_check(feedback: str = ""):
            nonlocal previous_code

            python_code = self.painter.generate_code(
                visual_spec=visual_spec,
                concept_design=concept_design,
                feedback=feedback,
                previous_code=previous_code,
            )
            previous_code = python_code

            code_error = self._rule_based_code_check(python_code)
            if code_error:
                raise ValueError(f"Code check failed: {code_error}")

            image_path = os.path.join(output_image_dir, image_filename)
            self.execute_python_code(python_code, save_path=image_path)

            self._ensure_image_ok(image_path)

            audit_pass, audit_feedback = self.auditor.audit_stage3_image(
                image_path=image_path,
                visual_spec=visual_spec,
                python_code=python_code,
            )
            if not audit_pass:
                raise ValueError(f"Image quality audit failed:\n{audit_feedback}")

            return image_path, python_code

        image_path, python_code = self.run_with_retry(
            stage_name="Painter", func=paint_with_check, max_attempts=3
        )

        print(f"Stage 3 complete: Painter (image saved to {image_filename})")

        print("Stage 3 audit passed: image quality is acceptable")

        def qa_with_check(feedback: str = ""):
            qa = self.qa_generator.generate(
                concept_design=concept_design,
                visual_spec=visual_spec,
                image_path=image_path,
                difficulty=difficulty,
                feedback=feedback,
            )

            qa_error = self._rule_based_qa_check(qa, difficulty)
            if qa_error:
                raise ValueError(f"QA check failed: {qa_error}")

            audit_pass, audit_feedback = self.auditor.audit_stage4_qa(
                qa_result=qa,
                image_path=image_path,
                concept_design=concept_design,
                visual_spec=visual_spec,
            )
            if not audit_pass:
                raise ValueError(f"QA audit failed:\n{audit_feedback}")

            return qa

        qa_result = self.run_with_retry(
            stage_name="QA_Generator", func=qa_with_check, max_attempts=3
        )

        print("Stage 4 complete: QA Generation")

        print("Stage 4 audit passed: text-image consistency and image quality are acceptable")

        from utils import (
            MASKING_MAX_RETRIES,
            rewrite_question_for_figure_dependency,
            validate_masked_question,
        )

        original_question_text = qa_result.get("question_text", "")
        print("Stage 4.5: applying text masking...")

        masking_success = False
        for masking_attempt in range(1, MASKING_MAX_RETRIES + 1):
            try:
                rewrite_result = rewrite_question_for_figure_dependency(
                    original_question_text, image_path
                )

                if rewrite_result["already_figure_dependent"]:
                    print("Stage 4.5: question already depends on the image")
                    masking_success = True
                    break

                rewritten_question = rewrite_result["rewritten_question"]
                if not rewritten_question.strip():
                    print(f"Stage 4.5 attempt {masking_attempt}: empty rewrite, retrying...")
                    continue

                masking_image_composition = (
                    visual_spec if isinstance(visual_spec, str) else str(visual_spec)
                )

                is_valid, reason = validate_masked_question(
                    original_question=original_question_text,
                    rewritten_question=rewritten_question,
                    image_composition=masking_image_composition,
                    image_path=image_path,
                )

                if is_valid:
                    if "question_text" in qa_result:
                        qa_result["question_text"] = rewritten_question
                    elif "question" in qa_result:
                        qa_result["question"] = rewritten_question
                    print(f"Stage 4.5: text masking passed (attempt {masking_attempt})")
                    masking_success = True
                    break
                else:
                    print(f"Stage 4.5 attempt {masking_attempt} failed validation: {reason}")

            except Exception as masking_error:
                print(f"Stage 4.5 attempt {masking_attempt} raised an error: {masking_error}")

        if not masking_success:
            raise ValueError(
                f"Figure-dependency check failed after {MASKING_MAX_RETRIES} attempts."
            )

        final_verification = self.auditor.dual_model_final_gate(
            concept_design=concept_design,
            visual_spec=visual_spec,
            qa_result=qa_result,
            image_path=image_path,
            difficulty=difficulty,
        )
        if not final_verification.get("pass_check", False):
            raise ValueError(
                "Terminal dual-model verification failed: "
                f"{final_verification.get('feedback', 'at least one verifier rejected the sample')}"
            )

        return {
            "design_detail": {"concept_design": concept_design, "image_design": visual_spec},
            "qa_result": qa_result,
            "final_verification": final_verification,
        }
