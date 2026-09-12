"""
agents.py
Define agent responsibilities and prompts - 4+1 Architecture
"""

import re

import matplotlib

matplotlib.use("Agg")

from typing import Any, Dict, List

from utils import AGENT_MODELS, chat_completion, multi_vision_completion, vision_completion


def _level_desc(difficulty: str) -> str:
    """Return description based on difficulty with clear distinction"""
    if str(difficulty).lower() in {"easy", "simple"}:
        return "1-10 reasoning steps, straightforward logic"
    else:
        return "3-20 reasoning steps, multi-layered logic requiring deeper analysis"


def _ensure_dict(x: Any, ctx: str = "") -> Dict[str, Any]:
    """Ensure x is a dict, otherwise raise error"""
    if not isinstance(x, dict):
        raise ValueError(f"{ctx}: Expected dict, got {type(x).__name__}")
    return x


def _clip(text: str, max_len: int = 8192) -> str:
    """Clip text to max_len characters"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


class ConceptDesignerAgent:
    """
    Concept Designer Agent - Focus on Logic Kernel Extraction
    Core Mission: Deeply understand seed question, extract logic kernel, design new concept
    """

    def __init__(self):
        """Initialize the concept designer."""

    def generate(
        self, seed_questions: List[Dict[str, Any]], difficulty: str, feedback: str = ""
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        sys_prompt = (
            f"You are Agent 1 (Concept Designer). Difficulty: [{difficulty}] ({_level_desc(difficulty)}).\n\n"
            "## 📥 Your Input\n"
            "- Seed question image\n"
            "- Seed question JSONL data (question, answer, solution steps, etc.)\n\n"
            "## 🎯 Your Mission (2 Stages)\n"
            "**Stage 1: Understand Seed Question Kernel**\n"
            "- Deeply analyze the seed question's mathematical logic\n"
            "- Extract the core logic kernel (NOT just copy the question)\n"
            "- Identify: quantity relations, spatial relations, logic relations, patterns, constraints\n\n"
            "**Stage 2: Design New Question Concept (Overall Framework)**\n"
            "- Design a NEW question concept using the SAME logic kernel\n"
            "- Provide concrete values and data specifications\n"
            "- Focus on OVERALL FRAMEWORK, NOT visual details (Agent 2 will handle that)\n"
            "- Your output = Blueprint for Agent 2 to design visuals\n"
            "- 🎯 **Encouraged (Not Mandatory)**: Apply the logic to a DIFFERENT scenario/context to increase diversity\n"
            "  - Example: If seed uses fruits → try sports/vehicles/time/money/geometry\n"
            "  - Example: If seed uses animals → try tools/buildings/weather/colors\n"
            "  - Keep the SAME mathematical logic, just change the scenario wrapper\n\n"
            "## 📤 Your Output (JSON)\n"
            "```json\n"
            "{\n"
            '  "core_logic_kernel": "[REQUIRED] Core mathematical logic (min 20 chars)",\n'
            '  "difficulty_requirements": "[REQUIRED] Difficulty description",\n'
            '  "concept_design": {\n'
            '    "logic_description": "[REQUIRED] Specific logic with concrete values",\n'
            '    "data_specifications": "[REQUIRED] Concrete data values",\n'
            '    "constraints": "Specific constraints",\n'
            '    "context_setting": "Concrete scenario"\n'
            "  },\n"
            '  "locked_data_fields": {"field_name": "value that must appear in the figure"},\n'
            '  "expected_answer": "[REQUIRED] Concrete answer",\n'
            '  "key_reasoning_steps": ["step1", "step2"],\n'
            '  "design_rationale": "Why this design"\n'
            "}\n"
            "```\n\n"
            "## ⚠️ What You DON'T Do\n"
            "❌ Design visual details (colors, layout, coordinates) - Agent 2 will do this\n"
            "❌ Generate Python code - Agent 3 will do this\n"
            "❌ Write question text - Agent 4 will do this\n\n"
            "## ✅ Critical Requirements\n"
            "1. **Be SPECIFIC**: Use concrete values (e.g., '3 apples', '5 paths', '8kg')\n"
            "2. **Be DETAILED**: Provide enough details for Agent 2 to design visuals\n"
            "3. **No Coordinates**: Do NOT specify positions (Agent 2 handles that)\n"
            "4. **Concrete Answer**: Must be a specific value (e.g., 'B', '12', 'Red circle')\n\n"
            "**All content in English.**"
        )

        if feedback:
            sys_prompt += (
                f"\n\n## ⚠️ Feedback from Previous Attempt\n"
                f"{feedback}\n\n"
                f"**Please carefully read the feedback and address the issues!**\n"
            )

        text_descriptions: List[str] = []
        image_paths: List[str] = []

        for q in seed_questions:
            if not isinstance(q, dict):
                raise ValueError(
                    f"ConceptDesignerAgent: seed_questions element must be dict, got {type(q).__name__}"
                )

            qid = q.get("id", "?")
            question = q.get("question", "")
            answer = q.get("answer", "")
            image_path = q.get("image", "")
            image_composition = q.get("image_composition", "") or q.get("Image composition", "")
            solution_steps = q.get("solution_steps", "")

            desc = f"【Seed Question {qid}】\n"
            desc += f"- Question: {_clip(question, 1500)}\n"
            desc += f"- Answer: {_clip(answer, 800)}\n"
            desc += f"- Solution Steps: {_clip(solution_steps, 2000)}\n" if solution_steps else ""
            if image_composition:
                desc += (
                    f"- Image Composition (Optional Reference): {_clip(image_composition, 1500)}\n"
                )

            text_descriptions.append(desc)
            image_paths.append(image_path)

        user_prompt = (
            f"Please deeply analyze the following seed question and extract its logic kernel:\n\n"
            f"{text_descriptions[0]}\n\n"
            "## 🎯 Your Tasks\n\n"
            "1. **Understand** the seed question thoroughly\n"
            "2. **Extract** the core logic kernel\n"
            "3. **Design** a new concept that uses the same logic\n\n"
            "Return your analysis in the specified JSON format."
        )

        result = multi_vision_completion(
            system_prompt=sys_prompt,
            image_paths=image_paths,
            user_prompt=user_prompt,
            timeout=120.0,
            temperature=1.0,
            model=AGENT_MODELS.get("concept_designer"),
        )
        concept_design = _ensure_dict(result, "ConceptDesignerAgent")
        return concept_design


class VisualSpecifierAgent:
    """
    Visual Specification Agent - Focus on Visual Form Design
    Core Mission: Design innovative visual presentation and communicate with Painter
    """

    def __init__(self):
        """Initialize the visual specifier."""

    def generate(
        self,
        seed_questions: List[Dict[str, Any]],
        concept_design: Dict[str, Any],
        difficulty: str,
        feedback: str = "",
    ) -> str:
        """Internal component of the AgSynth generation pipeline."""
        sys_prompt = (
            f"You are Agent 2 (Visual Specifier). Difficulty: [{difficulty}] ({_level_desc(difficulty)}).\n\n"
            "## 🎯 Your Task\n"
            "Design a COMPLETE visual blueprint for Agent 3 (Painter) to code.\n"
            "- Input: Agent 1's concept design + seed image (reference only)\n"
            "- Output: Concise natural language description of image design (PURE TEXT, NO JSON)\n\n"
            "## 📤 Output Format (Natural Language)\n"
            "Write a concise, structured natural language description that includes:\n\n"
            "**Visual Form**: [Brief name, e.g., 'Number Grid', 'Balance Scale']\n\n"
            "**Canvas**: 12.0 x 12.0 (origin at bottom-left)\n\n"
            "**Image Design**:\n"
            "Describe ALL visual elements with EXACT specifications:\n"
            "- Shape type (circle/rectangle/line/text/etc.)\n"
            "- EXACT position (x, y coordinates)\n"
            "- SPECIFIC color (hex code or color name)\n"
            "- Size/dimensions (radius/width/height)\n"
            "- Any labels/text with their positions\n"
            "- Layout arrangement and spacing\n"
            "- For 3D geometry/shapes, EXPLICITLY specify which lines are solid and which are dashed (hidden lines)\n\n"
            "Example:\n"
            "\"Draw a red circle (radius 0.8) at position (3.5, 7.2) with label 'A' at (3.5, 7.5). "
            "Draw a blue rectangle (2.0x1.5) at bottom-left corner (1.0, 2.0) with number '12' centered at (2.0, 2.75). "
            'Use 1.0 margin and 0.5 spacing between elements."\n\n'
            "**Data Values**: [Key data points, e.g., 'A=5, B=7, Answer=12']\n\n"
            "**Expected Answer**: [Concrete answer]\n\n"
            "## ✅ Critical Requirements\n"
            "1. **PURE NATURAL LANGUAGE**: NO JSON format, NO code blocks, ONLY plain text\n"
            "2. **MANDATORY DETAILS**: MUST specify for EVERY element:\n"
            "   - Shape type, EXACT position (x, y), SPECIFIC color, Size/dimensions, Labels/text positions\n"
            "3. **CONCISE**: NO reasoning, NO explanations, ONLY design specs\n"
            "4. **COMPLETE**: Describe ALL visual elements in the image\n"
            "5. **PURE VISUAL**: NO question text, NO options (A-E) in image\n"
            "6. **INNOVATIVE**: Choose DIFFERENT visual form from seed image\n"
            "7. 🎯 **Encouraged (Scenario Transformation)**: Apply the logic to a DIFFERENT scenario/context\n"
            "   - Example: If seed is about fruits → try sports/vehicles/time/money/animals\n"
            "   - Keep the SAME mathematical logic, but change the scenario setting\n"
            "   - This is encouraged but NOT mandatory\n\n"
            "**Keep description under 1500 chars to avoid truncation.**\n"
            "**All content in English.**"
        )

        if feedback:
            sys_prompt += (
                f"\n\n## ⚠️ Feedback from Previous Attempt\n"
                f"{feedback}\n\n"
                f"**Please address the issues!**\n"
            )

        text_descriptions: List[str] = []
        image_paths: List[str] = []

        for q in seed_questions:
            if not isinstance(q, dict):
                raise ValueError(
                    f"VisualSpecifierAgent: seed_questions element must be dict, got {type(q).__name__}"
                )

            qid = q.get("id", "?")
            question = q.get("question", "")
            image_path = q.get("image", "")
            image_composition = q.get("image_composition", "") or q.get("Image composition", "")

            desc = f"【Seed Question {qid} - Visual Reference】\n"
            desc += f"- Question: {_clip(question, 1000)}\n"
            if image_composition:
                desc += f"- Image Composition: {_clip(image_composition, 1500)}\n"

            text_descriptions.append(desc)
            image_paths.append(image_path)

        user_prompt = (
            f"{text_descriptions[0]}\n\n"
            f"## 🎯 Concept Design from ConceptDesigner\n\n"
            f"{_clip(str(concept_design), 4000)}\n\n"
            "Based on the concept design, create a detailed visual specification.\n"
            "Remember: Include all 5 essential elements (Graphic Elements, Layout, Coordinates, Colors, Annotations)."
        )

        result = multi_vision_completion(
            system_prompt=sys_prompt,
            image_paths=image_paths,
            user_prompt=user_prompt,
            timeout=120.0,
            temperature=1.0,
            json_mode=False,
            model=AGENT_MODELS.get("visual_specifier"),
        )

        if isinstance(result, dict):
            visual_spec_text = result.get("text", "") or result.get("content", "") or str(result)
        else:
            visual_spec_text = str(result)

        return visual_spec_text


class PainterAgent:
    """
    Painter Agent - Data Visualization Expert
    Core Mission: Generate Python code to create visualization using matplotlib
    """

    def __init__(self):
        """Initialize the painter."""

    def generate_code(
        self,
        visual_spec: str,
        concept_design: Dict[str, Any],
        feedback: str = "",
        previous_code: str = "",
    ) -> str:
        """
        Generate Python code based on visual specification

        Args:
            visual_spec: Visual specification from VisualSpecifier (natural language string)
            concept_design: Concept design from ConceptDesigner
            feedback: Feedback from previous attempt (error message)
            previous_code: Previous attempt's code (for retry context)

        Returns:
            Python code string (NOT JSON)
        """
        if not isinstance(visual_spec, str):
            raise ValueError(
                f"PainterAgent: visual_spec must be str, got {type(visual_spec).__name__}"
            )

        difficulty = concept_design.get("difficulty_requirements", "easy")

        sys_prompt = (
            f"You are Agent 3 (Painter). Difficulty: [{difficulty}].\n\n"
            "## 📥 Your Input\n"
            "- Agent 2's detailed image specification (design blueprint)\n\n"
            "## 🎯 Your Mission (2 Stages)\n"
            "**Stage 1: Understand Agent 2's Design**\n"
            "- Read Agent 2's visual specification carefully\n"
            "- Understand: graphic elements, layout, coordinates, colors, annotations\n"
            "- Identify all elements that need to be drawn\n\n"
            "**Stage 2: Generate Python Code to Draw Image**\n"
            "- Write COMPLETE Python code using matplotlib\n"
            "- Draw EXACTLY what Agent 2 specified\n"
            "- Your output = Executable Python code (NOT JSON)\n\n"
            "## 📤 Your Output\n"
            "Pure Python code (NO JSON wrapper, NO markdown fences, NO comments)\n\n"
            "## ⚠️ What You DON'T Do\n"
            "❌ Design visuals yourself (Agent 2 already did this)\n"
            "❌ Write question text (Agent 4 will do this)\n"
            "❌ Add comments in code (system will reject)\n\n"
            "**CRITICAL REQUIREMENTS** (Failure = Immediate Rejection):\n"
            "1. **MUST Include All Import Statements**: Write ALL required imports at the top\n"
            "   - ✅ REQUIRED: import matplotlib.pyplot as plt\n"
            "   - ✅ REQUIRED: import matplotlib.patches as patches (if using Rectangle, Circle, Wedge, etc.)\n"
            "   - ✅ REQUIRED: import numpy as np (if using numpy functions)\n"
            "   - ❌ NEVER call Rectangle() directly, ALWAYS use patches.Rectangle()\n"
            "   - ❌ NEVER call Circle() directly, ALWAYS use patches.Circle()\n"
            "   - ❌ NEVER call Wedge() directly, ALWAYS use patches.Wedge()\n\n"
            "2. **No Undefined Variables**: NEVER use variables before defining them\n"
            "   - ❌ BAD: `result = code_result + 1` (code_result not defined)\n"
            "   - ✅ GOOD: `code_result = 10; result = code_result + 1`\n\n"
            "3. **Balanced Parentheses**: MUST have equal number of ( and )\n"
            "   - Count carefully: plt.plot((x, y)) has 2 left, 2 right\n"
            "   - Use proper nesting: patches.Rectangle((x, y), width, height)\n"
            "   - 🚨 CRITICAL: Double-check EVERY function call has matching parentheses\n"
            "   - 🚨 CRITICAL: Count ( and ) before submitting - they MUST be equal\n\n"
            "4. **Correct Function Prefixes**: ALWAYS use proper module prefixes\n"
            "   - ✅ CORRECT: patches.Rectangle(...), patches.Circle(...), patches.Wedge(...)\n"
            "   - ✅ CORRECT: plt.plot(...), plt.scatter(...), plt.text(...)\n"
            "   - ✅ CORRECT: np.arange(...), np.linspace(...), np.array(...)\n"
            "   - ❌ WRONG: Rectangle(...), Circle(...), Wedge(...) without prefix\n\n"
            "5. **DO NOT include plt.savefig() or plt.close()**: The system will auto-add them\n"
            "   - ❌ BAD: plt.savefig(save_path)\n"
            "   - ✅ GOOD: Just write your plotting code\n\n"
            "6. **🚨 ABSOLUTELY NO COMMENTS**: Do NOT write ANY comments in the code\n"
            "   - ❌ FORBIDDEN: # This is a comment\n"
            "   - ❌ FORBIDDEN: # Create figure\n"
            "   - ❌ FORBIDDEN: # Draw circles\n"
            "   - ✅ CORRECT: Just write pure executable code without any comments\n"
            "   - 🚨 CRITICAL: Write concise, clean code without verbose comments\n\n"
            "7. **Forbidden Patterns**:\n"
            "   - No backslash line continuation (\\)\n"
            "   - No if __name__ == '__main__' block\n"
            "   - No undefined function calls (e.g., draw_fl() without definition)\n"
            "   - 🚨 NO COMMENTS of any kind (# ...)\n\n"
            "8. **Code Efficiency & Style Requirements**:\n"
            "   - If drawing complex shapes, use loops or numpy arrays instead of hard-coding every single coordinate if possible\n"
            "   - Keep the plotting code concise\n"
            "   - Break down complex drawing tasks into small functions\n"
            "   - Example: Use `for i in range(n):` instead of repeating similar code 10+ times\n\n"
            "**VALIDATION CHECKLIST** (Check before submitting):\n"
            "✅ All import statements are at the top\n"
            "✅ All matplotlib.patches classes use patches. prefix\n"
            "✅ All variables are defined before use\n"
            "✅ All functions are defined before calling\n"
            "✅ NO plt.savefig() or plt.close() (system will add them)\n"
            "✅ 🚨 ABSOLUTELY NO COMMENTS (no # anywhere in code)\n"
            "✅ Use loops/arrays for repetitive patterns (avoid hard-coding 10+ similar lines)\n\n"
            "**CODE STYLE**: Write concise, clean code. Break down complex tasks into small functions. Do NOT add verbose comments or explanations.\n\n"
            "**IMPORTANT**: Return ONLY the Python code, NO JSON wrapper, NO markdown fences, NO COMMENTS.\n"
            "Just pure executable Python code with ALL imports and correct prefixes."
        )

        if feedback and previous_code:
            sys_prompt += (
                f"\n\n## ⚠️ Previous Attempt Failed\n\n"
                f"**Previous Code**:\n"
                f"```python\n"
                f"{_clip(previous_code, 3000)}\n"
                f"```\n\n"
                f"**Error Message**:\n"
                f"{feedback}\n\n"
                f"**Your Task**: Fix the error and generate corrected code.\n"
            )
        elif feedback:
            sys_prompt += (
                f"\n\n## ⚠️ Feedback from Previous Attempt\n{feedback}\n\n**Fix the issues!**\n"
            )

        user_prompt = (
            f"## 🎨 Visual Specification\n\n"
            f"{_clip(str(visual_spec), 6000)}\n\n"
            "Generate COMPLETE Python code to create this visualization.\n\n"
            "**CRITICAL REMINDERS**:\n"
            "1. Include ALL import statements at the top\n"
            "2. Use correct prefixes: patches.Rectangle(), patches.Circle(), patches.Wedge()\n"
            "3. DO NOT include plt.savefig() or plt.close() (system will add them)\n\n"
            "**Return ONLY the Python code, nothing else.**"
        )

        result = chat_completion(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            json_mode=False,
            timeout=120.0,
            temperature=0.2,
            max_tokens=65536,
            model=AGENT_MODELS.get("painter"),
        )

        if isinstance(result, dict):
            python_code = result.get("code") or result.get("python_code") or str(result)
        else:
            python_code = str(result)

        python_code = python_code.strip()
        if python_code.startswith("```python"):
            python_code = python_code[9:]
        if python_code.startswith("```"):
            python_code = python_code[3:]
        if python_code.endswith("```"):
            python_code = python_code[:-3]
        python_code = python_code.strip()

        python_code = self._ensure_required_statements(python_code)

        return python_code

    def _fix_line_continuation_errors(self, code: str) -> str:
        """Fix line continuation errors in generated code"""

        return code

    def _ensure_required_statements(self, code: str) -> str:
        """Internal component of the AgSynth generation pipeline."""
        code_lower = code.lower()

        has_matplotlib = any(
            pattern in code_lower
            for pattern in ["plt.", "pyplot.", "matplotlib", "fig", "ax", "plot", "scatter", "bar"]
        )

        if not has_matplotlib:
            return code

        standard_header = """# ========== Required standard imports ==========
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Wedge, Circle, Rectangle, Polygon, Ellipse, Arc, FancyBboxPatch, FancyArrowPatch, Arrow, PathPatch
import numpy as np
import re
# ========== End standard imports ==========

"""

        import_patterns = [
            r"^\s*import\s+matplotlib\.pyplot\s+as\s+plt\s*$",
            r"^\s*import\s+matplotlib\.patches\s+as\s+patches\s*$",
            r"^\s*from\s+matplotlib\s+import\s+pyplot\s+as\s+plt\s*$",
            r"^\s*from\s+matplotlib\s+import\s+patches\s*$",
            r"^\s*from\s+matplotlib\.patches\s+import\s+.*$",
            r"^\s*import\s+numpy\s+as\s+np\s*$",
            r"^\s*import\s+numpy\s*$",
            r"^\s*import\s+re\s*$",
        ]

        cleaned_lines = []
        for line in code.split("\n"):
            is_duplicate_import = any(re.match(pattern, line) for pattern in import_patterns)
            if not is_duplicate_import:
                cleaned_lines.append(line)

        cleaned_code = "\n".join(cleaned_lines)

        patches_classes = [
            "Rectangle",
            "Circle",
            "Ellipse",
            "Polygon",
            "Wedge",
            "Arc",
            "FancyBboxPatch",
            "FancyArrowPatch",
            "Arrow",
            "PathPatch",
        ]
        for cls in patches_classes:
            pattern = rf"(?<!patches\.)(?<!\.)\b{cls}\s*\("
            replacement = rf"patches.{cls}("
            cleaned_code = re.sub(pattern, replacement, cleaned_code)

        standard_footer = """
# ========== Required image save ==========
plt.tight_layout()
plt.savefig(save_path, dpi=100, bbox_inches='tight')
plt.close()
# ========== End image save ==========
"""

        has_tight_layout = "plt.tight_layout" in code_lower or "tight_layout()" in code_lower
        has_savefig = "plt.savefig" in code_lower or "savefig(" in code_lower
        has_close = "plt.close" in code_lower or ".close()" in code_lower

        if has_tight_layout or has_savefig or has_close:
            save_patterns = [
                r"^\s*plt\.tight_layout\(\)\s*$",
                r"^\s*plt\.savefig\(.*\)\s*$",
                r"^\s*plt\.close\(\)\s*$",
            ]
            final_lines = []
            for line in cleaned_code.split("\n"):
                is_save_statement = any(re.match(pattern, line) for pattern in save_patterns)
                if not is_save_statement:
                    final_lines.append(line)
            cleaned_code = "\n".join(final_lines)

        final_code = standard_header + cleaned_code.strip() + standard_footer

        return final_code


class QAGeneratorAgent:
    """
    QA Generator Agent - Question & Answer Generator
    Core Mission: Generate precise question text and answer based on concept and visual spec
    """

    def __init__(self):
        """Initialize the question and answer generator."""

    def generate(
        self,
        concept_design: Dict[str, Any],
        visual_spec: str,
        image_path: str,
        difficulty: str,
        feedback: str = "",
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        if not isinstance(concept_design, dict):
            raise ValueError(
                f"QAGeneratorAgent: concept_design must be dict, got {type(concept_design).__name__}"
            )
        if not isinstance(visual_spec, str):
            raise ValueError(
                f"QAGeneratorAgent: visual_spec must be str, got {type(visual_spec).__name__}"
            )
        sys_prompt = (
            f"You are Agent 4 (QA Generator). Difficulty: [{difficulty}] ({_level_desc(difficulty)}).\n\n"
            "## 📥 Your Input\n"
            "- Agent 3's generated image (the actual visualization)\n"
            "- Agent 1's concept design (logic kernel understanding + new question concept)\n"
            "- Agent 1's new question framework (data specs, constraints)\n\n"
            "## 🎯 Your Mission (2 Stages)\n"
            "**Stage 1: Understand All Inputs**\n"
            "- View Agent 3's generated image\n"
            "- Read Agent 1's concept design\n"
            "- Understand the logic kernel and expected answer\n\n"
            "**Stage 2: Generate Question Text + Solution + Answer (JSONL)**\n"
            "- Write clear question text that describes the image\n"
            "- Write detailed solution steps (reasoning process)\n"
            "- Write precise answer\n"
            "- Your output = Complete QA data for JSONL export\n\n"
            "## 📤 Your Output (JSON)\n"
            "```json\n"
            "{\n"
            '  "question_text": "[REQUIRED] Detailed question text (min 50 chars)",\n'
            '  "answer_text": "[REQUIRED] Clear and specific answer",\n'
            '  "solution_steps": ["step1", "step2", "step3"],\n'
            '  "simple_image_description": "[REQUIRED] Brief description of the image content without colors/coordinates details"\n'
            "}\n"
            "```\n\n"
            "## ⚠️ What You DON'T Do\n"
            "❌ Design visuals (Agent 2 already did this)\n"
            "❌ Generate Python code (Agent 3 already did this)\n\n"
            "## ✅ Critical Requirements\n"
            "1. **Field Names**: MUST use `question_text`, `answer_text`, `solution_steps`, `simple_image_description`\n"
            "2. **Question Text**: Detailed description (min 50 chars)\n"
            "3. **Answer Text**: Clear and specific\n"
            "4. **Solution Steps**: Array of strings, logical reasoning process\n"
            "5. **Simple Image Description**: A simple, brief description of what the picture shows. Do not include specific colors, coordinates, or highly specific design details.\n\n"
            "**All content in English.**"
        )

        if feedback:
            sys_prompt += (
                f"\n\n## ⚠️ Feedback from Previous Attempt\n{feedback}\n\n**Address the issues!**\n"
            )

        user_prompt = (
            f"## 🧠 Concept Design\n\n"
            f"{_clip(str(concept_design), 3000)}\n\n"
            f"## 🎨 Visual Specification (Natural Language)\n\n"
            f"{_clip(visual_spec, 3000)}\n\n"
            "Generate the final question text and answer based on the above information."
        )

        result = vision_completion(
            system_prompt=sys_prompt,
            image_path=image_path,
            user_prompt=user_prompt,
            json_mode=True,
            timeout=120.0,
            temperature=1.0,
            model=AGENT_MODELS.get("qa_generator"),
        )
        qa = _ensure_dict(result, "QAGeneratorAgent")

        print(f"   🔍 QA_Generator LLM Response Keys: {list(qa.keys())}")
        print(f"   📝 Question Text Length: {len(qa.get('question_text', ''))} chars")
        print(f"   ✅ Answer Text Length: {len(qa.get('answer_text', ''))} chars")

        return qa
