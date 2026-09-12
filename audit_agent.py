from typing import List, Optional, Tuple

from board_validator import BoardValidator
from math_validator import MathValidator
from utils import (
    AGENT_MODELS,
    FINAL_VERIFIER_MODELS,
    chat_completion,
    multi_vision_completion,
    vision_completion,
)


class FullProcessAuditorAgent:
    """Internal component of the AgSynth generation pipeline."""

    def __init__(self):
        """Internal component of the AgSynth generation pipeline."""
        self.math_validator = MathValidator()
        self.board_validator = BoardValidator()

    def audit_stage1_concept(
        self, concept_design: dict, seed_questions: List[dict], difficulty: str
    ) -> Tuple[bool, Optional[str]]:
        """Internal component of the AgSynth generation pipeline."""
        sys_prompt = (
            "You are a **Senior Mathematical Logic Expert** conducting Stage 1 Audit.\n\n"
            "## 🎯 Your Mission\n\n"
            "Audit the **logic rigor and mathematical closure** of the concept design.\n\n"
            "## 📋 Audit Checklist\n\n"
            "### 1. Mathematical Closure (Mathematical closure) - RELAXED MODE\n"
            "- **Question**: Is the concept design CONCRETE and SPECIFIC enough?\n"
            "- **🎯 NEW: Focus on CONCRETENESS, not mathematical uniqueness!**\n"
            "  - **IMPORTANT**: Stage 1 should produce a CONCRETE question, not just a question type!\n"
            "  - **Expected Answer**: Can be descriptive (e.g., 'Option B', 'Red circle', 'Position (3,5)')\n"
            "  - **Multiple Solutions**: ACCEPTABLE (Stage 4 will handle answer format)\n"
            "  - **Descriptive Answers**: ACCEPTABLE (e.g., 'The path goes through 3 targets')\n"
            "- **Check for**:\n"
            "  - ❌ REJECT: Abstract description without concrete values (e.g., 'some objects with different weights')\n"
            "  - ❌ REJECT: Missing critical data (e.g., 'objects on a board' without specifying how many)\n"
            "  - ❌ REJECT: Contradictory conditions (e.g., 'A > B' and 'B > A')\n"
            "  - ✅ ACCEPT: Concrete values with descriptive answer (e.g., '3 red apples, 5 blue balls' → 'Total: 8 objects')\n"
            "  - ✅ ACCEPT: Multiple solutions with clear answer format (e.g., 'Any path through 3 targets')\n\n"
            "- **🔧 Python Validation Tool Available**:\n"
            "  - If the problem involves 24-point, equation systems, or arithmetic puzzles\n"
            "  - Use the math_verification_hint field to trigger Python validation\n"
            "  - The validator will return: solution_count, sample_solutions, recommendation\n\n"
            "### 1.5. Internal Consistency (Internal consistency) - NEW!\n"
            "- **Question**: Are all data points CONSISTENT throughout the entire concept design?\n"
            "- **Critical Check**: Extract ALL mentions of the SAME entity and verify they have the SAME value\n"
            "- **Examples of Contradictions to Detect**:\n"
            '  - ❌ "White pieces have NOT formed a line" (main description) BUT "White has WON" (notes)\n'
            '  - ❌ "Collect 4 targets" (logic kernel) BUT "Badge shows 3" (data specifications)\n'
            '  - ❌ "Object A weighs 5kg" (list) BUT "Object A weighs 6kg" (constraints)\n'
            '  - ❌ "3 objects" (quantity) BUT lists 4 objects (data)\n'
            "- **Verification Protocol**:\n"
            "  1. Extract all key entities (objects, targets, pieces, etc.)\n"
            "  2. For each entity, collect ALL mentions with their values\n"
            "  3. Check if the SAME entity has DIFFERENT values in different parts\n"
            '  4. Check if logical states are COHERENT (e.g., "not connected" should NOT imply "won")\n'
            '  5. Check if counts MATCH (e.g., "4 targets" should list exactly 4 items)\n\n'
            "### 2. Logic Consistency (Logic consistency)\n"
            "- **Question**: Is the extracted logic kernel truly EQUIVALENT to the seed question?\n"
            "- **Check for**:\n"
            "  - Missing key constraints\n"
            "  - Misinterpreted relationships\n"
            "  - Oversimplified logic\n\n"
            "### 3. Innovation Assessment (Innovation assessment) - ENCOURAGEMENT MODE\n"
            "- **Philosophy**: Innovation is ENCOURAGED but NOT MANDATORY\n"
            "- **Grading Scale**:\n"
            "  - 🌟 High Innovation (High innovation): New visual form + new data structure → EXCELLENT\n"
            "  - ⭐ Medium Innovation (Medium innovation): Same visual form but different data/context → ACCEPTABLE\n"
            "  - ✨ Low Innovation (Low innovation): Only number changes but logic preserved → ACCEPTABLE\n"
            "- **CRITICAL**: Even LOW innovation should PASS if logic is sound!\n"
            "- **Only REJECT if**:\n"
            "  - Exact copy of seed question (Exact copy of seed question)\n"
            "  - No effort to modify anything (No modification)\n"
            "- **Examples of ACCEPTABLE low innovation**:\n"
            '  - Seed: "3 apples + 5 oranges" → New: "4 apples + 6 oranges" (✅ PASS - numbers changed)\n'
            '  - Seed: "5×5 grid" → New: "6×6 grid" (✅ PASS - size changed)\n'
            '  - Seed: "Red circles" → New: "Blue squares" (✅ PASS - color/shape changed)\n\n'
            "## 🚨 Error Severity Classification (NEW - CRITICAL)\n\n"
            "**🔴 CRITICAL ERRORS (MUST REJECT)**:\n"
            "1. **Internal Contradiction Detected** (HIGHEST PRIORITY):\n"
            "   - Same entity has different values in different parts\n"
            '   - Logical states are incoherent (e.g., "not connected" + "won")\n'
            '   - Counts don\'t match (e.g., "4 targets" but lists 3 items)\n'
            '   - Example: "Board data is internally inconsistent... contradictory specifications"\n'
            '2. Logic contradiction detected (e.g., "Condition A conflicts with Condition B")\n'
            "3. **No solution exists** (Python validation confirms 0 solutions)\n"
            "4. Insufficient data to solve (missing critical information)\n"
            "5. **Exact copy of seed question** (no modification) - REMOVED: Low innovation is now ACCEPTABLE\n\n"
            "**🟡 MAJOR ERRORS (REJECT, but provide SPECIFIC fix instructions)**:\n"
            "1. **Coordinate out of bounds** (e.g., (12, 8) on 10×10 canvas)\n"
            "   - Severity: MAJOR\n"
            "   - Action: Provide exact corrected coordinates\n"
            '   - Example: "Change Target_A from (12, 8) to (9.0, 7.5)"\n'
            "2. **Overlapping elements** (distance < minimum spacing)\n"
            "   - Severity: MAJOR\n"
            "   - Action: Provide adjusted positions with spacing calculation\n"
            "3. **Missing key constraints** (problem becomes ambiguous)\n"
            "   - Severity: MAJOR\n"
            "   - Action: Specify exact constraint to add\n\n"
            "**🟢 MINOR ISSUES (ACCEPT with optional suggestions)**:\n"
            "1. **Low innovation level** (only number/color/size changes)\n"
            "   - Severity: MINOR\n"
            "   - Action: ACCEPT, optionally suggest more creative approaches for future\n"
            '   - Example: "Innovation level: Low (only numbers changed). ACCEPTED. Optional: Consider different visual forms next time."\n'
            "2. **Slightly insufficient margin** (e.g., 0.25 instead of 0.3)\n"
            "   - Severity: MINOR\n"
            "   - Action: ACCEPT, provide optional improvement suggestion\n"
            "3. **Minor spacing issues** (e.g., 0.15 instead of 0.2)\n"
            "   - Severity: MINOR\n"
            '   - Action: ACCEPT, note as "non-critical"\n'
            "4. **Stylistic preferences** (color choices, font sizes)\n"
            "   - Severity: MINOR\n"
            "   - Action: ACCEPT without comment\n\n"
            "**MUST INFORM (NOT REJECT) if**:\n"
            "1. **Multiple Solutions Detected** (Python validation confirms 2+ solutions):\n"
            "   - Provide exact solution count and sample solutions\n"
            '   - Recommend answer format: "Any of: [solution1, solution2, ...]"\n'
            "   - Example: \"This 24-point problem has 12 solutions. Suggest answer format: 'Any valid expression using [3,3,8,8] that equals 24'\"\n"
            "2. **Infinite Solutions Detected** (Python validation confirms infinite solutions):\n"
            "   - Recommend adding constraints or adjusting answer format\n"
            "   - Example: \"This equation system has infinite solutions. Suggest: Add one more constraint OR answer format: 'Any (x,y) satisfying 2x+3y=10'\"\n\n"
            "## 📤 Output Format (ENHANCED with Severity & Preservation)\n\n"
            "Return a JSON object:\n"
            "```json\n"
            "{\n"
            '  "pass_audit": true/false,\n'
            '  "audit_score": 1-10,\n'
            '  "innovation_level": "HIGH" | "MEDIUM" | "LOW",  // NEW: Innovation assessment (all levels acceptable)\n'
            '  "error_severity": "CRITICAL" | "MAJOR" | "MINOR",  // NEW: Error severity level\n'
            '  "feedback": "Brief diagnosis (1-2 sentences)",\n'
            '  "issues_found": [\n'
            '    {"issue": "Issue description", "severity": "CRITICAL" | "MAJOR" | "MINOR"}  // NEW: Each issue has severity\n'
            "  ],\n"
            '  "preserved_fields": [  // NEW: Fields that are CORRECT and should NOT be modified\n'
            '    "core_logic_kernel",\n'
            '    "concept_design.logic_description",\n'
            '    "expected_answer"\n'
            "  ],\n"
            '  "fields_to_modify": [  // NEW: ONLY these fields need modification\n'
            '    {"field": "concept_design.data_specifications", "reason": "Coordinate out of bounds", "severity": "MAJOR"},\n'
            '    {"field": "ascii_board_check", "reason": "Winning path not connected", "severity": "CRITICAL"}\n'
            "  ],\n"
            '  "correction_instructions": [\n'
            "    \"SPECIFIC instruction 1: e.g., 'PRESERVE: core_logic_kernel, concept_design.logic_description (these are correct)'\",\n"
            "    \"SPECIFIC instruction 2: e.g., 'MODIFY ONLY: concept_design.data_specifications line 5 - Change Target_A coordinate from (12, 8) to (9.0, 7.5)'\",\n"
            "    \"SPECIFIC instruction 3: e.g., 'MODIFY ONLY: ascii_board_check - Redraw row 3 to connect (C,3) and (D,3)'\"\n"
            "  ],\n"
            '  "consistency_violations": [\n'
            '    {"entity": "White pieces", "location1": "main description", "value1": "not connected", "location2": "notes", "value2": "won", "severity": "CRITICAL"},\n'
            '    {"entity": "Target count", "location1": "logic kernel", "value1": "4", "location2": "badge", "value2": "3", "severity": "CRITICAL"}\n'
            "  ],\n"
            '  "forbidden_patterns": ["Pattern to avoid"],\n'
            '  "required_patterns": ["Pattern to follow"],\n'
            '  "math_validation_result": {\n'
            '    "validated": true/false,\n'
            '    "solution_count": 0/1/12/-1,\n'
            '    "sample_solutions": ["solution1", "solution2", ...],\n'
            '    "recommendation": "Specific recommendation for answer format"\n'
            "  }\n"
            "}\n"
            "```\n\n"
            "**CRITICAL RULES for correction_instructions (ENHANCED)**:\n"
            '1. ❌ FORBIDDEN: "Your logic is not closed" (too vague)\n'
            '2. ✅ REQUIRED: "PRESERVE: core_logic_kernel, expected_answer (correct). MODIFY ONLY: data_specifications line 5 - Change Target_A from (12,8) to (9.0,7.5)" (specific + preservation)\n'
            '3. ❌ FORBIDDEN: "Data is inconsistent" (too vague)\n'
            "4. ✅ REQUIRED: \"PRESERVE: All logic descriptions (correct). MODIFY ONLY: Badge number from '3' to '4' in data_specifications to match logic kernel\" (specific + preservation)\n"
            '5. **NEW**: Always start with "PRESERVE: [list of correct fields]" to prevent full regeneration\n'
            '6. **NEW**: Then specify "MODIFY ONLY: [specific field] - [exact change]" for surgical fixes\n'
            "7. Each instruction must be DIRECTLY ACTIONABLE (copy-paste ready)\n"
            '8. **NEW**: For MINOR errors, use "OPTIONAL: [suggestion]" instead of "MODIFY"'
        )

        math_validation_result = None

        self_validation = concept_design.get("self_validation_result")
        if self_validation and self_validation.get("validated"):
            print("   ✅ Using ConceptDesigner self-validation result")
            math_validation_result = {
                "solution_count": self_validation.get("solution_count", 0),
                "recommendation": self_validation.get("recommendation", ""),
                "validation_result": {
                    "sample_solutions": self_validation.get("sample_solutions", [])
                },
            }
        else:
            math_hint = concept_design.get("math_verification_hint")
            if math_hint and isinstance(math_hint, dict):
                problem_type = math_hint.get("type")
                if problem_type:
                    print(f"   🔧 Auditor is validating mathematical logic: {problem_type}")
                    try:
                        validation = self.math_validator.detect_multiple_solutions(
                            problem_type=problem_type, problem_data=math_hint
                        )
                        math_validation_result = validation
                        print(
                            f"   🔧 Python validation result: {validation.get('solution_count')} solutions"
                        )
                    except Exception as e:
                        print(f"   ⚠️ Python Validation failed: {e}")

        board_validation_result = None
        ascii_board = concept_design.get("ascii_board_check", "")

        if ascii_board:
            print("   🎯 ASCII board detected; starting board-logic validation...")
            try:
                board_validation = self.board_validator.validate_concept_with_reflection(
                    concept_design=concept_design, expected_winner=None, required_length=5
                )
                board_validation_result = board_validation

                if not board_validation["is_valid"]:
                    print("   ❌ Board logic validation failed!")
                    print(
                        f"   💡 Revision feedback: {board_validation['reflection_prompt'][:200]}..."
                    )
                else:
                    print("   ✅ Board-logic validation passed")
            except Exception as e:
                print(f"   ⚠️ Board validation failed: {e}")

        user_prompt = (
            f"## 📊 Concept Design to Audit\n\n"
            f"**Difficulty**: {difficulty}\n\n"
            f"**Core Logic Kernel**:\n{concept_design.get('core_logic_kernel', 'N/A')}\n\n"
            f"**Concept Design**:\n{concept_design.get('concept_design', 'N/A')}\n\n"
            f"**Expected Answer**:\n{concept_design.get('expected_answer', 'N/A')}\n\n"
            f"**Design Rationale**:\n{concept_design.get('design_rationale', 'N/A')}\n\n"
        )

        if ascii_board:
            user_prompt += (
                f"## 🎯 ASCII Board Check (CRITICAL)\n\n"
                f"**ConceptDesigner provided an ASCII board**:\n"
                f"```\n{ascii_board}\n```\n\n"
            )

        if board_validation_result:
            is_valid = board_validation_result["is_valid"]
            reflection = board_validation_result["reflection_prompt"]

            user_prompt += (
                f"## 🎯 Board Logic Validation Result (HIGHEST PRIORITY)\n\n"
                f"**Validation Status**: {'✅ PASSED' if is_valid else '❌ FAILED'}\n\n"
            )

            if not is_valid:
                user_prompt += (
                    f"**⚠️ CRITICAL FAILURE - Board logic is INVALID**:\n\n"
                    f"{reflection}\n\n"
                    f"**MANDATORY ACTION**: You MUST REJECT this concept design and provide the reflection prompt to ConceptDesigner.\n"
                    f"**DO NOT PROCEED** until the board logic is fixed.\n\n"
                )

        if math_validation_result:
            solution_count = math_validation_result.get("solution_count", 0)
            recommendation = math_validation_result.get("recommendation", "")
            sample_solutions = math_validation_result.get("validation_result", {}).get(
                "sample_solutions", []
            )

            user_prompt += (
                f"## 🔧 Python Validation Result (CRITICAL)\n\n"
                f"**Solution Count**: {solution_count}\n"
                f"**Sample Solutions**: {sample_solutions[:3]}\n"
                f"**Recommendation**: {recommendation}\n\n"
                f"**⚠️ IMPORTANT**: \n"
                f"- If solution_count = 0 → REJECT (no solution)\n"
                f"- If solution_count = 1 → PASS (unique solution)\n"
                f"- If solution_count > 1 → PASS, but INFORM ConceptDesigner to adjust answer format\n"
                f"- If solution_count = -1 → PASS, but INFORM ConceptDesigner about infinite solutions\n\n"
            )

        user_prompt += (
            f"## 🌱 Seed Question Reference\n\n"
            f"**Seed Question Type**: {seed_questions[0].get('question_type', 'N/A')}\n"
            f"**Seed Question**: {seed_questions[0].get('question', 'N/A')[:500]}\n"
            f"**Seed Answer**: {seed_questions[0].get('answer', 'N/A')}\n\n"
            f"## 🔍 Your Task\n\n"
            f"Conduct a thorough audit of the concept design. Focus on:\n"
            f"1. **PRIORITY 1**: Check for INTERNAL CONTRADICTIONS (same entity with different values)\n"
            f"   - Extract all entities and their values from ALL parts of the design\n"
            f"   - Verify the SAME entity has the SAME value everywhere\n"
            f'   - Check logical coherence (e.g., "not connected" should NOT imply "won")\n'
            f"2. Is the problem SOLVABLE? (Use Python validation result if available)\n"
            f"3. Is the logic kernel truly equivalent to the seed question?\n"
            f"4. Does the design show genuine innovation (not just number changes)?\n\n"
            f'**Be EXTRA strict on internal consistency. This is the #1 cause of "dead questions".**\n'
            f"**If Python validation shows multiple solutions, INFORM (don't reject) and provide specific answer format recommendations.**"
        )

        try:
            result = chat_completion(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                json_mode=True,
                timeout=60.0,
                temperature=0.3,
                model=AGENT_MODELS.get("auditor"),
            )

            pass_audit = result.get("pass_audit", False)
            feedback = result.get("feedback", "")
            audit_score = result.get("audit_score", 5)
            issues = result.get("issues_found", [])
            error_severity = result.get("error_severity", "CRITICAL")

            if board_validation_result and not board_validation_result["is_valid"]:
                print("   ❌ Board logic validation failed,Forced rejection!")
                reflection_prompt = board_validation_result["reflection_prompt"]

                return False, (
                    f"❌ Stage 1 Audit Failed - Board Logic Invalid\n\n"
                    f"**Python Validator detected critical board logic errors**:\n\n"
                    f"{reflection_prompt}\n\n"
                    f"**You MUST redraw the ASCII board and fix the coordinate errors before proceeding.**"
                )

            if not pass_audit:
                structured_feedback = f"❌ Stage 1 Audit Failed (Score: {audit_score}/10)\n\n"
                structured_feedback += "## 🚨 REVISION GUIDE - Mandatory revision guide\n\n"
                structured_feedback += f"**Audit diagnosis**: {feedback}\n\n"

                correction_instructions = result.get("correction_instructions", [])
                if correction_instructions:
                    structured_feedback += "### ✅ Mandatory revision requirements (MANDATORY):\n\n"
                    for i, instruction in enumerate(correction_instructions, 1):
                        structured_feedback += f"{i}. {instruction}\n"
                    structured_feedback += "\n"

                if issues:
                    structured_feedback += "### 📋 Specific issue list:\n\n"
                    for i, issue in enumerate(issues, 1):
                        structured_feedback += f"{i}. {issue}\n"
                    structured_feedback += "\n"

                consistency_violations = result.get("consistency_violations", [])
                if consistency_violations:
                    structured_feedback += "### 🚨 Internal consistency violation (CRITICAL):\n\n"
                    for violation in consistency_violations:
                        entity = violation.get("entity", "Unknown")
                        loc1 = violation.get("location1", "")
                        val1 = violation.get("value1", "")
                        loc2 = violation.get("location2", "")
                        val2 = violation.get("value2", "")
                        structured_feedback += f"- **{entity}**: '{val1}' in '{loc1}' conflicts with '{val2}' in '{loc2}' (contradiction!)\n"
                    structured_feedback += "\n"

                forbidden_patterns = result.get("forbidden_patterns", [])
                required_patterns = result.get("required_patterns", [])

                if forbidden_patterns:
                    structured_feedback += "### ❌ Forbidden patterns:\n\n"
                    for pattern in forbidden_patterns:
                        structured_feedback += f"- {pattern}\n"
                    structured_feedback += "\n"

                if required_patterns:
                    structured_feedback += "### ✅ Required patterns:\n\n"
                    for pattern in required_patterns:
                        structured_feedback += f"- {pattern}\n"
                    structured_feedback += "\n"

                if error_severity == "MINOR":
                    structured_feedback += "### ℹ️ Decision: ACCEPT with warnings\n\n"
                    structured_feedback += "Minor issues remain, but they do not affect overall quality; accept and continue.\n"
                    return True, structured_feedback

                structured_feedback += "### 💡 Targeted suggestions:\n\n"

                if any(
                    "coordinates" in str(issue) or "out of bounds" in str(issue) for issue in issues
                ):
                    structured_feedback += "🎯 **Coordinate out-of-bounds issue**: check all coordinates against the canvas and draw the ASCII board first.\n"

                if any("overlap" in str(issue) or "spacing" in str(issue) for issue in issues):
                    structured_feedback += "🎯 **Point-overlap issue**: increase element spacing to avoid visual occlusion.\n"

                if any(
                    "logic" in str(issue).lower()
                    or "contradiction" in str(issue).lower()
                    or "invalid" in str(issue).lower()
                    for issue in issues
                ):
                    structured_feedback += "🎯 **Logic inconsistency**: check that the data specifications and logic kernel agree and yield a clear answer.\n"

                structured_feedback += "\n**🚨 CRITICAL**: Fix only the issues above and preserve all correct fields!\n"

                return False, structured_feedback

            return True, None

        except Exception as e:
            print(f"⚠️ Stage 1 Audit Error: {e}")
            return False, f"Stage 1 audit unavailable: {e}"

    def audit_stage2_visual(
        self, visual_spec: str, concept_design: dict, seed_image_paths: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Internal component of the AgSynth generation pipeline."""
        sys_prompt = (
            "You are a **Visual Communication and Spatial Geometry Expert** conducting Stage 2 Audit.\n\n"
            "## 🎯 Your Mission\n\n"
            "Audit the **visual design quality and logic-to-visual translation accuracy**.\n\n"
            "## 📋 Audit Checklist\n\n"
            "### 1. Logic Mapping (Logic mapping)\n"
            "- **Question**: Can the chosen visual form (e.g., gears, balance) accurately express the mathematical logic?\n"
            "- **Check for**:\n"
            "  - Visual metaphor mismatch (Visual metaphor mismatch)\n"
            "  - Logic lost in translation (Logic translation is lost)\n"
            "  - Ambiguous visual representation (Ambiguous visual representation)\n\n"
            "### 2. Design Completeness (Design completeness) - RELAXED STANDARDS\n"
            "- **Question**: Does the specification provide sufficient design details for Painter?\n"
            "- **RECOMMENDED (NOT MANDATORY)**:\n"
            "  1. Canvas size is helpful but NOT required (Painter can use default 12x12)\n"
            "  2. Coordinates are helpful but NOT required (Painter can auto-layout)\n"
            "  3. Element descriptions should be clear and specific\n"
            "- **ONLY REJECT if**:\n"
            "  - Design is completely vague (e.g., 'draw some shapes')\n"
            "  - Missing critical element descriptions (e.g., 'draw objects' without specifying what)\n"
            "  - Contradictory design instructions (e.g., 'place A at center' and 'place A at corner')\n"
            "- **ACCEPT**: Designs with general layout descriptions (Painter will handle specifics)\n\n"
            "### 3. Design Feasibility - TRUST PAINTER\n"
            "- **Philosophy**: Stage 2 focuses on DESIGN INTENT, not implementation details\n"
            "- **Question**: Is the design conceptually sound and implementable?\n"
            "- **DO NOT CHECK**:\n"
            "  - Exact coordinate calculations (Painter will handle this)\n"
            "  - Element spacing calculations (Painter will auto-adjust)\n"
            "  - Canvas boundary checks (Painter will ensure elements fit)\n"
            "- **ONLY CHECK**:\n"
            "  - Design logic makes sense (e.g., 'balance scale' for weight comparison)\n"
            "  - Element relationships are clear (e.g., 'A connects to B')\n"
            "  - No impossible requirements (e.g., '100 elements on 5x5 canvas')\n\n"
            "### 4. Visual Purity (Visual purity)\n"
            "- **Question**: Is the design pure visual (no question text in image)?\n"
            "- **Check for**:\n"
            "  - Question text in specification\n"
            "  - Instructions in image\n"
            "  - Excessive text annotations\n\n"
            "### 5. Form Innovation (Form innovation)\n"
            "- **Question**: Is the visual form DRASTICALLY DIFFERENT from the seed question?\n"
            "- **Check for**:\n"
            "  - Same visual structure\n"
            "  - Minor variations only\n"
            "  - Lack of creative thinking\n\n"
            "## 🚨 Rejection Criteria (RELAXED - TRUST PAINTER)\n\n"
            "**MUST REJECT ONLY if**:\n"
            "1. **Data Drift Detected** (HIGHEST PRIORITY):\n"
            "   - Any locked field value modified from Agent 1's specification\n"
            "   - Example: Agent 1 says '4 targets' but Agent 2 specifies '3 targets'\n"
            "2. **Design Completely Vague**:\n"
            "   - No clear element descriptions (e.g., 'draw something')\n"
            "   - Missing critical design intent (e.g., 'show relationship' without specifying how)\n"
            "3. **Visual Form Mismatch**:\n"
            "   - Visual form fundamentally doesn't match logic\n"
            "   - Question text found in specification (pure visual violated)\n"
            "4. **Lack of Innovation**:\n"
            "   - Visual form identical or highly similar to seed question\n\n"
            "**DO NOT REJECT for** (Painter will handle these):\n"
            "- Missing canvas size (Painter uses default 12x12)\n"
            "- Missing exact coordinates (Painter will auto-layout)\n"
            "- Potential spacing issues (Painter will adjust)\n"
            "- Potential overlap concerns (Painter will prevent)\n"
            "- Edge margin concerns (Painter will ensure fit)\n\n"
            "**ACCEPT (Minor Issues)**:\n"
            "- Slight visual style variations\n"
            "- Non-critical annotation placement adjustments\n\n"
            "## 📤 Output Format\n\n"
            "Return a JSON object:\n"
            "```json\n"
            "{\n"
            '  "pass_audit": true/false,\n'
            '  "audit_score": 1-10,\n'
            '  "feedback": "Brief diagnosis (1-2 sentences)",\n'
            '  "issues_found": ["Issue 1", "Issue 2"],\n'
            '  "layout_conflicts": ["Specific conflict with coordinates"],\n'
            '  "data_drift_violations": [\n'
            '    {"locked_field": "target_count", "locked_value": "4", "actual_value": "3", "severity": "critical"},\n'
            '    {"locked_field": "object_A_weight", "locked_value": "5kg", "actual_value": "6kg", "severity": "critical"}\n'
            "  ],\n"
            '  "correction_instructions": [\n'
            "    \"SPECIFIC instruction: e.g., 'Move Element B from (2.5, 5) to (3.0, 5) to ensure 0.2+ units spacing'\",\n"
            "    \"SPECIFIC calculation: e.g., 'Canvas: 12x10, Element radius: 0.4, Label: 0.6, Min spacing: 0.2, Min margin: 0.3'\",\n"
            "    \"DATA CONSISTENCY: e.g., 'Change target count from 3 to 4 to match Agent 1 locked_data_fields'\"\n"
            "  ],\n"
            '  "canvas_completeness_check": {\n'
            '    "canvas_size_specified": true/false,\n'
            '    "all_elements_have_coordinates": true/false,\n'
            '    "coordinates_within_bounds": true/false,\n'
            '    "missing_coordinates": ["Element_A", "Element_B"],\n'
            '    "out_of_bounds_coordinates": [{"element": "Element_C", "position": [15, 8], "canvas": [12, 12]}]\n'
            "  },\n"
            '  "spatial_requirements": {\n'
            '    "min_spacing": "0.2 units (between element centers) - RELAXED",\n'
            '    "min_margin": "0.3 units (from canvas edges) - RELAXED",\n'
            '    "element_size": "radius + label width + padding",\n'
            '    "canvas_size": "MUST be explicitly specified (e.g., 12x12)"\n'
            "  }\n"
            "}\n"
            "```\n\n"
            "**CRITICAL RULES for correction_instructions**:\n"
            '1. ❌ FORBIDDEN: "Elements are too close" (too vague)\n'
            '2. ✅ REQUIRED: "Element A at (2,5), Element B at (2.3,5), distance=0.3 > 0.2 required ✅" (specific)\n'
            "3. Must include EXACT coordinates and EXACT calculations\n"
            "4. Must show the MATH: current distance, required distance, proposed new position"
        )

        locked_data = concept_design.get("locked_data_fields", {})
        locked_data_str = str(locked_data) if locked_data else "N/A"

        user_prompt = (
            f"## 🎨 Visual Specification to Audit (Natural Language)\n\n"
            f"{visual_spec}\n\n"
            f"## 🧠 Logic Context (from ConceptDesigner)\n\n"
            f"**Core Logic**: {concept_design.get('core_logic_kernel', 'N/A')}\n\n"
            f"## 🔒 LOCKED DATA FIELDS (Reference for Verification)\n\n"
            f"**Agent 1's Locked Data**: {locked_data_str}\n\n"
            f"**CRITICAL AUDIT TASK**: Verify data consistency in Agent 2's specification\n"
            f"- Extract ALL key-value pairs from locked_data_fields\n"
            f"- For EACH locked field, verify Agent 2's specification mentions the EXACT same value\n"
            f"- Flag ANY mismatch as CRITICAL data drift\n\n"
            f"## 🔍 Your Task\n\n"
            f"Audit the visual design. Check in this order:\n"
            f"1. **PRIORITY 1**: DATA CONSISTENCY - Verify data values match locked_data_fields\n"
            f"   - Extract all locked fields from Agent 1's locked_data_fields\n"
            f"   - Verify Agent 2's specification contains the EXACT same values\n"
            f"   - Flag ANY mismatch (e.g., locked says '4 targets' but spec shows '3')\n"
            f"2. **PRIORITY 2**: DESIGN CLARITY - Is the design intent clear?\n"
            f"   - Are element descriptions specific and actionable?\n"
            f"   - Is the visual form choice appropriate for the logic?\n"
            f"   - Are there any contradictory instructions?\n"
            f"3. Does the visual form accurately represent the logic?\n"
            f"4. Is the design pure visual (no question text)?\n"
            f"5. Is the visual form truly different from the seed question?\n\n"
            f"**IMPORTANT: Trust Painter to handle layout details (canvas size, coordinates, spacing).**\n"
            f"**Focus on design intent and data consistency, not implementation specifics.**"
        )

        try:
            if seed_image_paths:
                result = multi_vision_completion(
                    system_prompt=sys_prompt,
                    image_paths=seed_image_paths,
                    user_prompt=user_prompt,
                    timeout=60.0,
                    temperature=0.3,
                    json_mode=True,
                    model=AGENT_MODELS.get("auditor"),
                )
            else:
                result = chat_completion(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    json_mode=True,
                    timeout=60.0,
                    temperature=0.3,
                    model=AGENT_MODELS.get("auditor"),
                )

            pass_audit = result.get("pass_audit", False)
            feedback = result.get("feedback", "")
            audit_score = result.get("audit_score", 5)
            layout_conflicts = result.get("layout_conflicts", [])

            if not pass_audit:
                conflict_info = (
                    "\n".join([f"  - {c}" for c in layout_conflicts]) if layout_conflicts else ""
                )
                return (
                    False,
                    f"❌ Stage 2 Audit Failed (Score: {audit_score}/10)\n\n{feedback}\n\n**Layout Conflicts**:\n{conflict_info}",
                )

            return True, None

        except Exception as e:
            print(f"⚠️ Stage 2 Audit Error: {e}")
            return False, f"Stage 2 audit unavailable: {e}"

    def audit_stage3_image(
        self, image_path: str, visual_spec: str, python_code: str
    ) -> Tuple[bool, Optional[str]]:
        """Internal component of the AgSynth generation pipeline."""
        sys_prompt = (
            "You are an **EXTREMELY STRICT Image Quality Control Officer (QC)** conducting Stage 3 Audit.\n\n"
            "## 🎯 Your Mission\n\n"
            "Audit the **generated image quality and specification compliance** with ZERO TOLERANCE for flaws.\n\n"
            "## 📋 Audit Checklist\n\n"
            "### 1. Rendering Completeness (Rendering completeness)\n"
            "- **Question**: Are all 5 required elements correctly rendered?\n"
            "  1. Graphic Elements (graphic elements)\n"
            "  2. Layout (layout)\n"
            "  3. Coordinates (coordinate positions)\n"
            "  4. Color Scheme (color scheme)\n"
            "  5. Annotations (annotations)\n"
            "- **Check for**:\n"
            "  - Missing elements (missing elements)\n"
            "  - Incorrect positions (incorrect positions)\n"
            "  - Wrong colors (wrong colors)\n\n"
            "### 2. Visual Readability (Visual readability) - IMAGE-BASED VALIDATION (EXTREMELY STRICT)\n"
            "- **Question**: Is the ACTUAL generated image clear, professional, and completely free from overlapping or occlusion?\n"
            "- **CRITICAL**: Analyze the ACTUAL IMAGE, not the specification!\n"
            "- **VISUAL CHECKS** (based on what you SEE in the image):\n"
            "  1. **Element Visibility**: Can you clearly see all main elements in the image?\n"
            "  2. **🚨 Actual Overlapping & Occlusion (CRITICAL)**: Are elements VISUALLY overlapping, touching, or covering each other?\n"
            "     - **ZERO TOLERANCE FOR OVERLAP**: Even the slightest overlap or touching of text/shapes MUST BE REJECTED.\n"
            "     - All labels MUST have clear spacing from graphics to avoid any visual confusion.\n"
            "  3. **Label Readability**: Are labels/annotations completely clear and highly readable?\n"
            "  4. **Canvas Fit**: Are elements completely within the canvas without touching the edges?\n"
            "  5. **Professional Quality**: Does the image look professional and crystal clear?\n"
            "  6. **🚨 Image Integrity**: Is the image file valid and high quality?\n"
            "- **MUST REJECT and SCORE < 5 if**:\n"
            "  - 🚨 **ANY OVERLAPPING OR OCCLUSION**: Elements touching, overlapping, or occluding each other in any way. Text unreadable.\n"
            "  - 🚨 **CRITICAL ELEMENTS MISSING**: Main graphics or data points are absent.\n"
            "  - 🚨 **IMAGE CORRUPTED or UNCLEAR**: Image file is damaged, blurry, or difficult to read.\n"
            "  - Elements are clipped by canvas edges.\n\n"
            "### 3. Compliance Check (Compliance check)\n"
            "- **Question**: Are there any unwanted elements or bugs?\n"
            "- **Check for**:\n"
            "  - Extra axis ticks (extra axis ticks)\n"
            "  - Unnecessary legends (unnecessary legends)\n"
            "  - Text watermarks (text watermarks)\n"
            "  - Code rendering errors (code rendering errors)\n\n"
            "## 🚨 Rejection Criteria (EXTREMELY STRICT)\n\n"
            "**MUST REJECT AND SCORE VERY LOW (< 5) if**:\n"
            "1. 🚨 **ANY OVERLAPPING / OCCLUSION**: Elements covering each other even slightly, text overlapping shapes or other text.\n"
            "2. 🚨 **LACK OF CLARITY**: Any part of the image is blurry, hard to distinguish, or confusing.\n"
            "3. 🚨 **CRITICAL ELEMENTS MISSING**: Main graphics or data completely absent.\n"
            "4. Code rendering completely failed (no image generated).\n\n"
            "**ACCEPT (Minor Issues)**:\n"
            "- Slightly thin lines or minor visual imperfections\n"
            "- Minor annotation misalignment (< 0.5 units off)\n"
            "- Slight element proximity to edges (margin > 0.3 units)\n"
            "- Non-critical rendering artifacts\n\n"
            "## 📤 Output Format\n\n"
            "Return a JSON object:\n"
            "```json\n"
            "{\n"
            '  "pass_audit": true/false,\n'
            '  "audit_score": 1-10,\n'
            '  "feedback": "Brief diagnosis (1-2 sentences)",\n'
            '  "missing_elements": ["Element 1 with expected position", ...],\n'
            '  "quality_issues": ["Issue 1 with specific metric", ...],\n'
            '  "correction_instructions": [\n'
            "    \"SPECIFIC code fix: e.g., 'Line 15: Change plt.text(x, y, label) to plt.text(x, y+0.3, label, ha=center)'\",\n"
            "    \"SPECIFIC parameter: e.g., 'Add linewidth=2 to all plt.plot() calls for better visibility'\"\n"
            "  ],\n"
            '  "code_line_fixes": [\n'
            '    {"line": 15, "current": "plt.text(x, y, label)", "fixed": "plt.text(x, y+0.3, label, ha=\'center\')"}\n'
            "  ]\n"
            "}\n"
            "```\n\n"
            "**CRITICAL RULES for correction_instructions**:\n"
            '1. ❌ FORBIDDEN: "Image quality is poor" (too vague)\n'
            '2. ✅ REQUIRED: "Line 12: Add linewidth=2.5 to plt.plot() for thicker lines" (specific)\n'
            "3. Must reference EXACT line numbers or function calls\n"
            "4. Must provide EXACT code snippets (copy-paste ready)"
        )

        visual_spec_truncated = (
            visual_spec[:1000] if isinstance(visual_spec, str) else str(visual_spec)[:1000]
        )
        python_code_str = str(python_code) if python_code is not None else ""
        python_code_truncated = python_code_str[:1500] if isinstance(python_code_str, str) else ""

        user_prompt = (
            f"## 🖼️ Generated Image to Audit\n\n"
            f"**Image Path**: {image_path}\n\n"
            f"## 📐 Visual Specification (Expected - Natural Language)\n\n"
            f"{visual_spec_truncated}\n\n"
            f"## 💻 Python Code (for reference)\n\n"
            f"```python\n{python_code_truncated}\n```\n\n"
            f"## 🔍 Your Task\n\n"
            f"Audit the generated image. Check:\n"
            f"1. Are all required elements (graphics, layout, coordinates, colors, annotations) present?\n"
            f"2. Is the image clear and readable?\n"
            f"3. Are there any rendering errors or unwanted elements?\n\n"
            f"**Be thorough but practical. Minor imperfections are acceptable.**"
        )

        try:
            result = vision_completion(
                system_prompt=sys_prompt,
                image_path=image_path,
                user_prompt=user_prompt,
                timeout=60.0,
                temperature=0.1,
                model=AGENT_MODELS.get("auditor"),
            )

            pass_audit = result.get("pass_audit", False)
            feedback = result.get("feedback", "")
            audit_score = result.get("audit_score", 5)
            missing = result.get("missing_elements", [])
            quality_issues = result.get("quality_issues", [])

            if not pass_audit or audit_score < 8:
                missing_info = "\n".join([f"  - {m}" for m in missing]) if missing else "None"
                quality_info = (
                    "\n".join([f"  - {q}" for q in quality_issues]) if quality_issues else "None"
                )
                return False, (
                    f"❌ Stage 3 Audit Failed (Score: {audit_score}/10)\n\n"
                    f"{feedback}\n\n"
                    f"**Missing Elements**:\n{missing_info}\n\n"
                    f"**Quality Issues**:\n{quality_info}\n\n"
                    f"**Requirement**: Score must be >= 8 to pass. Please rigorously fix all overlapping and clarify all visuals."
                )

            return True, None

        except Exception as e:
            print(f"⚠️ Stage 3 Audit Error: {e}")
            return False, f"Stage 3 audit unavailable: {e}"

    def audit_stage4_qa(
        self, qa_result: dict, image_path: str, concept_design: dict, visual_spec: str
    ) -> Tuple[bool, Optional[str]]:
        """Internal component of the AgSynth generation pipeline."""
        sys_prompt = (
            "You are an **EXTREMELY STRICT Senior Educational Researcher / Chief Editor** conducting Stage 4 Audit.\n\n"
            "## 🎯 Your Mission\n\n"
            "Audit the **text-image consistency and overall question quality** with ZERO TOLERANCE for any errors. Any flaw MUST result in a very low score (< 5) and rejection.\n\n"
            "## 📋 Audit Checklist\n\n"
            "### 1. Text-Image Consistency (Text-image consistency) - ABSOLUTE PERFECTION REQUIRED\n"
            "- **Question**: Does the question text ABSOLUTELY PERFECTLY match the actual image content?\n"
            "- **Check for**:\n"
            "  - Number mismatch (e.g., text says 5 apples, image shows 4)\n"
            "  - Direction mismatch (e.g., text says clockwise, image shows counterclockwise)\n"
            "  - Element mismatch (e.g., text mentions circles, image has squares)\n"
            '  - Position mismatch (e.g., text says "top", image shows "bottom")\n'
            "  - 🚨 **Image Quality Re-check**: Verify image has absolutely NO overlapping, occluded, or missing elements. The image MUST BE crystal clear and perfectly correspond to the text.\n\n"
            "### 2. Solution Logic Rigor (Solution rigor) - 100% CORRECTNESS REQUIRED\n"
            "- **Question**: Are the solution steps logically flawless and mathematically accurate?\n"
            "- **Check for**:\n"
            "  - Missing steps (missing steps)\n"
            "  - Calculation errors (calculationerror)\n"
            "  - Logic jumps (logic jumps)\n"
            "  - Unclear reasoning (unclear reasoning)\n\n"
            "### 3. Difficulty Alignment (Difficulty alignment)\n"
            "- **Question**: Does the final question match the required difficulty?\n"
            "- **Check for**:\n"
            '  - Too simple for "difficult" setting (too simple for a difficult setting)\n'
            '  - Too complex for "simple" setting (too complex for a simple setting)\n'
            "  - Step count mismatch (step count mismatch)\n\n"
            "## 🚨 Rejection Criteria (MUST REJECT AND SCORE < 5 IF ANY FLAW EXISTS)\n\n"
            "**MUST REJECT and give LOW SCORE if** (CRITICAL issues):\n"
            "1. 🚨 **IMAGE QUALITY ISSUES** (Re-check from Stage 3):\n"
            "   - ANY overlapping or occlusion: Elements touching or covering each other even slightly\n"
            "   - ANY unreadable text or numbers\n"
            "   - Critical elements missing: Main graphics or data points absent\n"
            "2. 🚨 **TEXT-IMAGE MISMATCH** (CRITICAL):\n"
            "   - ANY mismatch between the text description and the image visuals\n"
            "   - The image contains elements not mentioned in the text that cause confusion\n"
            "3. 🚨 **SOLUTION LOGIC ERRORS**:\n"
            "   - Solution steps contain ANY logical errors, calculation mistakes, or jumps in reasoning\n"
            "   - Steps don't lead perfectly to the stated answer\n"
            "4. 🚨 **ANSWER CORRECTNESS**:\n"
            "   - Answer is INCORRECT, ambiguous, or lacks solid proof from the steps\n"
            "5. 🚨 **DIFFICULTY MISMATCH**:\n"
            "   - Simple question with 10+ steps, or difficult with 1 step\n\n"
            "**NOTE**: Stage 4 is the FINAL checkpoint. Be EXTREMELY thorough and strict! ZERO TOLERANCE for errors.\n\n"
            "## 📤 Output Format\n\n"
            "Return a JSON object:\n"
            "```json\n"
            "{\n"
            '  "pass_audit": true/false,\n'
            '  "audit_score": 1-10,\n'
            '  "feedback": "Detailed diagnosis explaining the exact flaw if rejected",\n'
            '  "consistency_issues": ["Specific mismatch with exact values", ...],\n'
            '  "logic_issues": ["Specific logic error with step number", ...],\n'
            '  "correction_instructions": [\n'
            "    \"SPECIFIC text fix: e.g., 'Question text line 2: Change clockwise to counterclockwise to match image'\",\n"
            "    \"SPECIFIC number fix: e.g., 'Answer: Change 5 to 4 (image shows 4 apples, not 5)'\"\n"
            "  ],\n"
            '  "text_replacements": [\n'
            '    {"location": "Question line 2", "wrong": "clockwise", "correct": "counterclockwise", "reason": "Image shows counterclockwise"}\n'
            "  ]\n"
            "}\n"
            "```\n\n"
            "**CRITICAL RULES for correction_instructions**:\n"
            '1. ❌ FORBIDDEN: "Text doesn\'t match image" (too vague)\n'
            "2. ✅ REQUIRED: \"Question says '5 apples' but image shows 4. Change '5' to '4' in question text\" (specific)\n"
            "3. Must identify EXACT word/number to change\n"
            "4. Must provide EXACT replacement value\n"
            "5. Must explain WHY (reference to image content)"
        )

        user_prompt = (
            f"## 📝 QA Result to Audit\n\n"
            f"**Question Text**: {qa_result.get('question_text', 'N/A')}\n\n"
            f"**Answer Text**: {qa_result.get('answer_text', 'N/A')}\n\n"
            f"**Solution Steps**:\n"
        )

        solution_steps = qa_result.get("solution_steps", [])
        if isinstance(solution_steps, list):
            for i, step in enumerate(solution_steps, 1):
                user_prompt += f"{i}. {step}\n"
        else:
            user_prompt += f"{solution_steps}\n"

        user_prompt += (
            f"\n## 🧠 Logic Context\n\n"
            f"**Expected Answer**: {concept_design.get('expected_answer', 'N/A')}\n"
            f"**Core Logic**: {concept_design.get('core_logic_kernel', 'N/A')[:500]}\n\n"
            f"## 🎨 Visual Context (Natural Language)\n\n"
            f"{visual_spec[:500] if isinstance(visual_spec, str) else str(visual_spec)[:500]}\n\n"
            f"## 🔍 Your Task (ENHANCED - FINAL CHECKPOINT)\n\n"
            f"Audit the QA result comprehensively. Check in this order:\n"
            f"1. 🚨 **PRIORITY 1 - IMAGE QUALITY RE-CHECK**:\n"
            f"   - Look at the ACTUAL image carefully\n"
            f"   - Are there any severe overlapping issues? (elements covering each other)\n"
            f"   - Are all critical elements visible and clear?\n"
            f"   - Is the image professional and readable?\n"
            f"2. **TEXT-IMAGE CONSISTENCY**:\n"
            f"   - Does the question text accurately describe what's in the image?\n"
            f"   - Do numbers, directions, positions match?\n"
            f"3. **SOLUTION LOGIC**:\n"
            f"   - Are the solution steps logically sound and complete?\n"
            f"   - Do the steps lead to the stated answer?\n"
            f"4. **ANSWER CORRECTNESS**:\n"
            f"   - Is the answer correct based on the image?\n"
            f"   - Does it match the expected answer from concept design?\n"
            f"5. **DIFFICULTY MATCH**:\n"
            f"   - Does the difficulty match the requirement?\n\n"
            f"**🚨 CRITICAL: This is the FINAL checkpoint before quality rating.**\n"
            f"**If you find ANY severe issues (especially image quality problems), REJECT immediately!**"
        )

        try:
            result = vision_completion(
                system_prompt=sys_prompt,
                image_path=image_path,
                user_prompt=user_prompt,
                timeout=60.0,
                temperature=0.1,
                model=AGENT_MODELS.get("auditor"),
            )

            pass_audit = result.get("pass_audit", False)
            feedback = result.get("feedback", "")
            audit_score = result.get("audit_score", 5)
            consistency_issues = result.get("consistency_issues", [])
            logic_issues = result.get("logic_issues", [])

            if not pass_audit or audit_score < 8:
                consistency_info = (
                    "\n".join([f"  - {c}" for c in consistency_issues])
                    if consistency_issues
                    else "None"
                )
                logic_info = (
                    "\n".join([f"  - {issue}" for issue in logic_issues])
                    if logic_issues
                    else "None"
                )
                return False, (
                    f"❌ Stage 4 Audit Failed (Score: {audit_score}/10)\n\n"
                    f"{feedback}\n\n"
                    f"**Consistency Issues**:\n{consistency_info}\n\n"
                    f"**Logic Issues**:\n{logic_info}\n\n"
                    f"**Requirement**: Score must be >= 8 to pass. Please rigorously fix all consistency and logic issues."
                )

            return True, None

        except Exception as e:
            print(f"⚠️ Stage 4 Audit Error: {e}")
            return False, f"Stage 4 audit unavailable: {e}"

    def dual_model_final_gate(
        self,
        concept_design: dict,
        visual_spec: str,
        qa_result: dict,
        image_path: str,
        difficulty: str,
    ) -> dict:
        """Require unanimous approval from two independently configured verifiers."""
        verifier_prompt = (
            "You are an independent terminal verifier for a multimodal mathematics dataset. "
            "Inspect the actual image and compare it with the question, solution, answer, "
            "visual specification, and logic concept. Check four criteria: image quality "
            "without overlap or occlusion, text-image consistency, solution correctness, "
            "and answer verifiability. Reject any material defect. Return only JSON:\n"
            '{"pass": true, "score": 1-10, '
            '"image_quality": true, "text_image_consistency": true, '
            '"solution_correctness": true, "answer_verifiable": true, '
            '"feedback": "brief reason"}'
        )
        user_prompt = (
            f"Difficulty: {difficulty}\n\n"
            f"Concept design:\n{str(concept_design)[:4000]}\n\n"
            f"Visual specification:\n{str(visual_spec)[:4000]}\n\n"
            f"Question, solution, and answer:\n{str(qa_result)[:6000]}\n\n"
            "Verify the actual attached image and make an independent accept/reject decision."
        )

        if not FINAL_VERIFIER_MODELS[0] or not FINAL_VERIFIER_MODELS[1]:
            raise RuntimeError(
                "Two final verifier models must be configured through the runtime environment."
            )
        if FINAL_VERIFIER_MODELS[0] == FINAL_VERIFIER_MODELS[1]:
            raise RuntimeError("The two final verifier models must be different.")

        verdicts = []
        for verifier_model in FINAL_VERIFIER_MODELS:
            result = vision_completion(
                system_prompt=verifier_prompt,
                image_path=image_path,
                user_prompt=user_prompt,
                timeout=90.0,
                temperature=0.1,
                max_tokens=4096,
                model=verifier_model,
            )
            if not isinstance(result, dict):
                raise RuntimeError("A final verifier returned a non-JSON response.")

            score = float(result.get("score", 0))
            criteria_pass = all(
                bool(result.get(key, False))
                for key in (
                    "image_quality",
                    "text_image_consistency",
                    "solution_correctness",
                    "answer_verifiable",
                )
            )
            passed = bool(result.get("pass", False)) and criteria_pass and score >= 8
            verdicts.append(
                {
                    "model_configured": True,
                    "pass": passed,
                    "score": score,
                    "feedback": str(result.get("feedback", "")),
                }
            )

        passed = all(verdict["pass"] for verdict in verdicts)
        feedback = " | ".join(verdict["feedback"] for verdict in verdicts if verdict["feedback"])
        return {
            "pass_check": passed,
            "verifiers": verdicts,
            "feedback": feedback,
        }
