"""Internal component of the AgSynth generation pipeline."""

import math
from decimal import getcontext
from fractions import Fraction
from itertools import permutations, product
from typing import Any, Dict, List, Optional, Tuple, Union

getcontext().prec = 50


class MathValidator:
    """Internal component of the AgSynth generation pipeline."""

    def __init__(self):
        """Internal component of the AgSynth generation pipeline."""
        self.operators = ["+", "-", "*", "/"]
        self.tolerance = 1e-9

    def validate_24_point(
        self, numbers: List[int], target: int = 24, allow_duplicates: bool = True
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        if len(numbers) != 4:
            return {
                "has_solution": False,
                "solution_count": 0,
                "solutions": [],
                "sample_solutions": [],
                "error": "24-pointproblem requires exactly four numbers",
            }

        solutions = set()

        for perm in permutations(numbers):
            for ops in product(self.operators, repeat=3):
                expressions = self._generate_24_expressions(perm, ops)

                for expr in expressions:
                    try:
                        result = self._safe_eval(expr)
                        if result is not None and abs(result - target) < 1e-9:
                            solutions.add(expr)
                    except Exception:
                        continue

        solution_list = sorted(list(solutions))

        return {
            "has_solution": len(solutions) > 0,
            "solution_count": len(solutions),
            "solutions": solution_list,
            "sample_solutions": solution_list[:5],
        }

    def _generate_24_expressions(
        self, numbers: Tuple[int, int, int, int], ops: Tuple[str, str, str]
    ) -> List[str]:
        """Internal component of the AgSynth generation pipeline."""
        a, b, c, d = numbers
        op1, op2, op3 = ops

        patterns = [
            f"(({a} {op1} {b}) {op2} {c}) {op3} {d}",
            f"({a} {op1} ({b} {op2} {c})) {op3} {d}",
            f"({a} {op1} {b}) {op2} ({c} {op3} {d})",
            f"{a} {op1} (({b} {op2} {c}) {op3} {d})",
            f"{a} {op1} ({b} {op2} ({c} {op3} {d}))",
        ]

        return patterns

    def _safe_eval(self, expr: str) -> Optional[float]:
        """Internal component of the AgSynth generation pipeline."""
        try:
            if "/0" in expr.replace(" ", ""):
                return None

            result = eval(expr, {"__builtins__": {}}, {})
            return float(result)
        except (ZeroDivisionError, ValueError, SyntaxError):
            return None

    def validate_equation_system(
        self, equations: List[str], variables: List[str]
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        try:
            eq_count = len(equations)
            var_count = len(variables)

            if eq_count < var_count:
                return {
                    "has_solution": True,
                    "solution_type": "infinite",
                    "solutions": {},
                    "solution_count": -1,
                    "message": f"Equation count({eq_count}) < Variable count({var_count})，has infinitely many solutions",
                }
            elif eq_count > var_count:
                return {
                    "has_solution": False,
                    "solution_type": "overdetermined",
                    "solutions": {},
                    "solution_count": 0,
                    "message": f"Equation count({eq_count}) > Variable count({var_count})，may have no solution or may be inconsistent",
                }

            return {
                "has_solution": True,
                "solution_type": "unknown",
                "solutions": {},
                "solution_count": 1,
                "message": "requires a solver for verification (consider integrating SymPy)",
            }

        except Exception as e:
            return {
                "has_solution": False,
                "solution_type": "error",
                "solutions": {},
                "solution_count": 0,
                "error": str(e),
            }

    def validate_arithmetic_puzzle(self, puzzle: str, answer: Dict[str, int]) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        try:
            substituted = puzzle
            for letter, digit in answer.items():
                substituted = substituted.replace(letter, str(digit))

            if "=" in substituted:
                left, right = substituted.split("=")
                left_result = self._safe_eval(left.strip())
                right_result = self._safe_eval(right.strip())

                is_correct = (
                    left_result is not None
                    and right_result is not None
                    and abs(left_result - right_result) < 1e-9
                )

                return {
                    "is_correct": is_correct,
                    "calculated_result": f"{left_result} = {right_result}",
                    "expected_result": "both sides are equal",
                    "all_solutions": [],
                }
            else:
                return {
                    "is_correct": False,
                    "calculated_result": "",
                    "expected_result": "",
                    "error": "Alphametic format error: missing equals sign",
                }

        except Exception as e:
            return {
                "is_correct": False,
                "calculated_result": "",
                "expected_result": "",
                "error": str(e),
            }

    def validate_arithmetic_calculation(
        self, expression: str, expected_answer: Union[int, float, str], allow_tolerance: bool = True
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        try:
            calculated = self._safe_eval(expression)
            if calculated is None:
                return {
                    "is_correct": False,
                    "error": "Expression evaluation failed (possible division by zero or syntax error)",
                }

            if isinstance(expected_answer, str):
                expected = self._safe_eval(expected_answer)
            else:
                expected = float(expected_answer)

            if expected is None:
                return {"is_correct": False, "error": "Invalid expected-answer format"}

            difference = abs(calculated - expected)
            is_correct = difference < self.tolerance if allow_tolerance else calculated == expected

            return {
                "is_correct": is_correct,
                "calculated_result": calculated,
                "expected_result": expected,
                "difference": difference,
                "expression": expression,
                "message": "✅ Answer is correct"
                if is_correct
                else f"❌ Answer is incorrect（difference: {difference}）",
            }

        except Exception as e:
            return {"is_correct": False, "error": f"Validation failed: {str(e)}"}

    def validate_fraction_calculation(
        self,
        fraction1: Tuple[int, int],
        operator: str,
        fraction2: Tuple[int, int],
        expected_answer: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        try:
            f1 = Fraction(fraction1[0], fraction1[1])
            f2 = Fraction(fraction2[0], fraction2[1])
            expected = Fraction(expected_answer[0], expected_answer[1])

            if operator == "+":
                result = f1 + f2
            elif operator == "-":
                result = f1 - f2
            elif operator == "*":
                result = f1 * f2
            elif operator == "/":
                if f2 == 0:
                    return {"is_correct": False, "error": "Divisor cannot be zero"}
                result = f1 / f2
            else:
                return {"is_correct": False, "error": f"Unsupported operator: {operator}"}

            is_correct = result == expected

            return {
                "is_correct": is_correct,
                "calculated_result": f"{result.numerator}/{result.denominator}",
                "expected_result": f"{expected.numerator}/{expected.denominator}",
                "simplified_result": str(result),
                "message": "✅ Fraction calculation is correct"
                if is_correct
                else "❌ Fraction calculation is incorrect",
            }

        except Exception as e:
            return {
                "is_correct": False,
                "error": f"Fraction calculation validation failed: {str(e)}",
            }

    def validate_percentage_calculation(
        self, base_value: float, percentage: float, operation: str, expected_answer: float
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        try:
            if operation == "of":
                calculated = base_value * (percentage / 100)
            elif operation == "increase":
                calculated = base_value * (1 + percentage / 100)
            elif operation == "decrease":
                calculated = base_value * (1 - percentage / 100)
            elif operation == "is_what_percent":
                if base_value == 0:
                    return {"is_correct": False, "error": "Base cannot be zero"}
                calculated = (percentage / base_value) * 100
            else:
                return {"is_correct": False, "error": f"Unsupported operation type: {operation}"}

            difference = abs(calculated - expected_answer)
            is_correct = difference < self.tolerance

            return {
                "is_correct": is_correct,
                "calculated_result": calculated,
                "expected_result": expected_answer,
                "difference": difference,
                "operation": operation,
                "message": "✅ Percentage calculation is correct"
                if is_correct
                else f"❌ Percentage calculation is incorrect（difference: {difference}）",
            }

        except Exception as e:
            return {
                "is_correct": False,
                "error": f"Percentage calculation validation failed: {str(e)}",
            }

    def validate_ratio_problem(
        self,
        ratio1: Tuple[int, int],
        ratio2: Tuple[int, int],
        expected_answer: Union[Tuple[int, int], float],
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        try:
            value1 = ratio1[0] / ratio1[1] if ratio1[1] != 0 else None
            value2 = ratio2[0] / ratio2[1] if ratio2[1] != 0 else None

            if value1 is None or value2 is None:
                return {"is_correct": False, "error": "Ratio denominator cannot be zero"}

            if isinstance(expected_answer, tuple):
                expected_value = (
                    expected_answer[0] / expected_answer[1] if expected_answer[1] != 0 else None
                )
                if expected_value is None:
                    return {
                        "is_correct": False,
                        "error": "Expected-answer denominator cannot be zero",
                    }
                is_correct = (
                    abs(value1 - expected_value) < self.tolerance
                    or abs(value2 - expected_value) < self.tolerance
                )
            else:
                is_correct = (
                    abs(value1 - expected_answer) < self.tolerance
                    or abs(value2 - expected_answer) < self.tolerance
                )

            return {
                "is_correct": is_correct,
                "ratio1_value": value1,
                "ratio2_value": value2,
                "expected_answer": expected_answer,
                "message": "✅ Ratio calculation is correct"
                if is_correct
                else "❌ Ratio calculation is incorrect",
            }

        except Exception as e:
            return {"is_correct": False, "error": f"Ratio-problem validation failed: {str(e)}"}

    def validate_geometry_calculation(
        self,
        shape_type: str,
        dimensions: Dict[str, float],
        calculation_type: str,
        expected_answer: float,
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        try:
            calculated = None

            if shape_type == "rectangle":
                if calculation_type == "area":
                    calculated = dimensions.get("length", 0) * dimensions.get("width", 0)
                elif calculation_type == "perimeter":
                    calculated = 2 * (dimensions.get("length", 0) + dimensions.get("width", 0))

            elif shape_type == "circle":
                radius = dimensions.get("radius", 0)
                if calculation_type == "area":
                    calculated = math.pi * radius**2
                elif calculation_type == "perimeter":
                    calculated = 2 * math.pi * radius

            elif shape_type == "triangle":
                if calculation_type == "area":
                    base = dimensions.get("base", 0)
                    height = dimensions.get("height", 0)
                    calculated = 0.5 * base * height
                elif calculation_type == "perimeter":
                    calculated = (
                        dimensions.get("side1", 0)
                        + dimensions.get("side2", 0)
                        + dimensions.get("side3", 0)
                    )

            elif shape_type == "cube":
                side = dimensions.get("side", 0)
                if calculation_type == "volume":
                    calculated = side**3
                elif calculation_type == "surface_area":
                    calculated = 6 * side**2

            elif shape_type == "sphere":
                radius = dimensions.get("radius", 0)
                if calculation_type == "volume":
                    calculated = (4 / 3) * math.pi * radius**3
                elif calculation_type == "surface_area":
                    calculated = 4 * math.pi * radius**2

            else:
                return {"is_correct": False, "error": f"Unsupported shape type: {shape_type}"}

            if calculated is None:
                return {
                    "is_correct": False,
                    "error": f"Unsupported calculation type: {calculation_type}",
                }

            difference = abs(calculated - expected_answer)
            is_correct = difference < self.tolerance

            return {
                "is_correct": is_correct,
                "calculated_result": calculated,
                "expected_result": expected_answer,
                "difference": difference,
                "shape_type": shape_type,
                "calculation_type": calculation_type,
                "message": "✅ Geometry calculation is correct"
                if is_correct
                else f"❌ Geometry calculation is incorrect（difference: {difference}）",
            }

        except Exception as e:
            return {
                "is_correct": False,
                "error": f"Geometry calculation validation failed: {str(e)}",
            }

    def validate_sequence_pattern(
        self, sequence: List[Union[int, float]], pattern_type: str, expected_next: Union[int, float]
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        try:
            if len(sequence) < 2:
                return {"is_correct": False, "error": "A sequence requires at least two elements"}

            calculated_next = None

            if pattern_type == "arithmetic":
                diff = sequence[1] - sequence[0]
                is_arithmetic = all(
                    sequence[i + 1] - sequence[i] == diff for i in range(len(sequence) - 1)
                )
                if is_arithmetic:
                    calculated_next = sequence[-1] + diff

            elif pattern_type == "geometric":
                if sequence[0] == 0:
                    return {
                        "is_correct": False,
                        "error": "The first term of a geometric sequence cannot be zero",
                    }
                ratio = sequence[1] / sequence[0]
                is_geometric = all(
                    abs(sequence[i + 1] / sequence[i] - ratio) < self.tolerance
                    for i in range(len(sequence) - 1)
                    if sequence[i] != 0
                )
                if is_geometric:
                    calculated_next = sequence[-1] * ratio

            elif pattern_type == "fibonacci":
                is_fibonacci = all(
                    sequence[i] == sequence[i - 1] + sequence[i - 2]
                    for i in range(2, len(sequence))
                )
                if is_fibonacci:
                    calculated_next = sequence[-1] + sequence[-2]

            elif pattern_type == "custom":
                diff = sequence[1] - sequence[0]
                if all(
                    abs((sequence[i + 1] - sequence[i]) - diff) < self.tolerance
                    for i in range(len(sequence) - 1)
                ):
                    calculated_next = sequence[-1] + diff
                elif sequence[0] != 0:
                    ratio = sequence[1] / sequence[0]
                    if all(
                        abs(sequence[i + 1] / sequence[i] - ratio) < self.tolerance
                        for i in range(len(sequence) - 1)
                        if sequence[i] != 0
                    ):
                        calculated_next = sequence[-1] * ratio

            else:
                return {"is_correct": False, "error": f"Unsupported pattern type: {pattern_type}"}

            if calculated_next is None:
                return {"is_correct": False, "error": "Unable to identify the sequence pattern"}

            difference = abs(calculated_next - expected_next)
            is_correct = difference < self.tolerance

            return {
                "is_correct": is_correct,
                "calculated_next": calculated_next,
                "expected_next": expected_next,
                "difference": difference,
                "pattern_type": pattern_type,
                "message": "✅ Sequence pattern is correct"
                if is_correct
                else f"❌ Sequence pattern is incorrect（difference: {difference}）",
            }

        except Exception as e:
            return {"is_correct": False, "error": f"Sequence-pattern validation failed: {str(e)}"}

    def validate_comparison_problem(
        self,
        value1: Union[int, float],
        value2: Union[int, float],
        comparison_type: str,
        expected_result: bool,
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        try:
            calculated_result = None

            if comparison_type == "greater":
                calculated_result = value1 > value2
            elif comparison_type == "less":
                calculated_result = value1 < value2
            elif comparison_type == "equal":
                calculated_result = abs(value1 - value2) < self.tolerance
            elif comparison_type == "greater_equal":
                calculated_result = value1 >= value2 or abs(value1 - value2) < self.tolerance
            elif comparison_type == "less_equal":
                calculated_result = value1 <= value2 or abs(value1 - value2) < self.tolerance
            else:
                return {
                    "is_correct": False,
                    "error": f"Unsupported comparison type: {comparison_type}",
                }

            is_correct = calculated_result == expected_result

            return {
                "is_correct": is_correct,
                "calculated_result": calculated_result,
                "expected_result": expected_result,
                "value1": value1,
                "value2": value2,
                "comparison_type": comparison_type,
                "message": "✅ Comparison result is correct"
                if is_correct
                else "❌ Comparison result is incorrect",
            }

        except Exception as e:
            return {"is_correct": False, "error": f"Comparison-problem validation failed: {str(e)}"}

    def validate_math_problem(
        self, problem_type: str, problem_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        if problem_type == "24_point":
            return self.validate_24_point(
                numbers=problem_data.get("numbers", []), target=problem_data.get("target", 24)
            )

        elif problem_type == "equation_system":
            return self.validate_equation_system(
                equations=problem_data.get("equations", []),
                variables=problem_data.get("variables", []),
            )

        elif problem_type == "arithmetic_puzzle":
            return self.validate_arithmetic_puzzle(
                puzzle=problem_data.get("puzzle", ""), answer=problem_data.get("answer", {})
            )

        elif problem_type == "arithmetic":
            return self.validate_arithmetic_calculation(
                expression=problem_data.get("expression", ""),
                expected_answer=problem_data.get("expected_answer", 0),
                allow_tolerance=problem_data.get("allow_tolerance", True),
            )

        elif problem_type == "fraction":
            return self.validate_fraction_calculation(
                fraction1=problem_data.get("fraction1", (1, 1)),
                operator=problem_data.get("operator", "+"),
                fraction2=problem_data.get("fraction2", (1, 1)),
                expected_answer=problem_data.get("expected_answer", (1, 1)),
            )

        elif problem_type == "percentage":
            return self.validate_percentage_calculation(
                base_value=problem_data.get("base_value", 0),
                percentage=problem_data.get("percentage", 0),
                operation=problem_data.get("operation", "of"),
                expected_answer=problem_data.get("expected_answer", 0),
            )

        elif problem_type == "ratio":
            return self.validate_ratio_problem(
                ratio1=problem_data.get("ratio1", (1, 1)),
                ratio2=problem_data.get("ratio2", (1, 1)),
                expected_answer=problem_data.get("expected_answer", (1, 1)),
            )

        elif problem_type == "geometry":
            return self.validate_geometry_calculation(
                shape_type=problem_data.get("shape_type", "rectangle"),
                dimensions=problem_data.get("dimensions", {}),
                calculation_type=problem_data.get("calculation_type", "area"),
                expected_answer=problem_data.get("expected_answer", 0),
            )

        elif problem_type == "sequence":
            return self.validate_sequence_pattern(
                sequence=problem_data.get("sequence", []),
                pattern_type=problem_data.get("pattern_type", "arithmetic"),
                expected_next=problem_data.get("expected_next", 0),
            )

        elif problem_type == "comparison":
            return self.validate_comparison_problem(
                value1=problem_data.get("value1", 0),
                value2=problem_data.get("value2", 0),
                comparison_type=problem_data.get("comparison_type", "greater"),
                expected_result=problem_data.get("expected_result", True),
            )

        else:
            return {
                "error": f"Unsupported problem type: {problem_type}",
                "supported_types": [
                    "24_point",
                    "equation_system",
                    "arithmetic_puzzle",
                    "arithmetic",
                    "fraction",
                    "percentage",
                    "ratio",
                    "geometry",
                    "sequence",
                    "comparison",
                ],
            }

    def detect_multiple_solutions(
        self, problem_type: str, problem_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        validation_result = self.validate_math_problem(problem_type, problem_data)

        solution_count = validation_result.get("solution_count", 0)

        if solution_count == 0:
            recommendation = "❌ This problem has no solution. Redesign the data specifications."
        elif solution_count == 1:
            recommendation = "✅ This problem has a unique solution and meets the requirements."
        elif solution_count > 1:
            recommendation = (
                f"⚠️ This problem has  {solution_count}  solutions. Recommendations:\n"
                f"1. Convert it to a multiple-choice format\n"
                f"2. Require students to provide all solutions\n"
                f"3. Add constraints to make the solution unique"
            )
        else:
            recommendation = "⚠️ This problem has infinitely many solutions. Add more constraints."

        return {
            "has_multiple_solutions": solution_count > 1,
            "solution_count": solution_count,
            "recommendation": recommendation,
            "validation_result": validation_result,
        }


if __name__ == "__main__":
    validator = MathValidator()

    print("=" * 60)
    print("Example1：24-point validation")
    print("=" * 60)

    result = validator.validate_24_point([3, 3, 8, 8])
    print("Numbers: [3, 3, 8, 8]")
    print(f"Has solution: {result['has_solution']}")
    print(f"Solution count: {result['solution_count']}")
    print(f"Example solutions: {result['sample_solutions'][:3]}")

    print("\n" + "=" * 60)
    print("Example2：unsolvable 24-point case (distractor validation)")
    print("=" * 60)

    result = validator.validate_24_point([1, 1, 1, 1])
    print("Numbers: [1, 1, 1, 1]")
    print(f"Has solution: {result['has_solution']}")
    print(f"Solution count: {result['solution_count']}")

    print("\n" + "=" * 60)
    print("Example3：Multiple-solution check")
    print("=" * 60)

    result = validator.detect_multiple_solutions(
        problem_type="24_point", problem_data={"numbers": [3, 3, 8, 8], "target": 24}
    )
    print(f"Has multiple solutions: {result['has_multiple_solutions']}")
    print(f"Solution count: {result['solution_count']}")
    print(f"Recommendation: {result['recommendation']}")
