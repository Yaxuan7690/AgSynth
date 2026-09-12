"""Internal component of the AgSynth generation pipeline."""

import re
from typing import Any, Dict, List, Optional, Tuple


class BoardValidator:
    """Internal component of the AgSynth generation pipeline."""

    def __init__(self):
        """Internal component of the AgSynth generation pipeline."""
        self.board_symbols = {
            "black": ["B", "X", "●", "○", "1"],
            "white": ["W", "O", "◯", "◎", "2"],
            "empty": [".", "-", "_", " ", "0"],
        }

    def parse_ascii_board(self, ascii_board: str) -> Tuple[List[List[str]], Dict[str, Any]]:
        """Internal component of the AgSynth generation pipeline."""
        lines = [line.strip() for line in ascii_board.strip().split("\n") if line.strip()]

        board_lines = []
        for line in lines:
            if any(symbol in line for symbols in self.board_symbols.values() for symbol in symbols):
                cleaned = re.sub(r"^\d+\s*", "", line)
                board_lines.append(cleaned)

        if not board_lines:
            return [], {"error": "Unable to parse ASCII board: no valid board data found"}

        board_matrix = []
        for line in board_lines:
            row = []
            chars = [c for c in line if c not in [" ", "\t", "|"]]
            for char in chars:
                if char in self.board_symbols["black"]:
                    row.append("B")
                elif char in self.board_symbols["white"]:
                    row.append("W")
                elif char in self.board_symbols["empty"]:
                    row.append(".")
                else:
                    row.append(".")

            if row:
                board_matrix.append(row)

        rows = len(board_matrix)
        cols = len(board_matrix[0]) if board_matrix else 0
        black_count = sum(row.count("B") for row in board_matrix)
        white_count = sum(row.count("W") for row in board_matrix)

        metadata = {
            "rows": rows,
            "cols": cols,
            "black_count": black_count,
            "white_count": white_count,
            "total_pieces": black_count + white_count,
        }

        return board_matrix, metadata

    def validate_ascii_board(
        self,
        ascii_board: str,
        expected_black: Optional[int] = None,
        expected_white: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        board_matrix, metadata = self.parse_ascii_board(ascii_board)

        errors = []
        warnings = []

        if not board_matrix:
            errors.append("Board parsing failed or returned no data")
            return {
                "is_valid": False,
                "board_matrix": [],
                "metadata": metadata,
                "errors": errors,
                "warnings": warnings,
            }

        row_lengths = [len(row) for row in board_matrix]
        if len(set(row_lengths)) > 1:
            errors.append(f"Board is not rectangular: row lengths are inconsistent {row_lengths}")

        if expected_black is not None and metadata["black_count"] != expected_black:
            errors.append(
                f"Black-piece count mismatch: expected{expected_black},actual{metadata['black_count']}"
            )

        if expected_white is not None and metadata["white_count"] != expected_white:
            errors.append(
                f"White-piece count mismatch: expected{expected_white},actual{metadata['white_count']}"
            )

        if metadata["rows"] < 3 or metadata["cols"] < 3:
            warnings.append(f"Board size too small: {metadata['rows']}x{metadata['cols']}")

        if metadata["rows"] > 20 or metadata["cols"] > 20:
            warnings.append(f"Board size too large: {metadata['rows']}x{metadata['cols']}")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "board_matrix": board_matrix,
            "metadata": metadata,
            "errors": errors,
            "warnings": warnings,
        }

    def check_winning_line(
        self, board_matrix: List[List[str]], player: str, required_length: int = 5
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """Internal component of the AgSynth generation pipeline."""
        if not board_matrix:
            return False, []

        rows = len(board_matrix)
        cols = len(board_matrix[0])

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(rows):
            for c in range(cols):
                if board_matrix[r][c] != player:
                    continue

                for dr, dc in directions:
                    positions = [(r, c)]

                    for step in range(1, required_length):
                        nr, nc = r + dr * step, c + dc * step

                        if not (0 <= nr < rows and 0 <= nc < cols):
                            break

                        if board_matrix[nr][nc] != player:
                            break

                        positions.append((nr, nc))

                    if len(positions) == required_length:
                        return True, positions

        return False, []

    def validate_board_logic(
        self,
        ascii_board: str,
        expected_winner: Optional[str] = None,
        expected_no_winner: bool = False,
        required_length: int = 5,
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        validation = self.validate_ascii_board(ascii_board)
        if not validation["is_valid"]:
            return {
                "is_valid": False,
                "actual_winner": None,
                "black_winning_line": [],
                "white_winning_line": [],
                "errors": validation["errors"],
                "reflection_prompt": self._generate_reflection_prompt(validation["errors"]),
            }

        board_matrix = validation["board_matrix"]
        errors = []

        black_wins, black_line = self.check_winning_line(board_matrix, "B", required_length)

        white_wins, white_line = self.check_winning_line(board_matrix, "W", required_length)

        actual_winner = None
        if black_wins and white_wins:
            errors.append("Logic error: both sides cannot win simultaneously")
            actual_winner = "BOTH"
        elif black_wins:
            actual_winner = "B"
        elif white_wins:
            actual_winner = "W"

        if expected_winner is not None:
            if actual_winner != expected_winner:
                errors.append(
                    f"Winner mismatch: expected {expected_winner}, actual {actual_winner or 'none'}"
                )

        if expected_no_winner and actual_winner is not None:
            errors.append(f"Expected no winner, but the actual winner is {actual_winner}")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "actual_winner": actual_winner,
            "black_winning_line": black_line,
            "white_winning_line": white_line,
            "errors": errors,
            "reflection_prompt": self._generate_reflection_prompt(errors) if errors else "",
        }

    def _generate_reflection_prompt(self, errors: List[str]) -> str:
        """Internal component of the AgSynth generation pipeline."""
        if not errors:
            return ""

        reflection = "## 🔍 Python validation found the following issues:\n\n"

        for i, error in enumerate(errors, 1):
            reflection += f"{i}. {error}\n"

        reflection += "\n## 💡 Revision suggestions:\n\n"

        for error in errors:
            if "Winner mismatch" in error:
                reflection += "- **Redraw the board** and verify which side forms a winning line.\n"
                reflection += "- **Check the winning path** one coordinate at a time.\n"

            elif "both sides cannot win" in error:
                reflection += "- **Remove the extra winning line** so only one side can win.\n"
                reflection += "- **Adjust piece positions** to break one winning line.\n"

            elif "count mismatch" in error:
                reflection += "- **Recount the pieces** in the ASCII board.\n"
                reflection += "- **Update data specifications** to match the ASCII board.\n"

            elif "not rectangular" in error:
                reflection += (
                    "- **Check the ASCII board format** and use the same column count per row.\n"
                )
                reflection += "- **Fill missing cells** with '.'.\n"

            elif "Expected no winner, but the actual winner is" in error:
                reflection += (
                    "- **Break the winning line** with an opposing piece or an empty cell.\n"
                )
                reflection += "- **Redesign the board** so neither side forms a line.\n"

        reflection += "\n## 🎯 Next steps:\n\n"
        reflection += "1. Draw the ASCII board ('B' = black, 'W' = white, '.' = empty).\n"
        reflection += "2. Mark the suspected winning line on the ASCII board.\n"
        reflection += "3. Check that the line coordinates are continuous.\n"
        reflection += "4. Revise the complete concept_design after correcting the error.\n"

        return reflection

    def generate_reflection_for_agent(
        self, concept_design: dict, validation_result: Dict[str, Any]
    ) -> str:
        """Internal component of the AgSynth generation pipeline."""
        if validation_result.get("is_valid"):
            return "✅ Board-logic validation passed! No revision is required."

        reflection = "## ❌ Board logic validation failed\n\n"

        reflection += "### 📋 Original design:\n\n"
        reflection += (
            f"**Logic kernel**: {concept_design.get('core_logic_kernel', 'N/A')[:200]}\n\n"
        )
        reflection += f"**Data specifications**: {concept_design.get('concept_design', {}).get('data_specifications', 'N/A')[:300]}\n\n"

        reflection += validation_result.get("reflection_prompt", "")

        reflection += "\n## 📝 Correct ASCII board example:\n\n"
        reflection += "```\n"
        reflection += "  1 2 3 4 5\n"
        reflection += "1 . . B . .\n"
        reflection += "2 . . B . .\n"
        reflection += "3 . . B . .\n"
        reflection += "4 . . B . .\n"
        reflection += "5 . . B . .  <- black pieces at (1,3)-(5,3) form a vertical line\n"
        reflection += "```\n\n"

        reflection += "**Key points**:\n"
        reflection += "- Coordinates start at (1,1), not (0,0).\n"
        reflection += (
            "- A winning line must be continuous horizontally, vertically, or diagonally.\n"
        )
        reflection += "- Draw the ASCII board before extracting coordinates to avoid spatial hallucinations.\n"

        return reflection

    def generate_board_creation_code(
        self,
        board_size: Tuple[int, int],
        black_positions: List[Tuple[int, int]],
        white_positions: List[Tuple[int, int]],
    ) -> str:
        """Internal component of the AgSynth generation pipeline."""
        code = f"""# Board-generation code
def create_board():
    rows, cols = {board_size}
    board = [['.' for _ in range(cols)] for _ in range(rows)]

    # Place black pieces
    black_positions = {black_positions}
    for r, c in black_positions:
        board[r][c] = 'B'

    # Place white pieces
    white_positions = {white_positions}
    for r, c in white_positions:
        board[r][c] = 'W'

    return board

def print_board(board):
    print("  " + " ".join(str(i+1) for i in range(len(board[0]))))
    for i, row in enumerate(board):
        print(f"{{i+1}} " + " ".join(row))

# Execute
board = create_board()
print_board(board)
"""
        return code

    def suggest_valid_winning_positions(
        self, board_size: Tuple[int, int], required_length: int = 5, direction: str = "horizontal"
    ) -> List[Tuple[int, int]]:
        """Internal component of the AgSynth generation pipeline."""
        rows, cols = board_size

        if direction == "horizontal":
            mid_row = rows // 2
            if cols >= required_length:
                start_col = (cols - required_length) // 2
                return [(mid_row, start_col + i) for i in range(required_length)]

        elif direction == "vertical":
            mid_col = cols // 2
            if rows >= required_length:
                start_row = (rows - required_length) // 2
                return [(start_row + i, mid_col) for i in range(required_length)]

        elif direction == "diagonal":
            if rows >= required_length and cols >= required_length:
                start_row = (rows - required_length) // 2
                start_col = (cols - required_length) // 2
                return [(start_row + i, start_col + i) for i in range(required_length)]

        return []

    def validate_concept_with_reflection(
        self, concept_design: dict, expected_winner: Optional[str] = None, required_length: int = 5
    ) -> Dict[str, Any]:
        """Internal component of the AgSynth generation pipeline."""
        ascii_board = concept_design.get("ascii_board_check", "")

        if not ascii_board:
            return {
                "is_valid": False,
                "validation_result": {},
                "reflection_prompt": "❌ ASCII board self-check is missing. Draw the ASCII board before extracting coordinates.",
                "suggested_fixes": {
                    "action": "add_ascii_board",
                    "instruction": "Add an 'ascii_board_check' field to concept_design and draw the complete ASCII board",
                },
            }

        validation_result = self.validate_board_logic(
            ascii_board=ascii_board,
            expected_winner=expected_winner,
            required_length=required_length,
        )

        reflection_prompt = self.generate_reflection_for_agent(
            concept_design=concept_design, validation_result=validation_result
        )

        suggested_fixes = {}
        if not validation_result["is_valid"]:
            suggested_fixes = {
                "action": "fix_board_logic",
                "errors": validation_result["errors"],
                "reflection": reflection_prompt,
            }

        return {
            "is_valid": validation_result["is_valid"],
            "validation_result": validation_result,
            "reflection_prompt": reflection_prompt,
            "suggested_fixes": suggested_fixes,
        }


if __name__ == "__main__":
    validator = BoardValidator()

    print("=" * 60)
    print("Example1:parse ASCII board")
    print("=" * 60)

    ascii_board = """
      1 2 3 4 5
    1 . . B . .
    2 . . B . .
    3 . W B . .
    4 . . B . .
    5 . . B . .
    """

    validation = validator.validate_ascii_board(ascii_board)
    print(f"Is valid: {validation['is_valid']}")
    print(f"Board size: {validation['metadata']['rows']}x{validation['metadata']['cols']}")
    print(f"Black-piece count: {validation['metadata']['black_count']}")
    print(f"White-piece count: {validation['metadata']['white_count']}")

    print("\n" + "=" * 60)
    print("Example2:verify winning logic")
    print("=" * 60)

    logic_validation = validator.validate_board_logic(
        ascii_board=ascii_board, expected_winner="B", required_length=5
    )

    print(f"Is logic valid: {logic_validation['is_valid']}")
    print(f"Actual winner: {logic_validation['actual_winner']}")
    print(f"Black winning line: {logic_validation['black_winning_line']}")

    if not logic_validation["is_valid"]:
        print("\nRevision feedback:")
        print(logic_validation["reflection_prompt"])

    print("\n" + "=" * 60)
    print("Example3:generate valid winning positions")
    print("=" * 60)

    positions = validator.suggest_valid_winning_positions(
        board_size=(10, 10), required_length=5, direction="diagonal"
    )
    print(f"Suggested diagonal winning positions: {positions}")
