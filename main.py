import argparse
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional

from orchestrator import Orchestrator
from seed_manager import SeedManager
from utils import validate_runtime_config


IMAGE_DIR = ""
SEED_JSON_PATH = ""
OUTPUT_BASE_DIR = ""
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_BASE_DIR, "generated_images")
RESULT_JSON_PATH = os.path.join(OUTPUT_BASE_DIR, "generated_questions.jsonl")


_RESULT_FILE_LOCK = threading.Lock()


GENERATE_COUNT = 1


PER_QUESTION_MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0


MAX_CONCURRENT_WORKERS = 1

DIFFICULTY = "easy"


SEED_STATE_FILE = os.path.join(OUTPUT_BASE_DIR, "seed_state.json")


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for command-line options."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Build the public command-line interface for one generation run."""
    parser = argparse.ArgumentParser(
        description="Generate visually grounded mathematics data with the AgSynth pipeline."
    )
    parser.add_argument(
        "--seed-json", required=True, help="Path to seed data in JSON or JSONL format."
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing the seed images.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for generated images, JSONL records, and state (default: output).",
    )
    parser.add_argument(
        "--count", type=_positive_int, default=1, help="Accepted samples to generate (default: 1)."
    )
    parser.add_argument(
        "--workers", type=_positive_int, default=1, help="Concurrent workers (default: 1)."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries after a complete sample failure (default: 2).",
    )
    parser.add_argument(
        "--difficulty", choices=("easy", "hard"), default="easy", help="Target difficulty level."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate seed data and print the run plan without contacting a model or writing outputs.",
    )
    return parser.parse_args(argv)


def configure_runtime(args: argparse.Namespace) -> None:
    """Apply command-line configuration to the existing worker functions."""
    global IMAGE_DIR, SEED_JSON_PATH, OUTPUT_BASE_DIR, OUTPUT_IMAGE_DIR
    global RESULT_JSON_PATH, GENERATE_COUNT, MAX_CONCURRENT_WORKERS
    global PER_QUESTION_MAX_RETRIES, SEED_STATE_FILE, DIFFICULTY

    IMAGE_DIR = os.path.abspath(args.image_dir)
    SEED_JSON_PATH = os.path.abspath(args.seed_json)
    OUTPUT_BASE_DIR = os.path.abspath(args.output_dir)
    OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_BASE_DIR, "generated_images")
    RESULT_JSON_PATH = os.path.join(OUTPUT_BASE_DIR, "generated_questions.jsonl")
    SEED_STATE_FILE = os.path.join(OUTPUT_BASE_DIR, "seed_state.json")
    GENERATE_COUNT = args.count
    MAX_CONCURRENT_WORKERS = args.workers
    PER_QUESTION_MAX_RETRIES = max(0, args.retries)
    DIFFICULTY = args.difficulty


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_result_to_json(result_path: str, new_result: Dict[str, Any]) -> None:
    """Append one JSON object per line without silently discarding old records."""
    with _RESULT_FILE_LOCK:
        lock_file = result_path + ".lock"
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                lock_acquired = False
                for _ in range(10):
                    try:
                        with open(lock_file, "x") as lf:
                            lf.write(str(os.getpid()))
                        lock_acquired = True
                        break
                    except FileExistsError:
                        time.sleep(0.5)

                if not lock_acquired:
                    if attempt < max_retries - 1:
                        print("Lock busy, retrying...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise TimeoutError(f"Could not acquire result lock: {lock_file}")

                try:
                    existing_data = read_result_records(result_path)
                    if os.path.exists(result_path):
                        with open(result_path, "r", encoding="utf-8") as f:
                            existing_content = f.read().lstrip()
                        if existing_content.startswith("["):
                            with open(result_path, "w", encoding="utf-8") as f:
                                for record in existing_data:
                                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    with open(result_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(new_result, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

                    print(
                        f"Appended to {os.path.basename(result_path)} (total: {len(existing_data) + 1})"
                    )
                    break
                finally:
                    try:
                        if os.path.exists(lock_file):
                            os.remove(lock_file)
                    except OSError:
                        pass
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Append failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"Append failed (all retries exhausted): {e}")
                    raise


def read_result_records(result_path: str) -> List[Dict[str, Any]]:
    """Read JSONL output and accept the previous JSON-array format for migration."""
    if not os.path.exists(result_path):
        return []

    with open(result_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    if content.startswith("["):
        records = json.loads(content)
        if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
            raise ValueError("Existing result file must contain JSON objects.")
        return records

    records = []
    for line_number, line in enumerate(content.splitlines(), 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL record at line {line_number}: {e}") from e
        if not isinstance(item, dict):
            raise ValueError(f"JSONL record at line {line_number} is not an object.")
        records.append(item)
    return records


def resolve_image_path(image_dir: str, name_or_path: str) -> str:
    """Resolve image path from filename (with/without extension) or absolute path."""
    if os.path.isabs(name_or_path):
        return name_or_path

    name_or_path = str(name_or_path).strip()
    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    stem, ext = os.path.splitext(name_or_path)

    if ext:
        candidates = [name_or_path]
        if stem.isdigit():
            candidates.insert(0, f"img_{stem}{ext}")
    else:
        bases = [name_or_path]
        if name_or_path.startswith("img_"):
            bases.insert(0, name_or_path[4:])
        else:
            bases.insert(0, f"img_{name_or_path}")
        candidates = [base + candidate_ext for base in bases for candidate_ext in exts]

    for candidate in candidates:
        path = os.path.join(image_dir, candidate)
        if os.path.exists(path):
            return path

    return os.path.join(image_dir, candidates[0])


def normalize_solution_steps(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return "" if value is None else str(value)


def load_seed_questions(json_path: str, image_dir: str) -> List[Dict[str, Any]]:
    """Load seed questions from JSON or JSONL file.

    Supported formats:
    1. JSON: standard array [{"field": "value"}, ...]
    2. JSONL: one JSON object per line

    New format fields:
    - id: unique identifier (paired with img_<id>.png)
    - question: question text
    - answer: standard answer
    - knowledge-level1/2/3/4: knowledge classification
    - knowledge: knowledge description
    - principle: math principle
    - idx: question index

    Legacy format fields:
    - question_id: unique identifier
    - standard_answer: answer
    - human_sol / solution: solution steps
    - image_path: image filename
    - Name/ID/name: question identifier
    - Revised_Question: question text
    - Revised_Answer: answer
    - Solution steps: solution
    - Image composition: image description
    """
    print(f"Loading seed questions from {json_path}...")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Seed file not found: {json_path}")
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    file_ext = os.path.splitext(json_path)[1].lower()
    is_jsonl = file_ext in [".jsonl", ".ndjson"]

    raw_data = []

    if is_jsonl:
        print("Detected JSONL format, parsing line by line...")
        with open(json_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    raw_data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Skipping line {line_num}: JSON parse error - {e}")
                    continue
    else:
        print("Detected JSON array format, parsing...")
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError("Seed JSON top-level must be an array.")

    seed_questions: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_data, 1):
        if not isinstance(item, dict):
            print(f"Skipping entry {idx}: not a dict.")
            continue

        raw_id = (
            item.get("id")
            or item.get("question_id")
            or item.get("Name/ID")
            or item.get("Name")
            or item.get("ID")
            or item.get("name")
            or f"question_{idx}"
        )
        name_id = str(raw_id)

        if item.get("id") is not None and not item.get("image_path"):
            image_filename = f"img_{item['id']}.png"
        else:
            image_filename = (
                item.get("image_path")
                or item.get("Name/ID")
                or item.get("Name")
                or item.get("ID")
                or item.get("name")
                or name_id
            )
        image_path = resolve_image_path(image_dir, str(image_filename))
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Seed image for record {name_id} was not found: {image_path}")

        question_text = (
            item.get("question")
            or item.get("Revised_Question")
            or item.get("revised_question")
            or ""
        )

        answer_text = (
            item.get("answer")
            or item.get("standard_answer")
            or item.get("Revised_Answer")
            or item.get("revised_answer")
            or ""
        )

        solution_steps = (
            item.get("human_sol")
            or item.get("solution")
            or item.get("Solution steps")
            or item.get("solution_steps")
            or ""
        )
        solution_steps = normalize_solution_steps(solution_steps)

        image_composition = (
            item.get("Image composition")
            or item.get("image_composition")
            or item.get("Image Composition")
            or "(Not provided)"
        )

        knowledge_level1 = item.get("knowledge-level1") or ""
        knowledge_level2 = item.get("knowledge-level2") or ""
        knowledge_level3 = item.get("knowledge-level3") or ""
        knowledge_level4 = item.get("knowledge-level4") or ""
        knowledge = item.get("knowledge") or ""
        principle = item.get("principle") or ""
        item_idx = item.get("idx") or ""

        seed_questions.append(
            {
                "id": idx,
                "name": name_id,
                "image": image_path,
                "question": question_text,
                "answer": answer_text,
                "solution_steps": solution_steps,
                "image_composition": image_composition,
                "knowledge_level1": knowledge_level1,
                "knowledge_level2": knowledge_level2,
                "knowledge_level3": knowledge_level3,
                "knowledge_level4": knowledge_level4,
                "knowledge": knowledge,
                "principle": principle,
                "item_idx": item_idx,
            }
        )

    print(
        f"Loaded {len(seed_questions)} seed questions (format: {'JSONL' if is_jsonl else 'JSON'})"
    )
    return seed_questions


def get_next_available_index(output_image_dir: str) -> int:
    """Scan output dir to find next available image index."""
    if not os.path.exists(output_image_dir):
        return 1

    existing_indices = []
    for filename in os.listdir(output_image_dir):
        if filename.startswith("generated_question_") and filename.endswith(".png"):
            try:
                index_str = filename.replace("generated_question_", "").replace(".png", "")
                index = int(index_str)
                existing_indices.append(index)
            except ValueError:
                continue

    if not existing_indices:
        return 1

    return max(existing_indices) + 1


def run_one_question(
    i: int,
    seed_questions: List[Dict[str, Any]],
    output_image_dir: str,
    image_name: str,
    difficulty: str,
    result_json_path: str,
    seed_manager: Optional["SeedManager"] = None,
) -> Optional[Dict[str, Any]]:
    """Generate one accepted question and persist it immediately."""
    print(f"\n{'=' * 55}")
    print(f"Generating question #{i}")
    print(f"{'=' * 55}")

    if seed_manager is None:
        target_per_seed = GENERATE_COUNT // len(seed_questions)
        seed_manager = SeedManager(
            state_file=SEED_STATE_FILE, target_count_per_seed=target_per_seed
        )

    selected_seed = seed_manager.get_next_seed_to_generate(seed_questions)

    if selected_seed is None:
        print("All seeds reached target quota")
        print(f"Skipping question #{i}")
        return None

    single_seed = [selected_seed]

    seed_id = seed_manager.get_seed_id(selected_seed)
    current_success = seed_manager.seed_states.get(seed_id, {}).get("success_count", 0)
    target_per_seed = seed_manager.get_target_count(selected_seed)

    print(f"Selected seed: ID={seed_id}")
    print(f"Progress: {current_success}/{target_per_seed}")

    stats = seed_manager.get_seed_stats(selected_seed)
    print(
        f"Seed stats: attempts={stats['total_attempts']}, success={stats['success_count']}, rate={stats['success_rate']:.1%}"
    )

    orch = Orchestrator()

    try:
        result = orch.run(
            single_seed,
            output_image_dir,
            image_filename=image_name,
            difficulty=difficulty,
        )

    except Exception as e:
        failure_reason = str(e)[:200]
        seed_manager.record_failure(selected_seed, failure_reason)
        print(f"Seed '{seed_manager.get_seed_id(selected_seed)}' failed")

        raise

    qa_result = result.get("qa_result", {})

    solution_list = (
        qa_result.get("solution_steps", [])
        if isinstance(qa_result.get("solution_steps"), list)
        else str(qa_result.get("solution_steps", "")).splitlines()
    )
    relative_image = os.path.join(
        os.path.basename(OUTPUT_BASE_DIR),
        os.path.basename(output_image_dir),
        image_name,
    ).replace(os.sep, "/")
    question_text = qa_result.get("question_text", "")
    answer_text = qa_result.get("answer_text", "")
    formatted_result = {
        "image": relative_image,
        "problem": question_text,
        "standard_answer": answer_text,
        "detailed_solution": solution_list,
    }

    try:
        append_result_to_json(result_json_path, formatted_result)

        seed_manager.record_success(selected_seed)
        print(f"Seed '{seed_manager.get_seed_id(selected_seed)}' succeeded")
        print(f"Question #{i} saved")
    except Exception as e:
        print(f"Failed to save question #{i}: {e}")
        seed_manager.record_failure(selected_seed, f"result persistence failed: {e}")
        raise

    return {
        "image_name": image_name,
        "problem": question_text,
        "standard_answer": answer_text,
        "explanation": "\n".join(qa_result.get("solution_steps", []))
        if isinstance(qa_result.get("solution_steps"), list)
        else str(qa_result.get("solution_steps", "")),
        "difficulty": difficulty,
    }


def run_one_question_with_retry(
    i: int,
    seed_questions: List[Dict[str, Any]],
    output_image_dir: str,
    image_name: str,
    difficulty: str,
    result_json_path: str,
    max_retries: int = 1,
    seed_manager: Optional["SeedManager"] = None,
) -> Optional[Dict[str, Any]]:
    """Generate single question with retries (for concurrent execution)."""
    for attempt in range(1, max_retries + 2):
        try:
            result = run_one_question(
                i=i,
                seed_questions=seed_questions,
                output_image_dir=output_image_dir,
                image_name=image_name,
                difficulty=difficulty,
                result_json_path=result_json_path,
                seed_manager=seed_manager,
            )
            return result
        except Exception as e:
            if attempt < max_retries + 1:
                print(f"Question #{i} failed (attempt {attempt}/{max_retries + 1}): {e}")
                print(f"Retrying in {RETRY_BACKOFF_SECONDS}s...")
                time.sleep(RETRY_BACKOFF_SECONDS)
            else:
                print(f"Question #{i} failed after {max_retries + 1} attempts: {e}")

                image_full_path = os.path.join(output_image_dir, image_name)
                if os.path.exists(image_full_path):
                    try:
                        os.remove(image_full_path)
                        print(f"Deleted failed image: {image_name}")
                    except Exception as cleanup_err:
                        print(f"Image cleanup error: {cleanup_err}")
                return None

    return None


def main(argv: Optional[List[str]] = None) -> None:
    """Run the public AgSynth command-line workflow."""
    args = parse_args(argv)
    configure_runtime(args)

    print("Starting multi-agent question generation pipeline...")
    print("-" * 50)
    print(f"Image dir: {IMAGE_DIR}")
    print(f"Seed JSON: {SEED_JSON_PATH}")
    print(f"Output images: {OUTPUT_IMAGE_DIR}")
    print(f"Output JSONL: {RESULT_JSON_PATH}")

    print("-" * 50)

    seed_questions = load_seed_questions(SEED_JSON_PATH, IMAGE_DIR)
    if not seed_questions:
        raise ValueError("No valid seed questions were loaded; generation cannot continue.")

    if args.dry_run:
        print("\nDry run succeeded.")
        print(f"Validated seed records: {len(seed_questions)}")
        print(f"Would generate: {GENERATE_COUNT} accepted sample(s)")
        print(f"Output directory: {OUTPUT_BASE_DIR}")
        return

    validate_runtime_config()
    ensure_dir(OUTPUT_IMAGE_DIR)

    print("\n" + "=" * 60)
    print("Starting generation...")
    print("=" * 60 + "\n")

    print("\n" + "=" * 60)
    print("Initializing seed quota manager...")
    print("=" * 60)

    base_target, remainder = divmod(GENERATE_COUNT, len(seed_questions))
    target_counts = {}
    for position, seed in enumerate(seed_questions):
        seed_id = str(
            seed.get("name")
            or seed.get("Name")
            or seed.get("id")
            or seed.get("ID")
            or seed.get("image_path")
            or seed.get("image")
            or "unknown"
        )
        target_counts[seed_id] = base_target + (1 if position < remainder else 0)

    seed_manager = SeedManager(
        state_file=SEED_STATE_FILE,
        target_count_per_seed=base_target,
        target_counts=target_counts,
    )

    print(f"Base target per seed: {base_target}")
    print(f"Exact total target: {sum(target_counts.values())}")

    all_stats = seed_manager.get_all_stats()
    print("Seed manager initialized")
    print(f"  Total seeds: {all_stats['total_seeds']}")
    print(f"  Total attempts: {all_stats['total_attempts']}")
    print(f"  Total successes: {all_stats['total_successes']}")
    print(f"  Success rate: {all_stats['overall_success_rate']:.1%}")

    print("\nScanning output directory...")
    next_available_index = get_next_available_index(OUTPUT_IMAGE_DIR)

    existing_json_count = 0
    if os.path.exists(RESULT_JSON_PATH):
        try:
            existing_json_count = len(read_result_records(RESULT_JSON_PATH))
        except Exception as e:
            print(f"Failed to read existing results: {e}")

    print(f"Next available index: {next_available_index}")
    print(f"Existing results in JSON: {existing_json_count}")

    if GENERATE_COUNT < next_available_index:
        print(
            f"\nTarget reached ({next_available_index - 1} >= {GENERATE_COUNT}). No generation needed."
        )
        print(f"Increase GENERATE_COUNT (currently {GENERATE_COUNT}) to generate more.")
        return

    start_index = next_available_index
    target_end = GENERATE_COUNT

    questions_to_generate = list(range(start_index, target_end + 1))

    progress = seed_manager.get_generation_progress(seed_questions)
    print("\nCurrent progress:")
    print(f"  Target: {progress['total_target']}")
    print(f"  Generated: {progress['total_generated']}")
    print(f"  Progress: {progress['progress_percentage']:.1f}%")
    print(f"  Completed seeds: {progress['completed_seeds']}/{progress['total_seeds']}")

    print(f"\nStarting concurrent execution (workers={MAX_CONCURRENT_WORKERS})")
    print(f"Queue: {len(questions_to_generate)} questions (#{start_index} - #{target_end})\n")

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        question_iterator = iter(questions_to_generate)
        future_to_index = {}

        def submit_next() -> bool:
            try:
                i = next(question_iterator)
            except StopIteration:
                return False
            image_name = f"generated_question_{i}.png"
            future = executor.submit(
                run_one_question_with_retry,
                i=i,
                seed_questions=seed_questions,
                output_image_dir=OUTPUT_IMAGE_DIR,
                image_name=image_name,
                difficulty=DIFFICULTY,
                result_json_path=RESULT_JSON_PATH,
                max_retries=PER_QUESTION_MAX_RETRIES,
                seed_manager=seed_manager,
            )
            future_to_index[future] = i
            return True

        for _ in range(MAX_CONCURRENT_WORKERS):
            if not submit_next():
                break

        completed_count = 0
        failed_count = 0

        while future_to_index:
            done, _ = wait(future_to_index, return_when=FIRST_COMPLETED)
            for future in done:
                i = future_to_index.pop(future)
                try:
                    result = future.result()
                    if result is None:
                        print(f"Question #{i} skipped (all seeds at quota)")
                    elif result:
                        completed_count += 1
                        print(
                            f"Question #{i} done ({completed_count}/{len(questions_to_generate)})"
                        )
                        progress = seed_manager.get_generation_progress(seed_questions)
                        print(
                            f"Progress: {progress['total_generated']}/{progress['total_target']} ({progress['progress_percentage']:.1f}%)"
                        )
                    else:
                        failed_count += 1
                        print(f"Question #{i} failed (all retries exhausted)")
                except Exception as e:
                    failed_count += 1
                    print(f"Question #{i} error: {e}")
                submit_next()

    print(f"\n{'=' * 55}")
    print("Execution complete")
    print(f"{'=' * 55}")
    print(f"Succeeded: {completed_count}")
    print(f"Failed: {failed_count}")

    try:
        total_count = len(read_result_records(RESULT_JSON_PATH))
        print(f"Total in output file: {total_count}")
    except Exception as e:
        print(f"Failed to read output file: {e}")

    print("\n" + "=" * 60)
    print("Final progress:")
    print("=" * 60)

    final_progress = seed_manager.get_generation_progress(seed_questions)
    print("\nOverall:")
    print(f"  Target: {final_progress['total_target']}")
    print(f"  Generated: {final_progress['total_generated']}")
    print(f"  Progress: {final_progress['progress_percentage']:.1f}%")
    print(f"  Completed seeds: {final_progress['completed_seeds']}/{final_progress['total_seeds']}")

    print("\nPer-seed progress:")
    for seed_info in final_progress["seed_progress"]:
        status = "✅" if seed_info["completed"] else "⏳"
        print(f"   {status} {seed_info['seed_id']}: {seed_info['progress']}")

    print("=" * 60)

    print("\n" + "=" * 55)
    print("Generation complete!")
    print(f"Results saved to: {RESULT_JSON_PATH}")

    print("=" * 55)


if __name__ == "__main__":
    main()
