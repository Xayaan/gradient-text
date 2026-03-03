import os

import toml
import yaml
from fiber.logging_utils import get_logger

import core.constants as cst
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import TextDatasetType


logger = get_logger(__name__)


RESTRICTED_EXECUTION_HELPER = """def restricted_execution(code: str, input_data: str) -> tuple[str, str]:
    \"\"\"Execute Python code in a RestrictedPython sandbox.

    Returns:
        tuple[str, str]: (output, error)
    \"\"\"
    import contextlib
    import io

    try:
        from RestrictedPython import compile_restricted
        from RestrictedPython.Guards import safe_builtins
        from RestrictedPython.Guards import safe_globals
        from RestrictedPython.PrintCollector import PrintCollector
    except Exception as exc:
        return "", f"RestrictedPython unavailable: {exc}"

    stderr_capture = io.StringIO()

    try:
        compiled_code = compile_restricted(code, "<string>", "exec")
        if compiled_code is None:
            return "", "Failed to compile restricted code"

        restricted_builtins = safe_builtins.copy()
        restricted_builtins.update(
            {
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "reversed": reversed,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
            }
        )

        input_lines = input_data.split("\\n") if input_data else []

        def create_input_func(lines):
            lines_iter = iter(lines)

            def input_func(prompt=""):
                try:
                    return next(lines_iter)
                except StopIteration:
                    return ""

            return input_func

        restricted_globals = {
            "__builtins__": restricted_builtins,
            "_print_": PrintCollector,
            "_getattr_": getattr,
            "_getitem_": lambda obj, key: obj[key],
            "_getiter_": iter,
            "input": create_input_func(input_lines),
            "sum": sum,
            "min": min,
            "max": max,
            "enumerate": enumerate,
            "map": map,
            "filter": filter,
            "list": list,
            "dict": dict,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "len": len,
            "range": range,
        }
        restricted_globals.update(safe_globals)

        local_vars = {}

        with contextlib.redirect_stderr(stderr_capture):
            exec(compiled_code, restricted_globals, local_vars)

        print_collector = local_vars.get("_print")
        if print_collector and hasattr(print_collector, "txt"):
            output = "\\n".join(str(item) for item in print_collector.txt)
        else:
            output = ""

        error = stderr_capture.getvalue()
        return output, error

    except Exception as exc:
        return "", str(exc)
"""


def create_dataset_entry(
    dataset: str,
    dataset_type: TextDatasetType,
    file_format: FileFormat,
    is_eval: bool = False,
) -> dict:
    dataset_entry = {"path": dataset}

    logger.info(dataset_type)

    if file_format == FileFormat.JSON:
        if not is_eval:
            dataset_entry = {"path": "/workspace/input_data/"}
        else:
            dataset_entry = {"path": f"/workspace/input_data/{os.path.basename(dataset)}"}

    if isinstance(dataset_type, EnvironmentDatasetType):
        dataset_entry.update(_process_environment_dataset_fields(dataset_type))
    elif isinstance(dataset_type, InstructTextDatasetType):
        instruct_type_dict = {key: value for key, value in dataset_type.model_dump().items() if value is not None}
        dataset_entry.update(_process_instruct_dataset_fields(instruct_type_dict))
    elif isinstance(dataset_type, DpoDatasetType):
        dataset_entry.update(_process_dpo_dataset_fields(dataset_type))
    elif isinstance(dataset_type, GrpoDatasetType):
        dataset_entry.update(_process_grpo_dataset_fields(dataset_type))
    elif isinstance(dataset_type, ChatTemplateDatasetType):
        dataset_entry.update(_process_chat_template_dataset_fields(dataset_type))
    else:
        raise ValueError("Invalid dataset_type provided.")

    if file_format != FileFormat.HF:
        dataset_entry["ds_type"] = file_format.value
        dataset_entry["data_files"] = [os.path.basename(dataset)]

    return dataset_entry


def update_flash_attention(config: dict, model: str):
    # You might want to make this model-dependent
    config["flash_attention"] = False
    return config


def save_config(config: dict, config_path: str):
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def save_config_toml(config: dict, config_path: str):
    with open(config_path, "w") as file:
        toml.dump(config, file)


def _process_grpo_dataset_fields(dataset_type: GrpoDatasetType) -> dict:
    return {"split": "train"}


def _process_environment_dataset_fields(dataset_type: EnvironmentDatasetType) -> dict:
    return {"split": "train"}


def _process_dpo_dataset_fields(dataset_type: DpoDatasetType) -> dict:
    # Enable below when https://github.com/axolotl-ai-cloud/axolotl/issues/1417 is fixed
    # context: https://discord.com/channels/1272221995400167588/1355226588178022452/1356982842374226125

    # dpo_type_dict = dataset_type.model_dump()
    # dpo_type_dict["type"] = "user_defined.default"
    # if not dpo_type_dict.get("prompt_format"):
    #     if dpo_type_dict.get("field_system"):
    #         dpo_type_dict["prompt_format"] = "{system} {prompt}"
    #     else:
    #         dpo_type_dict["prompt_format"] = "{prompt}"
    # return dpo_type_dict

    # Fallback to https://axolotl-ai-cloud.github.io/axolotl/docs/rlhf.html#chatml.intel
    # Column names are hardcoded in axolotl: "DPO_DEFAULT_FIELD_SYSTEM",
    # "DPO_DEFAULT_FIELD_PROMPT", "DPO_DEFAULT_FIELD_CHOSEN", "DPO_DEFAULT_FIELD_REJECTED"
    return {"type": cst.DPO_DEFAULT_DATASET_TYPE, "split": "train"}


def _process_instruct_dataset_fields(instruct_type_dict: dict) -> dict:
    if not instruct_type_dict.get("field_output"):
        return {
            "type": "completion",
            "field": instruct_type_dict.get("field_instruction"),
        }

    processed_dict = instruct_type_dict.copy()
    processed_dict.setdefault("no_input_format", "{instruction}")
    if processed_dict.get("field_input"):
        processed_dict.setdefault("format", "{instruction} {input}")
    else:
        processed_dict.setdefault("format", "{instruction}")

    return {"format": "custom", "type": processed_dict}


def _process_chat_template_dataset_fields(dataset_dict: dict) -> dict:
    processed_dict = {}

    processed_dict["chat_template"] = dataset_dict.chat_template
    processed_dict["type"] = "chat_template"
    processed_dict["field_messages"] = dataset_dict.chat_column
    processed_dict["message_field_role"] = dataset_dict.chat_role_field
    processed_dict["message_field_content"] = dataset_dict.chat_content_field
    processed_dict["roles"] = {
        "assistant": [dataset_dict.chat_assistant_reference],
        "user": [dataset_dict.chat_user_reference],
    }

    processed_dict["message_property_mappings"] = {
        "role": dataset_dict.chat_role_field,
        "content": dataset_dict.chat_content_field
    }

    return processed_dict


def create_reward_funcs_file(reward_funcs: list[str], task_id: str, destination_dir: str = cst.CONFIG_DIR) -> list[str]:
    """
    Create a Python file with reward functions for GRPO training.
    Args:
        reward_funcs: List of strings containing Python reward function implementations
        task_id: Unique task identifier
    """
    filename = f"rewards_{task_id}"
    filepath = os.path.join(destination_dir, f"{filename}.py")

    func_names = []
    for reward_func in reward_funcs:
        if "def " in reward_func:
            func_name = reward_func.split("def ")[1].split("(")[0].strip()
            func_names.append(func_name)

    needs_restricted_execution = any("restricted_execution" in reward_func for reward_func in reward_funcs)
    defines_restricted_execution = any("def restricted_execution" in reward_func for reward_func in reward_funcs)

    with open(filepath, "w") as f:
        f.write("# Auto-generated reward functions file\n\n")
        if needs_restricted_execution and not defines_restricted_execution:
            f.write(RESTRICTED_EXECUTION_HELPER)
            f.write("\n\n")
        for reward_func in reward_funcs:
            f.write(f"{reward_func}\n\n")

    return filename, func_names

