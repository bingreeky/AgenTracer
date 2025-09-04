import base64
from sandbox_fusion import (
    RunCodeRequest,
    run_code,
    run_code_async,
    set_dataset_endpoint,
    set_sandbox_endpoint,
    TestConfig,
    SubmitRequest,
    submit,
    submit_async,
)

# set_sandbox_endpoint("https://faas-code-sandbox.bytedance.net/")
# set_dataset_endpoint("https://faas-code-sandbox.bytedance.net/online_judge/")
set_sandbox_endpoint("http://localhost:8080/")
set_dataset_endpoint("http://localhost:8080/online_judge/")


def code_exec(code: str, test: str):
    base64_content = base64.b64encode(code.encode("utf-8")).decode("utf-8")
    request = RunCodeRequest(
        language="pytest",
        code=test,
        files={"solution.py": base64_content},
    )
    return run_code(request)


async def code_exec_async(code: str, test: str):
    import logging

    base64_content = base64.b64encode(code.encode("utf-8")).decode("utf-8")
    request = RunCodeRequest(
        language="pytest",
        code=test,
        files={"solution.py": base64_content},
    )
    try:
        return await run_code_async(request, client_timeout=60)
    except Exception as e:
        # Log exception details
        logging.error(f"Error in code_exec_async: {e}", exc_info=True)
        # Depending on needs, you can return a custom error object or None
        print(f"Error in code_exec_async: {e}")
        return None


def oj_code_exec(code: str, test_case: tuple):
    test_data_formatted = []
    test_data_formatted.append(
        {
            "type": "StandardIO",
            "data": {"input": test_case["input"], "output": test_case["output"]},
        }
    )
    config = TestConfig(
        dataset_type="CommonOJDataset",
        language="python",
        run_timeout=60,
        provided_data={
            "test": {
                "location": "Embeded",
                "data": test_data_formatted,
            }
        },
    )

    request = SubmitRequest(dataset="oj", id="oj", completion=code, config=config)
    return submit(request)


async def oj_code_exec_async(code: str, test_case: tuple):
    test_data_formatted = []
    test_data_formatted.append(
        {
            "type": "StandardIO",
            "data": {"input": test_case["input"], "output": test_case["output"]},
        }
    )
    config = TestConfig(
        dataset_type="StandardIO",
        language="python",
        run_timeout=60,
        provided_data={
            "test": {
                "location": "Embeded",
                "data": test_data_formatted,
            }
        },
    )

    request = SubmitRequest(dataset="oj", id="oj", completion=code, config=config)
    return submit_async(request)


def eval_solution(solution: str, task: dict):
    pass
