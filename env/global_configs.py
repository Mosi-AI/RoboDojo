import os

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

ROOT_DIR = os.path.join(current_dir, "..")

BENCHMARK = "RoboDojo"
# RoboICL keeps simulator assets and evaluator output outside the RoboDojo
# checkout. Preserve the checkout-relative behavior when those environment
# variables are not set so standalone RoboDojo launches remain compatible.
DATA_ROOT = os.path.abspath(
    os.environ.get("ROBOICL_DATA_ROOT", os.path.join(ROOT_DIR, "data"))
)
ASSETS_PATH = os.path.join(DATA_ROOT, "Assets")
OBJECTS_PATH = os.path.join(ASSETS_PATH, "Object", BENCHMARK)
ROBOTS_PATH = os.path.join(ASSETS_PATH, "Robots")
EVAL_RESULT_PATH = os.path.join(
    os.environ.get("ROBOICL_RESULTS_ROOT", os.path.join(ROOT_DIR, "results")),
    "eval_result",
)
ENV_CONFIG_PATH = os.path.join(ROOT_DIR, "env_cfg")

ENV_REGEX_NAMESPACE = "{ENV_REGEX_NS}"

BATCH_NUM = 10

USD_PATH = os.environ.get("ROBODOJO_USD_ASSET_PREFIX") or None
