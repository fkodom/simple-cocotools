import logging
import os
from subprocess import getoutput


def get_version_tag() -> str:
    try:
        env_key = "TIKIT_VERSION".upper()
        version = os.environ[env_key]
    except KeyError:
        version = getoutput("git describe --tags --abbrev=0")

    if version.lower().startswith("fatal"):
        version = "0.0.0"

    return version


VERSION = get_version_tag()

try:
    from simple_cocotools.evaluator import CocoEvaluator  # noqa: F401
except ModuleNotFoundError:
    logging.warning(
        "Could not import 'CocoEvaluator' from 'simple_cocotools'.  If you are "
        "installing simple-cocotools for the first time, this is not an issue. "
        "Otherwise, please check your installation."
    )
