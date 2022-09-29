from logging import root
import os, sys

from pathlib import Path

ROOT_DIR = os.path.abspath(os.curdir)

SUBSCRIPTION_KEY = "fa6da66b31e94018abac3eff8316027a"
CUSTOMER_ID = "2096"
BASE = 'https://api.powerfactorscorp.com/drive/v2'

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def mapper(a,b):
    list_compile = lambda a,b: a + '-' +b
    return list(map(list_compile, a,b))




   