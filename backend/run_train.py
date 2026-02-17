#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import sys

result = subprocess.run([sys.executable, 'train_models.py'], cwd='.')
sys.exit(result.returncode)
