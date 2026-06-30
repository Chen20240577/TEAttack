# -*- coding: utf-8 -*-
import re


def code_tokenizer(text):
    return re.findall(r'\w+|[^\w\s]+', text)
