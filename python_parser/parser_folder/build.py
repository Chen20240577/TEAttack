# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from tree_sitter import Language

Language.build_library(
    # Store the library in the `build` directory
    'my-languages.so',

    # Include one or more languages
    [
        'tree-sitter-python',
        'tree-sitter-java',
        'tree-sitter-cpp',
        'tree-sitter-c',

        'tree-sitter-go',
        'tree-sitter-javascript',
        # 'tree-sitter-php',
        'tree-sitter-php/php',  # 修改路径
        'tree-sitter-ruby',
        'tree-sitter-c-sharp',

    ]
)
