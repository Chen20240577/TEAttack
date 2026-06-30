#!/bin/bash
#!/bin/bash

# 执行第一个 Python 脚本
# 等待第一个脚本完成后，再执行第二个
# 如果还有其他脚本，可以继续添加
# nohup ./run_scripts.sh > run_scripts.log 2>&1 &
# tail -f run_scripts.log


python fine-tuning.py

